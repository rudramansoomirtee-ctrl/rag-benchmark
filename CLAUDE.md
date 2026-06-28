# CLAUDE.md

Repo memory for Claude Code sessions. Read this before editing.

## What this is

Reproducible Dockerised benchmark of RAG systems (A naive, B agentic, F decomposition)
against MultiHop-RAG and MuSiQue, with every metric persisted to Postgres
and every span traced to Phoenix. Single-user research tool for a dissertation
— **not** a production service.

System E (vendored OpenRag) has been **removed** from the project. It depended on
OpenAI/Cohere external APIs that are not part of the dissertation's working setup;
the SOTA reference role it played is now handled by citation (Ammann et al. 2025;
OpenRag paper) rather than an in-repo run. Historical E runs remain in the DB.

## Stack (4 containers, `docker-compose.yml`)

| Service       | Port | Role                                          |
|---------------|------|-----------------------------------------------|
| `opensearch`  | 9200 | Dense + BM25 retrieval (HNSW / Lucene engine) |
| `postgres`    | 5432 | Structured runs/metrics — source of truth     |
| `phoenix`     | 6006 | OTel trace store + dashboard                  |
| `api`         | 8000 | FastAPI + LangGraph + HHEM + CLI + SPA        |

LLM is **AWS Bedrock** (Claude Haiku 4.5) via **LiteLLM**. Cost is read from
`response._hidden_params["response_cost"]` and persisted per run.

## Repo layout

```
api/src/
├── main.py                     # FastAPI app, lifespan tracing, serves /
├── cli.py                      # Typer commands (the primary surface)
├── config.py                   # pydantic-settings — single config source
├── tracing.py                  # Phoenix register, idempotent
├── api/
│   ├── routes/
│   │   ├── health.py           # GET /health
│   │   ├── ask.py              # POST /api/ask  (try one query)
│   │   ├── data.py             # GET /api/datasets, POST /api/ingest|index/{ds}
│   │   └── experiments.py      # GET /api/experiments[/{id}/runs|metrics]
│   └── static/index.html       # Vanilla HTML+JS SPA, three tabs
├── db/
│   ├── models.py               # 5 SQLAlchemy tables
│   ├── session.py              # engine + SessionLocal
│   └── migrations/             # Alembic (0001_initial, 0002_secondary_metrics)
├── datasets/{multihop,musique}.py
├── evaluation/{runner,metrics,judge}.py
├── faithfulness/hhem.py        # vectara/hallucination_evaluation_model
├── llm/client.py               # LiteLLM wrapper
├── retrieval/{embeddings,opensearch_client,indexer,retrieve}.py
└── systems/{base,schemas,system_a,system_b,system_f}.py
notebooks/analysis.py           # Marimo notebook for Chapter 4 figures
```

## The systems

Lineup: **A** (naive), **A-minus** (naive over a dense-kNN **semantic-search-only**
retriever — no BM25/RRF/rerank; A-minus-vs-A isolates the retrieval-pipeline
effect), **B** (agentic single-tool loop), **F** (PARALLEL query decomposition),
**F-seq** (SEQUENTIAL self-ask decomposition — resolves each hop and carries the
bridge answer forward into the next; shares F's few-shot decomposer).
Faithfulness (HHEM) is computed for every run, not by a system
— the old passive Systems C/D were folded into that metric. System E (vendored
OpenRag) was removed; see top-of-file note.

**System F-tuned was removed** (replaced by F-seq) after a chunk-level analysis of
exp36/37 (DeepSeek-V3, MuSiQue) showed the real multi-hop lever is *sequential
bridge resolution*, not F-tuned's parallel reranking/source-aware levers. F-tuned's
residual deltas vs F (per-query top-10 pools, weighted RRF, source fan-out, CoT
prompt) never beat plain F on accuracy. Historical F-tuned runs (exp18-26) remain
in the DB. The shared answer-context budget that F-tuned pioneered now lives in
`settings.fused_answer_top_k` (raised 10→20, see below).

**System G was also removed** after consistently underperforming A/F across Haiku,
Qwen3 and Nova Lite (e.g. Qwen3 passages exp14: G=0.444 vs A/F=0.889). G was the
deliberate multi-tool-agentic comparator to Ferrazzi et al. (2026)'s single-tool gap,
but its design (multiple per-step retrieval tools + accumulation across steps) lost
chunks at the answer-step budget cap and the typed decision schema was fragile
under non-Anthropic LLMs. Historical G runs remain in the DB (exp11/12/14) as
evidence; the *finding* that multi-tool agentic doesn't pay off here is part of
the dissertation's narrative.

All implement `systems/base.py:System` protocol → `answer(query: str) -> RunResult`.

`RunResult` fields: `answer, retrieved_chunk_ids, hhem_score, flagged,
n_steps, tokens_in, tokens_out, latency_ms, cost_usd, phoenix_trace_id,
all_retrieved_chunk_ids`. `retrieved_chunk_ids` is the final answering context;
`all_retrieved_chunk_ids` is the union of everything retrieved across the run
(all B iterations / all F-tuned fan-out), for the retrieval-ceiling analysis.

### Shared (every system uses the same underlying calls)

| Concern          | Module / Function                                            |
|------------------|--------------------------------------------------------------|
| Query embedding  | `retrieval/embeddings.py:embed_one` — `BAAI/llm-embedder`, 768-dim (the *model* is lru_cached via `get_model()`; per-query embeds are not) |
| Retrieval        | `retrieval/retrieve.py:retrieve` — hybrid BM25 + dense kNN, RRF-fused (`opensearch_client.py:hybrid_search`), then cross-encoder rerank to `top_k`. A/B/F/F-seq all share this; `knn_search`/`bm25_search` are its building blocks |
| Multi-list fusion | `retrieval/retrieve.py:rrf_fuse` — client-side RRF over per-query ranked lists; B fuses its iteration lists, F its sub-question lists. Answer context = fused top `FUSED_ANSWER_TOP_K = 20`, held **constant** across all fusing systems (B/F/F-seq) so the comparison isolates orchestration strategy, not the budget knob (raised 10→20 after exp36/37 showed retrieved gold evicted from a 10-slot context). The budget ablation (exp38/39/40) found the optimum is per-strategy — B@10=0.600 > B@20=0.540, but F-seq@20=0.540 ≫ F-seq@10=0.380 — so the fixed budget trades ~0.06 of B's accuracy for a budget-controlled comparison. One-list fusion is the identity ⇒ A's single retrieve still returns its top_k=10 |
| LLM call         | `llm/client.py:generate` — LiteLLM, `temperature=0`, returns content + tokens + cost |
| Top-k            | `settings.top_k = 10` per retrieve() call (A answers over all 10; B/F fuse their iteration/sub-question lists to top-10 via `FUSED_ANSWER_TOP_K`) |
| Trace capture    | `tracing.py:init_tracing` — auto-instruments LangChain + LiteLLM via openinference |

### Per-system spec (A and B; F has its own section below)

| Property              | A          | B                              |
|-----------------------|------------|--------------------------------|
| Pattern               | naive      | agent loop (LangGraph)         |
| Retrieve              | once       | each loop iter                 |
| LLM calls per query   | 1          | per iter: 1 route + 1 execute (reformulate **or** answer) |
| Reformulates query    | no         | yes (free-text, after a typed route via `instructor`) |
| Max steps             | n/a        | per-instance; default `settings.max_agent_steps = 5` (B1/B3/B5 sweep removed) |
| Cost accuracy         | ✅         | ✅ usage tokens + response_cost, falls back to litellm pricing |

**System B agent state machine** (`systems/system_b.py`):
```
RETRIEVE → ROUTE ──(reformulate)──┐
              │                   │
           (answer)               │
              ▼                   │
             END  ◀───────────────┘
```
- Each iteration is **two LLM calls**: a tiny one-field `systems/schemas.py:RouteDecision`
  (`action ∈ {reformulate, answer}` via `instructor`, so it can't parse-fail), then a
  free-text `generate()` that writes either the reformulated query or the final answer.
  The split makes B robust on Nova/Qwen3, which choked on the old single multi-field
  schema (`AgentDecision`, still defined but no longer used by B).
- Termination: `action == ANSWER` **or** `n_steps >= max_agent_steps`. The budget
  is **per-instance** (`SystemB(max_agent_steps=…)`, carried in `AgentState`),
  defaulting to `settings.max_agent_steps`. The route prompt instructs ANSWER on
  the final step, so `k=1` degenerates to one retrieve→answer (≈ A) rather than
  forcing "No answer".
- **Iteration budget** is per-instance. The B1/B3/B5 sweep (budgets 1/3/5 as separate
  registry entries) was **removed** — `SYSTEM_REGISTRY` is now just `A,B,F,F-seq`;
  historical sweep runs remain in the DB.
- **Evidence accumulates across iterations** (IRCoT-style union): route, reformulate
  and answer all operate on `rrf_fuse(iteration_hits)[:FUSED_ANSWER_TOP_K]` — the
  fused working memory, not just the latest batch. `retrieved_chunk_ids` persists
  that fused answering context; `all_retrieved_chunk_ids` the raw union. Before
  this, a chunk found at step 1 was invisible by step 3.

**Faithfulness — HHEM on every run** (`evaluation/runner.py:_faithfulness`, not a system):
- Computed for **every** system's run (A/B/F) ⇒ faithfulness is a column for all.
- Re-fetches retrieved chunk text via `mget(retrieved_chunk_ids)`, builds
  `premise = "\n\n".join(texts)`, scores `faithfulness/hhem.py:score([(premise, answer)])`.
- `flagged = score < settings.hhem_threshold`. Empty retrieval / HHEM error →
  `(None, None)` and the run still persists.
- Replaces the old passive Systems C/D, which only attached this score to A/B.

### Context formatting — `retrieval/retrieve.py:format_context`

A/B/F share `format_context(hits)` to build the per-chunk strings fed to the LLM.
Each chunk is rendered as `[<chunk_id>] (source: <X> | title: <Y>) <text>`,
surfacing `metadata.source` and `metadata.title` from the chunk's dataset metadata.
This closes a benchmark-specific gap: MultiHop-RAG identifies articles by their
`source` field ("the Hacker News article on The Epoch Times" → an NBC URL tagged
`source: Hacker News`), but the chunk text/URL alone doesn't expose it. Without
this, comparison queries that name a publisher fail across every system even when
the right chunk is retrieved. Don't revert this without re-running comparison-type
queries — the regression is visible.

### System F (query decomposition) — `systems/system_f.py`

Multi-hop decomposition baseline. One LLM call (`instructor` → `Decomposition`,
few-shot `DECOMPOSE_FEWSHOT_PROMPT` — shared with F-seq, which imports it plus F's
`_decompose`) splits the query into 2-4 single-hop sub-questions; F retrieves for
the original + each sub-question **in parallel** over the **same** `retrieve()` as
A/B, RRF-fuses the lists (`rrf_fuse`, deduped by chunk_id), and answers once with
the fused top-`FUSED_ANSWER_TOP_K` (=20) context. `retrieved_chunk_ids` = that
answering context (what HHEM scores against); `all_retrieved_chunk_ids` = the full
fused list.
- Retriever held constant ⇒ F-vs-A isolates the *decomposition* effect (vs E,
  which changes the retriever). Distinct from B: parallel decompose+fuse, not an
  iterative reformulation loop.
- Single-hop ⇒ empty decomposition ⇒ F reduces to A's retrieval and context.
- Decompose parse failure (Nova/Qwen JSON + trailing prose) degrades gracefully
  to no sub-questions instead of a stub row — same policy as F-seq.
- `n_steps` = number of retrievals (1 + #sub-questions); tokens/cost sum the
  decompose + answer calls (properly tracked, unlike B).

### System F-seq (sequential self-ask decomposition) — `systems/system_fseq.py`

F's sequential counterpart and the reason F-tuned was retired. Same decomposer and
`retrieve()`/answer prompt as F, but resolves the ordered sub-questions **one hop
at a time**: each hop substitutes the already-resolved bridge answers into its
retrieval query (so "that director" becomes the real name — F's dead-bridge
problem), retrieves, then a small LLM call answers that sub-question from its own
context (`SUB_HOP_TOP_K = 5`). Resolved facts carry forward; the final answer is
generated over the RRF-fused union of every hop (top `FUSED_ANSWER_TOP_K`) with the
resolved intermediate facts supplied as a reasoning scaffold.
- **F-vs-F-seq isolates PARALLEL vs SEQUENTIAL decomposition; F-seq-vs-B isolates
  pre-decomposed self-ask vs free-form iterative reformulation** — the three-way
  decomposition carve-out. Cites the self-ask / least-to-most lineage (Press et al.
  2023; Zhou et al. 2023).
- A failed hop answers `UNKNOWN` and is **not** carried forward (can't poison the
  next query). Single-hop ⇒ no sub-questions ⇒ F-seq reduces to A's context.
- `n_steps` = retrievals (1 + #sub-questions); tokens/cost sum decompose + per-hop
  answer calls + final answer.
- **Cite Ammann, Golde & Akbik (2025)**, *Question Decomposition for RAG* (ACL 2025
  SRW): F mirrors their decomposition + off-the-shelf reranker recipe on
  MultiHop-RAG (they report MRR@10 +36.7%, F1 +11.6%). They flag *"lack of iterative
  retrieval … a limitation compared to more adaptive approaches"* → **System B is
  the iterative comparator they lack**; A/B/F on a fixed retriever isolate
  naive/iteration/decomposition. This is the study's carve-out vs. their paper.
  (Verify the quote against the published PDF before quoting verbatim.)

## Evaluation pipeline

### 1. Ingest (one-time per dataset)

| Dataset    | HF source                                | Used for                       | Splits        |
|------------|------------------------------------------|--------------------------------|---------------|
| MultiHop   | `yixuantt/MultiHopRAG` (`corpus` + `MultiHopRAG` configs) | Retrieval + answer correctness | `eval`        |
| MuSiQue    | `dgslibisey/MuSiQue`                      | Multi-hop retrieval + answer correctness (anti-shortcut; per-question pooled corpus, index `rag-chunks-musique`) | `eval` |

- MultiHop chunks are keyed by **URL** (`external_id = row["url"]`); a query's
  `relevant_chunk_ids` is the unique URL list from its `evidence_list`.
- MuSiQue chunks are per-question paragraphs keyed `<question_id>::p<idx>`;
  gold = the `is_supporting` paragraph ids, hop count stored as `question_type`.
- Both ingests are **idempotent** via external_id skip-set.
- **RAGTruth was removed** as a dataset. It previously seeded HHEM-threshold
  calibration; HHEM faithfulness is still computed for every run but now against
  the fixed `settings.hhem_threshold`. Historical `ragtruth` query rows remain in
  the DB (hidden from `/api/datasets` via `HIDDEN_DATASETS`).

### 2. Index corpus (MultiHop + MuSiQue)

`retrieval/indexer.py:index_corpus` reads chunks from Postgres, embeds in
batches of 64, bulk-loads into OpenSearch with HNSW (Lucene, cosine).
Index settings: `knn.algo_param.ef_search = 100`. MuSiQue indexes into a
separate index (`OPENSEARCH_INDEX=rag-chunks-musique`).

### 3. HHEM faithfulness threshold

HHEM scores every run; `flagged = score < settings.hhem_threshold` (default
`0.10`, env-overridable). The RAGTruth-driven calibration step was removed — the
threshold is now a fixed config value, not fit from a calibration split.

### 4. Run experiment

`evaluation/runner.py:run_experiment` →
- Creates one `experiments` row with the config snapshot.
- Query selection: `--sample N [--seed S] [--no-stratify]` = seeded random subset,
  stratified by `question_type` by default, with the exact `query_ids` recorded in
  `config_json.selection` for reproducibility; `--limit N` = first-N by id (quick
  smoke test only, **not** a random sample). `--sample` wins if both are passed.
- For each `(system, query)` pair: calls `system.answer(q.query_text)`,
  scores correctness, upserts a `runs` row.
- Uses `ON CONFLICT DO NOTHING` against `UNIQUE(experiment_id, system, query_id)`.

**Resumability invariant** — do not break:
- Re-running the CLI command must pick up where it stopped.
- Per-query exceptions persist a stub row (`answer = NULL`) so resume skips them.
- To retry failed rows: `DELETE FROM runs WHERE answer IS NULL` and re-run.

### 5. Compute metrics

`cli.py:compute_metrics` groups runs by `(system, dataset)` and upserts a
`metrics` row containing:

| Metric                   | Source                                                       |
|--------------------------|--------------------------------------------------------------|
| `precision_at_5`         | mean over runs of `precision_at_k(retrieved, relevant, 5)`   |
| `recall_at_5`            | mean over runs of `recall_at_k(retrieved, relevant, 5)`      |
| `accuracy`               | `sum(is_correct) / n` (primary: `contains_match`)            |
| `accuracy_exact`         | mean `exact_match` over runs (secondary, stricter)           |
| `avg_token_f1`           | mean SQuAD-style `token_f1` over runs (secondary, lexical-overlap; comparable to Ammann et al. answer-F1) |
| `crag_score`             | mean CRAG truthfulness over judged runs (secondary, LLM-judge)|
| `avg_faithfulness`       | mean HHEM over runs where score is not null                  |
| `pct_flagged`            | fraction where `flagged = True`                              |
| `avg_trajectory_length`  | mean `n_steps`                                               |
| `pct_failed`             | fraction of runs that errored (`answer IS NULL`); these count as wrong in `accuracy` (deliberate "crash = wrong" policy) |
| `total_cost_usd`         | sum of `cost_usd`                                            |
| `cost_per_correct`       | `total_cost_usd / sum(is_correct)`                           |

### 6. Export / analyse

- `cli.py:export` → JSON dumps of runs + metrics per experiment.
- `notebooks/analysis.py` → Marimo notebook, reads Postgres directly,
  produces cost-accuracy Pareto chart.
- Phoenix UI (`:6006`) for per-query span drill-down.

## Correctness scoring

Primary correctness is **deterministic** (`evaluation/metrics.py`):
- `contains_match(predicted, gold)` — normalised gold ⊆ normalised prediction.
  **This is the primary metric**, used by the runner for all datasets, and it
  matches the MultiHop-RAG paper (Tang & Yang 2024): gold answers are short
  factoids (yes/no, entity, before/after) scored by containment in the response.
- `exact_match(predicted, gold)` — normalised full-string equality; a *stricter*
  secondary (the paper does **not** require this).

Secondary, opt-in: a **CRAG LLM-as-judge** (`evaluation/judge.py`, Yang et al. 2024
rubric perfect/acceptable/missing/incorrect → 1/0.5/0/−1), run post-hoc via
`cli.py:judge` (idempotent: only judges rows with `answer` and no `llm_judge_label`)
and aggregated into `metrics.crag_score`. Non-deterministic in principle (run at
`temperature=0`), which is why it is secondary and never the headline number. The
judge uses `settings.judge_model` (falls back to `litellm_model`) and is
provider-agnostic — pair cheap generation with a strong, independent judge.

## Database schema (`api/src/db/models.py`)

```
experiments(id, name, config_json, started_at, finished_at, notes)
queries    (id, dataset, external_id, split, task_type, query_text,
            ground_truth, relevant_chunk_ids JSONB, metadata JSONB)
chunks     (id, dataset, external_id, text, metadata JSONB)
runs       (id, experiment_id→experiments, system, query_id→queries,
            retrieved_chunk_ids JSONB, all_retrieved_chunk_ids JSONB, answer, hhem_score, flagged,
            n_steps, tokens_in, tokens_out, latency_ms, cost_usd,
            is_correct, llm_judge_label, phoenix_trace_id, created_at)
metrics    (id, experiment_id, system, dataset, n_queries,
            precision_at_5, recall_at_5, avg_faithfulness, pct_flagged,
            avg_trajectory_length, pct_failed, accuracy, accuracy_exact, avg_token_f1,
            crag_score, total_cost_usd, cost_per_correct, computed_at)
```

Key constraints (load-bearing):
- `UNIQUE(queries.dataset, queries.external_id)` — idempotent ingest
- `UNIQUE(chunks.dataset, chunks.external_id)` — idempotent ingest
- `UNIQUE(runs.experiment_id, runs.system, runs.query_id)` — **resumability**
- `UNIQUE(metrics.experiment_id, metrics.system, metrics.dataset)` — upsert target

## Config (`api/src/config.py`)

All settings via pydantic-settings, env-overridable. Defaults shown:

```python
database_url            = "postgresql+psycopg://rag:ragbench@postgres:5432/ragbench"
opensearch_url          = "http://opensearch:9200"
phoenix_collector_endpoint = "http://phoenix:6006"
litellm_model           = "bedrock/amazon.nova-lite-v1:0"   # Haiku 4.5 etc. via LITELLM_MODEL env
judge_model             = None    # LLM-as-judge model; falls back to litellm_model. Lets you pair cheap generation + strong judging
aws_region              = "eu-west-2"
embedding_model         = "BAAI/llm-embedder"   # 768-dim
embedding_dim           = 768
opensearch_index        = "rag-chunks"
top_k                   = 10
fused_answer_top_k      = 20                     # answer-context budget for ALL fusing systems (B/F/F-seq), held constant for a controlled comparison; per-strategy optimum differs (ablation exp38/39/40)
retrieval_pool          = 20                     # hybrid first-stage pool size before rerank
reranker_model          = "BAAI/bge-reranker-v2-m3"
rerank_provider         = "local"                # "local" cross-encoder | "bedrock-cohere"
max_agent_steps         = 5
hhem_threshold          = 0.10                   # fixed empirical threshold for HHEM-2.1-open on news (RAGTruth calibration removed)
```

## Conventions

- **Lazy singletons** via `@lru_cache(maxsize=1)` for models and OpenSearch
  client. Don't instantiate models at import time.
- **Composition over inheritance** for systems. Don't subclass one system from
  another; share via the `retrieve()` / `generate()` helpers.
- **Sync** everything except FastAPI route signatures. No `asyncio` work
  inside systems or evaluation code.
- **Idempotency** is a hard requirement: ingest, index, run-experiment,
  compute-metrics must all be safe to re-run.
- **Determinism** wherever LLMs are called (`temperature=0`).
- **No comments** unless WHY is non-obvious. Identifiers should carry intent.

## Known gaps (do not silently expand scope to fix these)

| # | Where                                  | What                                                       | Impact                            |
|---|----------------------------------------|------------------------------------------------------------|-----------------------------------|
| 1 | `systems/system_b.py`                  | ~~`cost_usd` reads 0 if instructor raw lacks response_cost~~ **fixed**: falls back to `litellm.cost_per_token` from usage | resolved (needs model in litellm pricing map) |
| 2 | All systems                            | `phoenix_trace_id` always `None`                           | No SQL→Phoenix link from `runs`   |
| 3 | `evaluation/runner.py`                 | Uses `contains_match` for MultiHop (= paper's containment metric) | OK as primary; `exact_match`/CRAG are stricter secondaries |
| 4 | `cli.py:compute_metrics`               | Accuracy denominator includes failed (`answer IS NULL`) rows — **deliberate**: a crash is a wrong answer. Now surfaced via the `metrics.pct_failed` column rather than hidden | resolved (policy + visible failure rate) |
| 5 | `datasets/multihop.py` + indexer       | MultiHop now ingests **256-token passages by default** (`DEFAULT_PASSAGE_TOKENS=256`; pass `passage_tokens=None` to revert to article/URL). Gold stays URL-keyed; `metrics.py:_article_id` maps passages→parent URL when scoring | mostly resolved; a *literal* Tang & Yang Table 5 replication still needs a fact→passage gold mapping, not just passage chunking |

Fix only when explicitly asked. When asked, fix only the requested item.

## Decisions deliberately made — do not reintroduce

From the design doc (`rag-benchmark-stack-guide-for llm.md` §2):

- **No Streamlit/Gradio/React frontend.** Phoenix UI + the in-repo SPA at `/` cover it.
- **No LangSmith.** Phoenix is self-hosted, zero-credentials equivalent.
- **No additional vector DB** (Qdrant/Weaviate/LanceDB). OpenSearch mirrors AWS prod.
- **No LiteLLM proxy container.** SDK is simpler for single-user.
- **No Prefect/Airflow/Dagster.** The runner is a `for`-loop.
- **No Ragas/DeepEval/TruLens for retrieval metrics.** Retrieval/IR scoring stays
  deterministic arithmetic. (Exception, added deliberately: a CRAG LLM-as-judge —
  `evaluation/judge.py` — is now a *secondary* answer-correctness metric alongside
  the primary `contains_match`. Do **not** let it replace the deterministic primary,
  and do not remove it as a "reintroduced" anti-pattern.)
- **No new Python dependencies** without explicit user approval. Stick to `requirements.txt`.
- **No tests** unless explicitly requested (none exist; not on the dissertation critical path).
- **No System E (vendored OpenRag), no OpenAI/Cohere deps.** Removed in favour of
  Bedrock-only stack. Cite Ammann et al. (2025) and OpenRag's published numbers as
  the SOTA reference instead of running an in-repo OpenRag baseline. Do not
  re-vendor `ultimate_rag` / `knowledge_base` without an explicit decision.

## Operating the repo

### CLI is the primary surface

```bash
docker compose run --rm api python -m src.cli healthcheck
docker compose run --rm api python -m src.cli ingest-dataset {multihop|musique}
docker compose run --rm api python -m src.cli index-corpus multihop
docker compose run --rm -e OPENSEARCH_INDEX=rag-chunks-musique api python -m src.cli index-corpus musique
docker compose run --rm api python -m src.cli run-experiment --name X --systems A,B,F --datasets multihop
docker compose run --rm api python -m src.cli run-experiment --name smoke --systems A --datasets multihop --limit 20  # quick smoke test (first 20)
docker compose run --rm api python -m src.cli run-experiment --name sub --systems A,B,F,F-seq --datasets multihop --sample 500 --seed 42  # defensible stratified subset
docker compose run --rm api python -m src.cli compute-metrics --experiment N
docker compose run --rm api python -m src.cli judge --experiment N            # CRAG LLM-as-judge (post-hoc, resumable)
docker compose run --rm api python -m src.cli metrics-by-type --experiment N  # accuracy by question type
docker compose run --rm api python -m src.cli retrieval-eval --dataset multihop --k 10  # MRR@k/MAP@k/Hits@k probe
docker compose run --rm api python -m src.cli export --experiment N
```

### SPA at `http://localhost:8000/`

Three tabs (Ask / Data / Experiments). Covers fast interactive paths only.
The **Data** tab links each dataset to its Hugging Face source and has an
**Explore** panel that browses ingested queries (filter by question_type, click a
row for gold-chunk text) and chunks (substring search), paginated via the
`/api/datasets/{ds}/queries|chunks` endpoints. `run-experiment` deliberately
remains CLI-only (too long for HTTP).
Experiment detail surfaces config/selection, the full metrics row (incl.
`accuracy_exact`, `crag_score`, steps), a per-question-type table
(`/api/experiments/{id}/by-type`), and per-run rows (judge label, steps, cost).
`LOG_LEVEL=DEBUG` (env) gives per-run + per-agent-step logs; Phoenix has the spans.

### Phoenix at `http://localhost:6006`

Auto-populated. No code touches Phoenix directly except `tracing.py:init_tracing`.

### Dissertation claim audit — `DISSERTATION_AUDIT.md`

`DISSERTATION_AUDIT.md` (repo root) maps the dissertation's Gaps/RQs/Aims/Objectives
to implementation status and holds the action register (C/N/P/W/D items). Read it
before any dissertation-claim, slide, or final-run work and update statuses in
place — do **not** re-audit the codebase from scratch.

## Environment setup notes

- `.env` is gitignored; `.env.example` is the template (allow-listed in `.gitignore`).
- `docker-compose.yml:83` mounts `C:\Users\MadhavS\.aws:/root/.aws:ro` —
  **Windows-only**. macOS/Linux users must change to `${HOME}/.aws:/root/.aws:ro`
  or remove the mount and rely on env vars.

## When making changes

- Edit existing files in preference to creating new ones.
- Stay within `requirements.txt`. If a task seems to need a new dep, ask first.
- Keep diffs surgical — don't refactor adjacent code while fixing a bug.
- Verify the resumability invariant still holds after any runner change.
- Don't add comments explaining *what* the code does; the names should tell you.
