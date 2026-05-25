# CLAUDE.md

Repo memory for Claude Code sessions. Read this before editing.

## What this is

Reproducible Dockerised benchmark of RAG systems (A naive, B agentic, F decomposition)
against MultiHop-RAG and RAGTruth, with every metric persisted to Postgres
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
├── datasets/{multihop,ragtruth}.py
├── evaluation/{runner,metrics,calibration,judge}.py
├── faithfulness/hhem.py        # vectara/hallucination_evaluation_model
├── llm/client.py               # LiteLLM wrapper
├── retrieval/{embeddings,opensearch_client,indexer,retrieve}.py
└── systems/{base,schemas,system_a,system_b,system_f}.py
notebooks/analysis.py           # Marimo notebook for Chapter 4 figures
```

## The systems

Lineup: **A** (naive), **B** (agentic single-tool loop), **F** (query decomposition),
**G** (multi-tool agentic retrieval). Faithfulness (HHEM) is computed for every
run, not by a system — the old passive Systems C/D were folded into that metric.
System E (vendored OpenRag) was removed; see top-of-file note.

G is the deliberate response to Ferrazzi et al. (2026, ACL Industry Track), who
benchmarked single-tool agentic RAG and listed "multi-tool agents" as a gap.
B reformulates the query but always calls the same hybrid retriever; G picks
between `retrieve_semantic` / `retrieve_bm25` / `retrieve_filtered(source=..., category=...)`
per step using a typed `instructor` decision. Same LangGraph state machine,
same per-instance step budget (G1/G3/G5 mirror B1/B3/B5).

All implement `systems/base.py:System` protocol → `answer(query: str) -> RunResult`.

`RunResult` fields: `answer, retrieved_chunk_ids, hhem_score, flagged,
n_steps, tokens_in, tokens_out, latency_ms, cost_usd, phoenix_trace_id`.

### Shared (every system uses the same underlying calls)

| Concern          | Module / Function                                            |
|------------------|--------------------------------------------------------------|
| Query embedding  | `retrieval/embeddings.py:embed_one` — `BAAI/llm-embedder`, 768-dim, lru_cached |
| Vector search    | `retrieval/opensearch_client.py:knn_search` — HNSW kNN over `rag-chunks` index |
| LLM call         | `llm/client.py:generate` — LiteLLM, `temperature=0`, returns content + tokens + cost |
| Top-k            | `settings.top_k = 5`                                          |
| Trace capture    | `tracing.py:init_tracing` — auto-instruments LangChain + LiteLLM via openinference |

### Per-system spec (A and B; F has its own section below)

| Property              | A          | B                              |
|-----------------------|------------|--------------------------------|
| Pattern               | naive      | agent loop (LangGraph)         |
| Retrieve              | once       | each loop iter                 |
| LLM calls per query   | 1          | N × decide + (final synthesis) |
| Reformulates query    | no         | yes (typed via `instructor`)   |
| Max steps             | n/a        | per-instance (B1/B3/B5); default `settings.max_agent_steps = 5` |
| Cost accuracy         | ✅         | ✅ usage tokens + response_cost, falls back to litellm pricing |

**System B agent state machine** (`systems/system_b.py`):
```
RETRIEVE → DECIDE ──(reformulate)──┐
              │                    │
           (answer)                │
              ▼                    │
             END  ◀────────────────┘
```
- Agent decision is a typed Pydantic model `systems/schemas.py:AgentDecision`
  enforcing `action ∈ {reformulate, answer}` via `instructor` → parse failures
  cannot happen.
- Termination: `action == ANSWER` **or** `n_steps >= max_agent_steps`. The budget
  is **per-instance** (`SystemB(max_agent_steps=…)`, carried in `AgentState`),
  defaulting to `settings.max_agent_steps`. The decide prompt instructs ANSWER on
  the final step, so `k=1` degenerates to one retrieve→answer (≈ A) rather than
  forcing "No answer".
- **Iteration sweep:** `B1`/`B3`/`B5` in `runner.py:SYSTEM_REGISTRY` are System B at
  budgets 1/3/5, run side-by-side in one experiment (`--systems B1,B3,B5`) →
  `compute-metrics` gives a row each = a cost-vs-accuracy curve; `B1` is the
  no-iteration ablation.

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

Multi-hop decomposition baseline. One LLM call (`instructor` → `Decomposition`)
splits the query into 2-4 single-hop sub-questions; F retrieves for the original
+ each sub-question over the **same** `retrieve()` as A/B, RRF-fuses the lists
(deduped by chunk_id), and answers once with the fused top-`top_k` context.
- Retriever held constant ⇒ F-vs-A isolates the *decomposition* effect (vs E,
  which changes the retriever). Distinct from B: parallel decompose+fuse, not an
  iterative reformulation loop.
- Single-hop ⇒ empty decomposition ⇒ F reduces to A's retrieval.
- `n_steps` = number of retrievals (1 + #sub-questions); tokens/cost sum the
  decompose + answer calls (properly tracked, unlike B).
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
| RAGTruth   | `ParticleMedia/RAGTruth`                 | HHEM threshold calibration     | `calibration` |

- MultiHop chunks are keyed by **URL** (`external_id = row["url"]`); a query's
  `relevant_chunk_ids` is the unique URL list from its `evidence_list`.
- RAGTruth `hallucination` label is derived from non-empty `labels` span list
  (1 = hallucinated). HHEM premise = `source` (fall back to `prompt`).
- Both ingests are **idempotent** via external_id skip-set.

### 2. Index corpus (MultiHop only)

`retrieval/indexer.py:index_corpus` reads chunks from Postgres, embeds in
batches of 64, bulk-loads into OpenSearch with HNSW (Lucene, cosine).
Index settings: `knn.algo_param.ef_search = 100`.

### 3. Calibrate HHEM threshold (one-time)

`cli.py:calibrate` →
- Loads RAGTruth `calibration` rows from Postgres.
- Scores each `(query_text, ground_truth)` pair through HHEM.
- `evaluation/calibration.py:fit_threshold` finds the F1-maximising threshold
  over the flipped-score ROC curve.
- Writes `/data/results/threshold.json` (+ `calibration_curve.png`).
- User pastes the threshold into `.env` as `HHEM_THRESHOLD`.

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
| `crag_score`             | mean CRAG truthfulness over judged runs (secondary, LLM-judge)|
| `avg_faithfulness`       | mean HHEM over runs where score is not null                  |
| `pct_flagged`            | fraction where `flagged = True`                              |
| `avg_trajectory_length`  | mean `n_steps`                                               |
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
            retrieved_chunk_ids JSONB, answer, hhem_score, flagged,
            n_steps, tokens_in, tokens_out, latency_ms, cost_usd,
            is_correct, llm_judge_label, phoenix_trace_id, created_at)
metrics    (id, experiment_id, system, dataset, n_queries,
            precision_at_5, recall_at_5, avg_faithfulness, pct_flagged,
            avg_trajectory_length, accuracy, accuracy_exact, crag_score,
            total_cost_usd, cost_per_correct, computed_at)
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
litellm_model           = "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
judge_model             = None    # LLM-as-judge model; falls back to litellm_model. Lets you pair cheap generation + strong judging
aws_region              = "eu-west-2"
embedding_model         = "BAAI/llm-embedder"   # 768-dim
embedding_dim           = 768
opensearch_index        = "rag-chunks"
top_k                   = 5
max_agent_steps         = 5
hhem_threshold          = 0.5                    # overridden post-calibration
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
| 4 | `cli.py:compute_metrics`               | Accuracy denominator includes `is_correct IS NULL` rows    | Underreports accuracy             |
| 5 | `datasets/multihop.py` + indexer       | MultiHop indexed at **article/URL** granularity, not 256-token passages | `retrieval-eval` numbers read higher than & aren't comparable to Tang & Yang Table 5; literal replication needs passage-level chunking + fact→passage gold |

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
docker compose run --rm api python -m src.cli ingest-dataset {multihop|ragtruth}
docker compose run --rm api python -m src.cli index-corpus multihop
docker compose run --rm api python -m src.cli calibrate
docker compose run --rm api python -m src.cli run-experiment --name X --systems A,B,F --datasets multihop
docker compose run --rm api python -m src.cli run-experiment --name smoke --systems A --datasets multihop --limit 20  # quick smoke test (first 20)
docker compose run --rm api python -m src.cli run-experiment --name sub --systems A,B,F --datasets multihop --sample 500 --seed 42  # defensible stratified subset
docker compose run --rm api python -m src.cli run-experiment --name ksweep --systems B1,B3,B5 --datasets multihop  # B iteration sweep
docker compose run --rm api python -m src.cli compute-metrics --experiment N
docker compose run --rm api python -m src.cli judge --experiment N            # CRAG LLM-as-judge (post-hoc, resumable)
docker compose run --rm api python -m src.cli metrics-by-type --experiment N  # accuracy by question type
docker compose run --rm api python -m src.cli retrieval-eval --dataset multihop --k 10  # MRR@k/MAP@k/Hits@k probe
docker compose run --rm api python -m src.cli export --experiment N
```

### SPA at `http://localhost:8000/`

Three tabs (Ask / Data / Experiments). Covers fast interactive paths only.
`run-experiment` and `calibrate` deliberately remain CLI-only (too long for HTTP).
Experiment detail surfaces config/selection, the full metrics row (incl.
`accuracy_exact`, `crag_score`, steps), a per-question-type table
(`/api/experiments/{id}/by-type`), and per-run rows (judge label, steps, cost).
`LOG_LEVEL=DEBUG` (env) gives per-run + per-agent-step logs; Phoenix has the spans.

### Phoenix at `http://localhost:6006`

Auto-populated. No code touches Phoenix directly except `tracing.py:init_tracing`.

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
