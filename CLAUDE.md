# CLAUDE.md

Repo memory for Claude Code sessions. Read this before editing.

## What this is

Reproducible Dockerised benchmark of four RAG architectures (Systems A/B/C/D)
against MultiHop-RAG and RAGTruth, with every metric persisted to Postgres
and every span traced to Phoenix. Single-user research tool for a dissertation
— **not** a production service.

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
│   └── migrations/             # Alembic (single migration: 0001_initial)
├── datasets/{multihop,ragtruth}.py
├── evaluation/{runner,metrics,calibration}.py
├── faithfulness/hhem.py        # vectara/hallucination_evaluation_model
├── llm/client.py               # LiteLLM wrapper
├── retrieval/{embeddings,opensearch_client,indexer}.py
└── systems/{base,schemas,system_a,system_b,system_c,system_d}.py
notebooks/analysis.py           # Marimo notebook for Chapter 4 figures
```

## The four systems

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

### Per-system spec

| Property              | A          | B                              | C              | D                              |
|-----------------------|------------|--------------------------------|----------------|--------------------------------|
| Pattern               | naive      | agent loop (LangGraph)         | A + HHEM gate  | B + HHEM gate                  |
| Wrapper / composition | —          | —                              | wraps SystemA  | wraps SystemB                  |
| Retrieve              | once       | each loop iter                 | once (via A)   | each loop iter (via B)         |
| LLM calls per query   | 1          | N × decide + (final synthesis) | 1              | N × decide                     |
| Reformulates query    | no         | yes (typed via `instructor`)   | no             | yes                            |
| Max steps             | n/a        | `settings.max_agent_steps = 5` | n/a            | 5                              |
| HHEM faithfulness     | no         | no                             | yes            | yes                            |
| Sets `flagged`        | `None`     | `None`                         | yes            | yes                            |
| Cost accuracy         | ✅         | ⚠️ tokens/cost hardcoded to 0  | ✅ (inherits A) | ⚠️ inherits B's gap            |

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
- Termination: `action == ANSWER` **or** `n_steps >= max_agent_steps`.

**HHEM gate (Systems C and D)**:
- Re-fetches chunk text from OpenSearch via `mget(ids=retrieved_chunk_ids)`
- Builds `premise = "\n\n".join(chunk texts)`
- Calls `faithfulness/hhem.py:score([(premise, answer)])` → 0..1 float
- `flagged = score < settings.hhem_threshold` (default 0.5; set by calibration)

### System E (vendored OpenRag) — `systems/system_e.py`

OpenRag's `ultimate_rag` engine is vendored as top-level packages
(`api/ultimate_rag/`, `api/knowledge_base/`) and called **in-process** — no HTTP.
- Retrieves over an **in-memory RAPTOR forest** (not OpenSearch), built once via
  `cli.py:build-openrag-index` and persisted to `settings.openrag_tree_dir`
  (`/data/openrag_trees`). Loaded lazily by an `lru_cache` singleton.
- All `ultimate_rag` / `knowledge_base` imports are **deferred inside the
  singleton/build fn** so importing `system_e` (which `runner.py` does next to
  A-D) never needs OpenRag's deps or a built tree. Keep them lazy.
- OpenRag returns chunk text without a URL → System E recovers the article URL by
  text-containment against the Postgres corpus, so it's scored URL-level like A-D.
- Answer generation reuses System A's prompt + the shared Bedrock LLM.
- `cost_usd` covers only the answer call (OpenRag's OpenAI/Cohere spend is external).

System E **deliberately waives** three invariants below (do not "fix"): it adds
deps (openai/cohere/tiktoken/umap-learn/rank-bm25/tenacity), uses an external
rerank API (Cohere), and bridges async→sync via `asyncio.run`. These apply to
System E only; A-D remain dep-light, local, and sync.

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
| `accuracy`               | `sum(is_correct) / n`                                        |
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

`evaluation/metrics.py` — deliberately **no LLM-as-judge** (deterministic
for the thesis):
- `exact_match(predicted, gold)` — normalised (lowercase, punctuation-stripped,
  whitespace-collapsed) equality.
- `contains_match(predicted, gold)` — normalised gold ⊆ normalised prediction.

Runner currently uses `contains_match` for all datasets (`runner.py:119`);
MultiHop paper spec is `exact_match`.

## Database schema (`api/src/db/models.py`)

```
experiments(id, name, config_json, started_at, finished_at, notes)
queries    (id, dataset, external_id, split, task_type, query_text,
            ground_truth, relevant_chunk_ids JSONB, metadata JSONB)
chunks     (id, dataset, external_id, text, metadata JSONB)
runs       (id, experiment_id→experiments, system, query_id→queries,
            retrieved_chunk_ids JSONB, answer, hhem_score, flagged,
            n_steps, tokens_in, tokens_out, latency_ms, cost_usd,
            is_correct, phoenix_trace_id, created_at)
metrics    (id, experiment_id, system, dataset, n_queries,
            precision_at_5, recall_at_5, avg_faithfulness, pct_flagged,
            avg_trajectory_length, accuracy, total_cost_usd,
            cost_per_correct, computed_at)
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
- **Composition over inheritance** for systems (C wraps A; D wraps B). Do not
  inherit from `SystemA`/`SystemB`.
- **Sync** everything except FastAPI route signatures. No `asyncio` work
  inside systems or evaluation code.
- **Idempotency** is a hard requirement: ingest, index, run-experiment,
  compute-metrics must all be safe to re-run.
- **Determinism** wherever LLMs are called (`temperature=0`).
- **No comments** unless WHY is non-obvious. Identifiers should carry intent.

## Known gaps (do not silently expand scope to fix these)

| # | Where                                  | What                                                       | Impact                            |
|---|----------------------------------------|------------------------------------------------------------|-----------------------------------|
| 1 | `systems/system_b.py:131-134`          | `tokens_in/out`, `cost_usd` hardcoded to 0                 | B/D `$/correct` always reads 0    |
| 2 | `systems/system_c.py`, `system_d.py`   | `mget(ids=[])` raises on empty retrieval                   | C/D crash on no-result queries    |
| 3 | `systems/system_c.py` ↔ `system_d.py`  | HHEM-gate code duplicated verbatim                         | Drift risk; trivial to extract    |
| 4 | All systems                            | `phoenix_trace_id` always `None`                           | No SQL→Phoenix link from `runs`   |
| 5 | `evaluation/runner.py:119`             | Uses `contains_match` for MultiHop                         | Not paper-spec; inflates accuracy |
| 6 | `cli.py:135`                           | Accuracy denominator includes `is_correct IS NULL` rows    | Underreports accuracy             |
| 7 | `retrieval/opensearch_client.py:89`    | `hybrid_search` falls back to `knn_search`                 | True BM25+kNN hybrid is a TODO    |

Fix only when explicitly asked. When asked, fix only the requested item.

## Decisions deliberately made — do not reintroduce

From the design doc (`rag-benchmark-stack-guide-for llm.md` §2):

- **No Streamlit/Gradio/React frontend.** Phoenix UI + the in-repo SPA at `/` cover it.
- **No LangSmith.** Phoenix is self-hosted, zero-credentials equivalent.
- **No additional vector DB** (Qdrant/Weaviate/LanceDB). OpenSearch mirrors AWS prod.
- **No LiteLLM proxy container.** SDK is simpler for single-user.
- **No Prefect/Airflow/Dagster.** The runner is a `for`-loop.
- **No Ragas/DeepEval/TruLens for retrieval metrics.** LLM-judge is non-deterministic.
- **No new Python dependencies** without explicit user approval. Stick to `requirements.txt`.
- **No tests** unless explicitly requested (none exist; not on the dissertation critical path).

## Operating the repo

### CLI is the primary surface

```bash
docker compose run --rm api python -m src.cli healthcheck
docker compose run --rm api python -m src.cli ingest-dataset {multihop|ragtruth}
docker compose run --rm api python -m src.cli index-corpus multihop
docker compose run --rm api python -m src.cli calibrate
docker compose run --rm api python -m src.cli run-experiment --name X --systems A,B,C,D --datasets multihop
docker compose run --rm api python -m src.cli compute-metrics --experiment N
docker compose run --rm api python -m src.cli export --experiment N
```

### SPA at `http://localhost:8000/`

Three tabs (Ask / Data / Experiments). Covers fast interactive paths only.
`run-experiment` and `calibrate` deliberately remain CLI-only (too long for HTTP).

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
