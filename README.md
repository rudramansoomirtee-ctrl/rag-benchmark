# RAG Benchmark Stack

Reproducible Dockerised environment to evaluate Systems A/B/C/D against MultiHop-RAG and RAGTruth.

Four services, one `docker compose up`, all telemetry to Phoenix, all metrics to Postgres.

---

## Stack

| Service    | What it does                                | Port |
|------------|---------------------------------------------|------|
| opensearch | Dense + BM25 retrieval (mirrors AWS prod)   | 9200 |
| postgres   | Structured runs, queries, chunks, metrics   | 5432 |
| phoenix    | OTel tracing + eval dashboard               | 6006 |
| api        | FastAPI + LangGraph + HHEM + CLI            | 8000 |

LLM is AWS Bedrock (Claude Haiku 4.5), called via LiteLLM — swap providers with one env var.

---

## Quick start

> **macOS/Linux:** `docker-compose.yml` mounts `C:\Users\MadhavS\.aws:/root/.aws:ro`.
> Change to `${HOME}/.aws:/root/.aws:ro` (or remove the line entirely and rely on the
> `AWS_*` env vars from `.env`).

```bash
# 1. Configure
cp .env.example .env
# fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for Bedrock

# 2. Bring everything up
docker compose up -d opensearch postgres phoenix

# 3. Build the api image (first time only, takes ~10 min for HHEM weights)
docker compose build api

# 4. Run database migrations
docker compose run --rm api alembic upgrade head

# 5. Verify everything is alive
docker compose run --rm api python -m src.cli healthcheck
# expected: 5 green lines

# 6. Start the api (with hot-reload on src/)
docker compose up -d api
```

Open Phoenix at <http://localhost:6006> and the FastAPI docs at <http://localhost:8000/docs>.

---

## Workflow

```bash
# Load datasets into Postgres
docker compose exec api python -m src.cli ingest-dataset multihop
docker compose exec api python -m src.cli ingest-dataset ragtruth

# Index the MultiHop corpus into OpenSearch
docker compose exec api python -m src.cli index-corpus multihop

# Fit the HHEM threshold on RAGTruth
docker compose exec api python -m src.cli calibrate
# writes /data/results/threshold.json + calibration_curve.png

# Run the full eval (resumable — re-run on failure picks up where it left off)
docker compose exec api python -m src.cli run-experiment \
  --name dissertation-final \
  --systems A,B,C,D \
  --datasets multihop \
  --split eval

# Compute aggregate metrics (writes the `metrics` table + prints a Rich table)
docker compose exec api python -m src.cli compute-metrics --experiment 1

# Export everything for the dissertation
docker compose exec api python -m src.cli export --experiment 1
```

After the run completes, open the Marimo notebook for Chapter 4 figures:

```bash
pip install marimo
marimo edit notebooks/analysis.py
```

---

## Implementation status

| File                                 | Status        | Notes                                                                          |
|--------------------------------------|---------------|--------------------------------------------------------------------------------|
| `src/datasets/multihop.py`           | implemented   | Loads `corpus` + `MultiHopRAG` HF configs; URL-keyed chunks; idempotent re-run |
| `src/datasets/ragtruth.py`           | implemented   | `hallucination` derived from non-empty `labels` span list                      |
| `src/evaluation/runner.py`           | implemented   | Per-query failures persist a stub row so resume skips them                     |
| `src/systems/system_b.py`            | partial       | Loop runs; `tokens_in/out` and `cost_usd` still 0 — affects B/D `$/correct`    |
| `src/retrieval/opensearch_client.py` | hybrid stub   | `hybrid_search` falls back to k-NN; needs OS search pipeline or client-side RRF |
| `src/evaluation/metrics.py`          | done          | Runner uses `contains_match`; switch to `exact_match` for MultiHop paper compliance |

Everything else (calibration, metrics aggregation, CLI, tracing, schema,
migrations, Phoenix wiring, Bedrock client) is wired and runnable.

### Ready to evaluate — checklist

Run these in order; each step is idempotent and safe to retry.

```bash
# 0. One-off setup
cp .env.example .env                  # fill in AWS keys
# (macOS/Linux: fix the AWS mount in docker-compose.yml — see Quick start note)

# 1. Stack up + schema
docker compose up -d opensearch postgres phoenix
docker compose build api
docker compose run --rm api alembic upgrade head
docker compose run --rm api python -m src.cli healthcheck   # expect 5 green lines

# 2. Load data
docker compose run --rm api python -m src.cli ingest-dataset multihop
docker compose run --rm api python -m src.cli ingest-dataset ragtruth
docker compose run --rm api python -m src.cli index-corpus multihop

# 3. Calibrate the HHEM gate
docker compose run --rm api python -m src.cli calibrate
# Inspect /data/results/threshold.json — set HHEM_THRESHOLD in .env to that value.

# 4. Smoke-test one system on a handful of queries before the full run
docker compose run --rm api python -m src.cli run-experiment \
  --name smoke --systems A --datasets multihop --limit 5
docker compose run --rm api python -m src.cli compute-metrics --experiment 1

# 5. Full run
docker compose run --rm api python -m src.cli run-experiment \
  --name dissertation-final --systems A,B,C,D --datasets multihop
docker compose run --rm api python -m src.cli compute-metrics --experiment 2
docker compose run --rm api python -m src.cli export --experiment 2
```

### Known caveats that affect headline numbers

- **B/D cost is under-reported** until `system_b.py` accumulates tokens/cost from the
  instructor raw response. `$/correct` and `total_cost_usd` for those two systems will
  read as 0 in the metrics table.
- **Accuracy denominator includes failed runs** (rows with `is_correct IS NULL`) in
  `compute-metrics`. Either delete those rows before aggregating, or filter the
  denominator in `cli.py:compute_metrics`.
- **Dataset field names** for `ParticleMedia/RAGTruth` were coded against the
  documented RAGTruth schema; if HF returns a different shape on first ingest, adjust
  the field accessors in `src/datasets/ragtruth.py`.

---

## Architectural choices and trade-offs

See `rag-benchmark-stack-guide-v2.md` for the full design doc.

The short version:

- **OpenSearch over Qdrant**: matches your prod, identical client code in lab and prod.
- **BAAI/llm-embedder over OpenAI embeddings**: matches the MultiHop-RAG paper baseline.
- **LiteLLM over a custom Bedrock client**: one env var swaps the entire LLM backend; cost is auto-tracked into `runs.cost_usd`.
- **Instructor for the agent decision**: cannot produce a malformed action, ever.
- **Phoenix over Streamlit/LangSmith**: self-hosted OTel, no credentials in the repo, drill-down per query.
- **SQLite for Phoenix's metadata** (default): simpler than wiring a second Postgres DB. Switch to Postgres only if you hit a concrete limit.
- **Hand-rolled exact-match for answer correctness, not an LLM judge**: deterministic for a thesis.
- **Resumable runner** via `UNIQUE(experiment_id, system, query_id)`: re-running the CLI command after a Bedrock throttle picks up exactly where it stopped.

---

## Cost

Full eval is <$10 total. LiteLLM writes the per-call cost into each `runs.cost_usd`,
and `metrics.cost_per_correct` = SUM(cost) / SUM(is_correct) per system.

---

## Troubleshooting

**`bedrock FAILED` in healthcheck.** Check that your AWS keys have Bedrock
access in `eu-west-2` and that the `LITELLM_MODEL` string matches a model your
account can invoke. For cross-region inference, prepend the region prefix:
`bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0`.

**`hhem FAILED` on first run.** The HHEM weights are baked into the image
during `docker compose build api`. If the build was interrupted, rebuild
with `docker compose build --no-cache api`.

**OpenSearch refuses to start.** It needs `vm.max_map_count >= 262144` on
the host. On Linux: `sudo sysctl -w vm.max_map_count=262144`.

**Phoenix port collision.** Phoenix uses 6006 (TensorBoard's default).
Change the host-side mapping in `docker-compose.yml` if needed.
