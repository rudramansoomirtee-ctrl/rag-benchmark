# RAG Benchmark Stack

Reproducible Dockerised environment to evaluate Systems A/B/F against MultiHop-RAG and MuSiQue, with HHEM faithfulness scored on every run.

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

Open:
- **Testing console** at <http://localhost:8000/> — three-tab UI (Ask, Data, Experiments) for trying systems A/B/F/F-tuned, ingesting datasets, and browsing experiment results.
- **Phoenix** at <http://localhost:6006> — trace tree per query.
- **FastAPI docs** at <http://localhost:8000/docs> — raw API surface.

---

## Workflow

```bash
# Load datasets into Postgres
docker compose exec api python -m src.cli ingest-dataset multihop
docker compose exec api python -m src.cli ingest-dataset musique

# Index corpora into OpenSearch (MuSiQue uses a separate index)
docker compose exec api python -m src.cli index-corpus multihop
docker compose exec -e OPENSEARCH_INDEX=rag-chunks-musique api python -m src.cli index-corpus musique

# HHEM faithfulness uses a fixed threshold (settings.hhem_threshold, default 0.10).
# The old RAGTruth calibration step has been removed.

# Run the full eval (resumable — re-run on failure picks up where it left off)
docker compose exec api python -m src.cli run-experiment \
  --name dissertation-final \
  --systems A,B,F \
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

## System F — query decomposition (multi-hop)

System F decomposes a multi-hop question into 2–4 single-hop sub-questions (one
few-shot `instructor`-typed LLM call), retrieves for the original + each
sub-question over the **same** hybrid+rerank pipeline as A/B, RRF-fuses the
results, and answers once over the fused top-8 (`FUSED_ANSWER_TOP_K`, shared
with B's iteration fusion). Because the retriever is held constant, F-vs-A
isolates the *decomposition* effect. It's the decomposition rung of the comparison study, distinct from B's
iterative reformulation loop. No new deps, no keys (Bedrock only).

System F deliberately mirrors **Ammann, Golde & Akbik (2025)**, *Question
Decomposition for Retrieval-Augmented Generation* (ACL 2025 Student Research
Workshop) — the same drop-in recipe of LLM question decomposition plus an
off-the-shelf reranker, with no training or specialised indexing, evaluated on
MultiHop-RAG (they report MRR@10 +36.7% and answer-F1 +11.6% over standard RAG).
Their pipeline is, by their own account, **non-iterative**, and they name that as
a limitation:

> "Lack of iterative retrieval. The pipeline operates in a non-iterative fashion,
> which represents a limitation compared to more adaptive approaches."

That is precisely the gap this study fills: **System B is the iterative comparator
they lack.** Holding the retriever and the generator LLM constant across A
(naive), F (single-pass decomposition, ≈ Ammann et al.) and B (iterative
reformulation) isolates *decomposition vs. iteration vs. neither* on the same
benchmark — the contribution of this work relative to theirs. (Verify the quoted
sentence against the published PDF before citing it verbatim.)

```bash
docker compose run --rm api python -m src.cli run-experiment \
  --name decomp --systems A,B,F --datasets multihop --limit 20
docker compose run --rm api python -m src.cli compute-metrics --experiment <id>
```

## Ablations & secondary metrics

**Systems A / B / F / F-tuned.** The lineup is A (naive), B (iterative agent,
default 5-step budget, evidence RRF-fused across iterations), F (decomposition +
RRF), and F-tuned (F + top-10 budget + weighted RRF + source-aware retrieval +
CoT answer prompt). A/B/F hold the retriever, generator and fused answer budget
constant so the comparison isolates *orchestration*; F-tuned stacks accuracy
levers on top of F. (The earlier B1/B3/B5 iteration sweep and the
multi-tool System G were both **removed** after the runs showed no gain over the
baselines; historical runs remain in the DB.)

```bash
docker compose run --rm api python -m src.cli run-experiment \
  --name lineup --systems A,B,F,F-tuned --datasets multihop
docker compose run --rm api python -m src.cli compute-metrics --experiment <id>
```

**Secondary correctness metrics.** Primary correctness stays `contains_match`
(the MultiHop-RAG containment metric, Tang & Yang 2024). Two secondary columns are
added to the metrics table: normalized **exact-match** (`accuracy_exact`) and a
**CRAG LLM-as-judge** truthfulness score (`crag_score`, Yang et al. 2024 rubric:
perfect = 1, acceptable = 0.5, missing = 0, incorrect = −1). The judge is a
post-hoc, resumable pass over existing runs — it never re-runs the systems and only
scores rows it hasn't judged yet:

```bash
docker compose run --rm api python -m src.cli judge --experiment <id>          # all systems
docker compose run --rm api python -m src.cli judge --experiment <id> --system B5
docker compose run --rm api python -m src.cli compute-metrics --experiment <id>  # fills the two columns
```

The judge model is **independent of generation**: set `JUDGE_MODEL` (any LiteLLM
provider — `bedrock/…`, `deepseek/deepseek-chat`, `gemini/…`, `openai/…`) to pair
cheap generation with a strong, reliable judge. It defaults to `LITELLM_MODEL`.

**Per-question-type breakdown.** MultiHop tags each query `inference` /
`comparison` / `temporal` / `null` (in `queries.metadata['question_type']`).
`metrics-by-type` breaks accuracy out by type per system — the hypothesis being F
wins on comparison/temporal, B on inference. Pure post-hoc, no schema change:

```bash
docker compose run --rm api python -m src.cli metrics-by-type --experiment <id>
```

**Sampling for cost control.** Two ways to run fewer queries:
- `--limit 20` — first 20 by id, for a **quick smoke test** (check the pipeline + read real per-call cost). Not random; don't report it.
- `--sample 500 --seed 42` — a **defensible subset**: seeded random draw, stratified by question type by default, with the exact query IDs recorded in the experiment config so it's reproducible. Use `--no-stratify` for a plain random draw. `--sample` overrides `--limit`.

```bash
docker compose run --rm api python -m src.cli run-experiment --name smoke --systems A --datasets multihop --limit 20
docker compose run --rm api python -m src.cli run-experiment --name sub --systems A,B,F --datasets multihop --sample 500 --seed 42
```

## Retrieval validation (Tang & Yang Table 5)

`retrieval-eval` probes the retriever alone — independent of A/B/F — and reports
MRR@k, MAP@k, Hits@4 and Hits@k over the eval set:

```bash
docker compose run --rm api python -m src.cli retrieval-eval --dataset multihop --k 10
```

**Read this before comparing to the paper.** Two things make these numbers *not*
a like-for-like Table 5 row:

1. **Granularity (the big one).** This repo indexes MultiHop at **article/URL**
   granularity — one document per article — so a "hit" means retrieving the right
   *article*. Table 5 retrieves over **~256-token passages**, so a "hit" means the
   right *passage*. Finding the right article out of ~609 is a much easier task
   than the right passage out of several thousand, so our numbers will read
   **higher** than Table 5 and are not directly comparable.
2. **Pipeline.** Ours is hybrid (BM25 + `llm-embedder` dense, RRF) → `ms-marco-MiniLM`
   cross-encoder rerank. Table 5 rows are single dense embedders, optionally with
   `bge-reranker-large`. Different first stage, different reranker.

So `retrieval-eval` is a sound **"does the retriever surface the right articles"**
sanity check, and the right way to defend the downstream B-vs-F claims is to state
the article-level retrieval quality explicitly. A *literal* Table 5 replication
needs passage-level (256-token) chunking + a fact→passage gold mapping + a separate
index — a deliberate change to the URL-keyed design, not yet built. Compare against
the `llm-embedder` row of Table 5 in arXiv:2401.15391 (the best reranked row there
reaches ≈ Hits@10 0.747 / Hits@4 0.663).

## Logs & transparency

- **Live logs:** `LOG_LEVEL=INFO` (default) prints experiment/system milestones and per-query failures; **`LOG_LEVEL=DEBUG`** adds a per-run line (correct / steps / cost / latency) and System B's per-step decisions (reformulate vs. answer). Set it in `.env`.
- **SPA** (`localhost:8000/` → Experiments → click a row) shows: the run **config/selection** (sample, seed, model, top_k, max steps), the metrics table incl. **Exact / CRAG / Steps**, a **per-question-type** breakdown, and per-run rows with **judge label, steps, cost** and the gold answer (hover the answer cell).
- **Phoenix** (`localhost:6006`): the deepest view — every span (retrieve → decide → answer), tokens and cost per call.
- **Files:** `export` dumps full runs + metrics JSON; `metrics-by-type` and `retrieval-eval` also write JSON under `/data/results/`.

## Implementation status

| File                                 | Status        | Notes                                                                          |
|--------------------------------------|---------------|--------------------------------------------------------------------------------|
| `src/datasets/multihop.py`           | implemented   | Loads `corpus` + `MultiHopRAG` HF configs; URL-keyed chunks; idempotent re-run |
| `src/datasets/musique.py`            | implemented   | Loads `dgslibisey/MuSiQue`; per-question pooled paragraphs; hop count as `question_type`; separate OpenSearch index |
| `src/evaluation/runner.py`           | implemented   | Per-query failures persist a stub row so resume skips them                     |
| `src/systems/system_b.py`            | implemented   | Per-instance step budget; two-call route/execute (provider-robust); cost via `response_cost` + litellm-pricing fallback |
| `src/retrieval/opensearch_client.py` | implemented   | `hybrid_search` = client-side RRF over BM25 + dense kNN; `retrieve()` adds a cross-encoder rerank |
| `src/evaluation/metrics.py`          | done          | `contains_match` is the paper's containment primary; `exact_match` + `token_f1` + CRAG judge are secondaries |

Everything else (metrics aggregation, CLI, tracing, schema, migrations,
Phoenix wiring, Bedrock client) is wired and runnable. HHEM faithfulness uses a
fixed `settings.hhem_threshold`; the former RAGTruth calibration step is removed.

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
docker compose run --rm api python -m src.cli ingest-dataset musique
docker compose run --rm api python -m src.cli index-corpus multihop
docker compose run --rm -e OPENSEARCH_INDEX=rag-chunks-musique api python -m src.cli index-corpus musique

# 3. HHEM gate uses a fixed threshold (HHEM_THRESHOLD in .env, default 0.10) — no calibration step.

# 4. Smoke-test one system on a handful of queries before the full run
docker compose run --rm api python -m src.cli run-experiment \
  --name smoke --systems A --datasets multihop --limit 5
docker compose run --rm api python -m src.cli compute-metrics --experiment 1

# 5. Full run
docker compose run --rm api python -m src.cli run-experiment \
  --name dissertation-final --systems A,B,F --datasets multihop
docker compose run --rm api python -m src.cli compute-metrics --experiment 2
docker compose run --rm api python -m src.cli export --experiment 2
```

### Known caveats that affect headline numbers

- **B cost** is tracked from `response_cost`, falling back to `litellm.cost_per_token`
  from the usage token counts when the instructor response omits it — so System B's
  `$/correct` no longer silently reads 0 (requires the model in litellm's pricing map).
- **Accuracy denominator includes failed runs** (rows with `is_correct IS NULL`) in
  `compute-metrics`. Either delete those rows before aggregating, or filter the
  denominator in `cli.py:compute_metrics`.
- **MuSiQue retrieval is scoped to its own OpenSearch index** (`rag-chunks-musique`);
  set `OPENSEARCH_INDEX=rag-chunks-musique` when ingesting/indexing/running MuSiQue so
  its per-question paragraphs never mix with the MultiHop news corpus.

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
