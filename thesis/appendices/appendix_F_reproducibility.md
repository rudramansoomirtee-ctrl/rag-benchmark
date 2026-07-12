# Appendix F — Reproducibility Record

> Appendix draft — content copied from `CLAUDE.md`, `thesis/musique_matrix_analysis.md`,
> `DISSERTATION_AUDIT.md`, and (hardware only, §F.7) `thesis/chapter3_methodology.md` §3.8;
> renumber/reformat at Word conversion.

## F.1 Frozen configuration values

Copied from `CLAUDE.md`, Config (`api/src/config.py`) — defaults shown, env-overridable:

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
top_k                   = 20                     # uniform answer budget; A/A-minus answer over their top-20, = fused_answer_top_k (removes the old A=10 asymmetry)
fused_answer_top_k      = 20                     # answer-context budget for B/F/F-seq (fused top-N); = top_k so the budget is uniform across all 8 systems
retrieval_pool          = 40                     # hybrid first-stage pool before rerank (~2× top_k so the reranker selects, not just reorders)
reranker_model          = "BAAI/bge-reranker-v2-m3"   # used only when rerank_provider="local"
rerank_provider         = "bedrock-cohere"       # DEFAULT: Cohere Rerank 3.5 via Bedrock (cohere.rerank-v3-5:0, eu-central-1). "local" = free CPU cross-encoder. NB Cohere rerank is a metered API cost NOT in runs.cost_usd
max_agent_steps         = 5
hhem_threshold          = 0.10                   # fixed empirical threshold for HHEM-2.1-open on news (RAGTruth calibration removed) — NB: HHEM computation itself is descoped; see F.6
```

The final-matrix run additionally froze, per `thesis/musique_matrix_analysis.md` header:

> **Frozen config:** git SHA `12f2a49` (E3: `ec457dc`, verified inert — resume-logic commit only),
> seed 42, `top_k=20`, `fused_answer_top_k=20`, `retrieval_pool=40`, `max_agent_steps=5`, Cohere Rerank
> 3.5 (Bedrock, eu-central-1), `BAAI/llm-embedder`, LiteLLM 1.83.0, temperature 0.

MultiHop-RAG arm (E4–E6) ran at a later, verified-inert SHA:

> **Completed 2026-07-11 at SHA `d03dd3b`** (trace-feature WIP stashed during the run to preserve the
> frozen-SHA guarantee).

And, per `thesis/musique_matrix_analysis.md` §10 closing note:

> E4–E6 ran at SHA `d03dd3b` (Part I at `12f2a49`/`ec457dc`); the intervening commits touched thesis
> prose and the resume/billing fix only — no retrieval, scoring, or generation semantics — and the
> budget/prompt/config snapshot is field-identical across all six experiments (verified in
> `config_json`).

`max_tokens = 800` (per-call generation cap) is confirmed as a frozen value in the anomaly-scan
evidence (Appendix E.3): "`max_tokens=800` cap held for every single-call system (A max 582)."

`temperature = 0` for every LLM call is stated repeatedly across `CLAUDE.md` (Conventions:
"Determinism wherever LLMs are called (`temperature=0`)") and in the final-matrix frozen-config line
above.

## F.2 Experiment IDs

Copied from `thesis/musique_matrix_analysis.md` header and `DISSERTATION_AUDIT.md` §5d:

| Arm | Dataset | Model | Experiment id |
|---|---|---|---|
| E1 | MuSiQue | DeepSeek-V3 | 50 |
| E2 | MuSiQue | Qwen3-32B | 51 |
| E3 | MuSiQue | Nova Lite | 53 |
| E4 | MultiHop-RAG | DeepSeek-V3 | 54 |
| E5 | MultiHop-RAG | Qwen3-32B | 56 |
| E6 | MultiHop-RAG | Nova Lite | 57 |

> **Part I (§1–§8): MuSiQue arm, E1–E3 (ids 50, 51, 53) · 3 models × 8 systems × 150 queries = 3,600
> runs · 0 failures.**
> **Part II (§9–§10): MultiHop-RAG arm, E4–E6 (ids 54, 56, 57) · 3 models × 8 systems × 200 queries =
> 4,800 runs · 1 failure (0.02%) — completed 2026-07-11 at SHA `d03dd3b`. Matrix total: 8,400 runs,
> $24.59 LLM spend.**

E4 resumability note, copied from `thesis/musique_matrix_analysis.md` §ahead of §9:

> E4 resumed from a 515-row partial via `--resume-id` with zero re-billing (the resumability fix
> working as designed).

Prior ablation-program experiment ids (exp36–43, DeepSeek-V3 only, superseded by 50/51/53/54/56/57)
are listed in Appendix B.3.

## F.3 Sample composition

**MuSiQue** — copied from `thesis/musique_matrix_analysis.md` header:

> Sample: 150 seeded stratified queries (78 2-hop / 45 3-hop / 27 4-hop), identical across every cell.

**MultiHop-RAG** — copied from the Part II header:

> 200 seeded stratified queries (64 inference / 67 comparison / 45 temporal / 24 null; seed 42),
> identical across every cell.

Both samples use `seed 42` (also stated in the frozen-config line, F.1) and are recorded exactly via
`config_json.selection` per the resumability/provenance mechanism described in `CLAUDE.md` §"Evaluation
pipeline" step 4 and `DISSERTATION_AUDIT.md` action C2.

## F.4 Systems (the 4×2 retrieval×orchestration factorial)

Copied from `CLAUDE.md` "The systems" section:

> Lineup: **A** (naive), **A-minus** (naive over a dense-kNN **semantic-search-only** retriever — no
> BM25/RRF/rerank; A-minus-vs-A isolates the retrieval-pipeline effect), **B** (agentic single-tool
> loop), **F** (PARALLEL query decomposition), **F-seq** (SEQUENTIAL self-ask decomposition — resolves
> each hop and carries the bridge answer forward into the next; shares F's few-shot decomposer).

`SYSTEM_REGISTRY` = A, A-minus, B, B-minus, F, F-minus, F-seq, F-seq-minus (8 systems), per
`DISSERTATION_AUDIT.md` §5c. Historical systems E, F-tuned, and G were removed from the codebase
before the final matrix (see `CLAUDE.md` top-of-file notes); their historical runs remain in the
database but are excluded from the final matrix.

## F.5 CLI command surface

Copied verbatim from `CLAUDE.md` "Operating the repo" → "CLI is the primary surface":

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

Per `DISSERTATION_AUDIT.md` action C2 / §6, the final-matrix invocation additionally prefixed each
`run-experiment` call with an explicit git SHA export (no `.git` is mounted in the container):

> export `GIT_SHA=$(git rev-parse HEAD)` when invoking `run-experiment` inside the container.

## F.6 Container stack

Copied from `CLAUDE.md` "Stack (4 containers, `docker-compose.yml`)":

| Service | Port | Role |
|---------------|------|------------------------------------------------|
| `opensearch` | 9200 | Dense + BM25 retrieval (HNSW / Lucene engine) |
| `postgres` | 5432 | Structured runs/metrics — source of truth |
| `phoenix` | 6006 | OTel trace store + dashboard |
| `api` | 8000 | FastAPI + LangGraph + HHEM + CLI + SPA |

> LLM is **AWS Bedrock** (Claude Haiku 4.5) via **LiteLLM**. Cost is read from
> `response._hidden_params["response_cost"]` and persisted per run.

Note: the container-role table above still names "HHEM" per its original text in `CLAUDE.md`, but
`CLAUDE.md` elsewhere states HHEM/faithfulness computation is **descoped** (the `faithfulness/hhem.py`
source was removed; `runs.hhem_score`/`flagged` are never written). Both statements are copied
verbatim from the same source file — the descope note supersedes the stack-table's role description
for faithfulness scoring specifically.

## F.7 Hardware

Not present in the four primary source documents listed in the task brief, but resolved as a
previously-open placeholder (`DISSERTATION_AUDIT.md` §5d: "remaining placeholder: hardware CPU/RAM in
Ch3 §3.8") and located, filled in, in `thesis/chapter3_methodology.md` §3.8:

> Hardware is recorded because the embedder runs locally and shapes latency (Intel Core Ultra 7 155U,
> 16 GB RAM, Windows 11 Pro, Docker Desktop/WSL2).

Also consistent with `CLAUDE.md` "Environment setup notes":

> `docker-compose.yml:83` mounts `C:\Users\MadhavS\.aws:/root/.aws:ro` — **Windows-only**. macOS/Linux
> users must change to `${HOME}/.aws:/root/.aws:ro` or remove the mount and rely on env vars.

## F.8 Total cost

Copied from `thesis/musique_matrix_analysis.md` header:

> Matrix total: 8,400 runs, $24.59 LLM spend.
>
> LLM spend: **$7.09** (DeepSeek $5.13 / Qwen $1.39 / Nova $0.57); Cohere rerank metered separately (not
> in `cost_usd`). [MuSiQue arm]

MultiHop-RAG arm costs, copied from the Part II header:

> Costs: DeepSeek $12.79 / Qwen $3.16 / Nova $1.55.

Sum check: MuSiQue $7.09 + MultiHop $12.79+$3.16+$1.55 = $24.59 — matches the stated matrix total.

Cohere Rerank 3.5 cost is explicitly disclosed as **not** included in `cost_usd` or the $24.59 total,
per `CLAUDE.md` config (`rerank_provider` note: "NB Cohere rerank is a metered API cost NOT in
runs.cost_usd") and `thesis/musique_matrix_analysis.md` §5: "Cohere rerank is a separately-metered
per-retrieval charge borne by hybrid systems only (disclosed in Ch3 §3.5)."

## F.9 Resumability and idempotency guarantees

Copied from `CLAUDE.md` "Resumability invariant — do not break":

> - Re-running the CLI command must pick up where it stopped.
> - Per-query exceptions persist a stub row (`answer = NULL`) so resume skips them.
> - To retry failed rows: `DELETE FROM runs WHERE answer IS NULL` and re-run.

Enforced via the database constraint `UNIQUE(runs.experiment_id, runs.system, runs.query_id)` (see
`CLAUDE.md` "Database schema").

---

**Not found in sources:** a Docker/Python package version pin list (beyond `LiteLLM 1.83.0` and the
git SHAs already recorded) was not located as a single table in the four reviewed source documents —
`CLAUDE.md` states version provenance is captured *per experiment* at runtime
(`config_json.litellm_version`) rather than hard-pinned in `requirements.txt` (see `DISSERTATION_AUDIT.md`
action C7), so no single global version table exists to copy. OpenSearch/Postgres/Phoenix image tags
were not found specified in the reviewed documents (only `docker-compose.yml`, not reviewed as a
source per the task brief, would hold them).
