# Dissertation Audit — Aims/Objectives vs Implementation

Audit of the dissertation slides (Gaps, RQ1–4, Aims A1–A4, Objectives O1–O6)
against this codebase, plus the action register that falls out of it.

**How to use this file:** read it INSTEAD of re-auditing the codebase. Update
statuses in place when an action is done (`[ ]` → `[x]`, verdicts ✅/⚠️/❌).
Evidence is given as `file:symbol` references. Audited at commit `59ccec3`
(2026-06-11).

Legend: ✅ on point · ⚠️ partial (wording fix or pending run) · ❌ missing/wrong.

---

## 1. Slide ↔ repo mapping

| Slide term | Repo reality |
|---|---|
| S1 | System A — naive (`systems/system_a.py`) |
| S2 | System B — LangGraph iterative agent (`systems/system_b.py`) |
| S3 | System F — query decomposition (`systems/system_f.py`) |
| S4 | System F-tuned — stacked levers (`systems/system_f_tuned.py`) |
| "3 LLMs / model tiers" | Per repo evidence: Haiku 4.5, Nova Lite, Qwen3 — **CONFIRM final matrix models** |
| "1 fixed retriever" | `retrieval/retrieve.py:retrieve` — hybrid BM25+dense+RRF + cross-encoder rerank, shared by all systems |

---

## 2. Aims checklist

| Aim | Claim | Verdict | Evidence / gap | Actions |
|---|---|---|---|---|
| A1 | Controlled platform: 4 strategies on one shared retrieval substrate | ✅ | Shared `retrieve()`; A/B/F share `ANSWER_SYSTEM_PROMPT`; `SYSTEM_REGISTRY` = exactly A,B,F,F-tuned. Caveats: F-tuned also varies prompt + context budget; environment provenance not recorded per experiment | W7, C2 |
| A2 | 12 runs (4×3), accuracy + Token F1, stratified incl. null | ⚠️ | Seeded stratified `--sample` records `query_ids` (`runner.py`) ✅; accuracy ✅; **Token F1 not implemented anywhere** ❌; matrix not yet run on a frozen pipeline | C1, P1, P2, W5 |
| A3 | Cost:quality — per-query cost, cost-per-correct, Pareto frontier | ✅* | `cost_usd` per call + `cost_per_token` fallback; `metrics.cost_per_correct`; notebook Pareto scatter. *Slide says "LiteLLM **proxy**" — repo deliberately uses the **SDK** (CLAUDE.md decision list). Frontier-per-accuracy-level derivation pending | W2, N5, C7, C6 |
| A4 | Metric divergence audit + retrieval ceiling | ❌ | **Only aim with zero implementation.** `compute-metrics` prints contains/exact/CRAG side by side but no agreement/correlation; no ceiling / failure attribution; Token F1 missing; System B drops `all_retrieved_ids` before persisting | C1, C3, N3, N4 |

## 3. Objectives checklist

| Obj | Claim | Verdict | Notes | Actions |
|---|---|---|---|---|
| O1 | Sophisticated retrieval across all four pipelines | ✅ | Hybrid BM25+dense+RRF+rerank, single entry point. Reword "a sophisticated retrieval" → "a shared hybrid retrieval substrate" | — |
| O2 | S1–S4 "in a single LangGraph framework" + full cost tracing | ⚠️ | Cost tracing ✅ (tokens + `cost_usd` per run, Phoenix spans). **LangGraph claim wrong: only System B uses LangGraph**; A/F/F-tuned are plain Python on the `System` protocol | W1 |
| O3 | Evaluate all 12 system–model combos on stratified sample | ⚠️ | Capability ✅; not yet executed as a frozen matrix. Slide omits **null** question type (dataset + stratification + refusal-equivalence scoring all include it). Grammar: "a … sample sets", duplicated "stratified" | P1, P2, W4 |
| O4 | Quantify quality via contains-match + Token F1 across types | ⚠️ | contains ✅ but **adapted** (post-marker, refusal-equivalence, entity-suffix layers in `evaluation/metrics.py:contains_match` — must be disclosed); by-type ✅ (`metrics-by-type`, `/by-type`); **Token F1 ❌** | C1, W3 |
| O5 | Cost-per-correct per system–model pair + Pareto frontier | ✅* | Per-experiment ✅; cross-model frontier needs the 12 runs + frontier derivation | N5, P1 |
| O6 | Metric divergence + retrieval-ceiling attribution | ❌ | Nothing computes either; data is sufficient (`runs.retrieved_chunk_ids` × `queries.relevant_chunk_ids` × `is_correct`) | N3, N4, C3 |

## 4. RQ coverage

| RQ | Status | Notes |
|---|---|---|
| RQ1 accuracy by orchestration + type | ✅ now | `contains_match` + by-type breakdown |
| RQ2 cost per correct + latency | ✅ cost / ⚠️ latency | Latency captured per-run (`runs.latency_ms`) but **never aggregated** — no metrics column, no notebook cell. Cross-provider latency = serving-infra confound, scope the claim |
| RQ3 ranking across models | ⚠️ | Needs frozen 12-run matrix (same SHA/sample/index) + rank-stability stat. "Model strength" wording overreaches if all 3 models are budget-class |
| RQ4 predictability (rank stability + variance) | ❌ until N1/N2 + P1 | No rank-stability or variance computation exists yet |

Gap slide citations: Trivedi 2023 (IRCoT) + Ammann 2025 fit "cost as side issue".
Shi et al. 2024 ambiguous (REPLUG vs the 2023 distraction paper) — verify bib.
Asai (Self-RAG) is arXiv 2023 / ICLR 2024 — check year style. Gao 2024 + Barnett
2024 support "frontier-only evidence" only **indirectly** — soften or re-cite.
Verify the Ammann quote against the PDF before verbatim use (CLAUDE.md already warns).

---

## 5. Action register

### C — Code changes

**Before the final runs:**
- [x] **C2** *(done, commit pending)* Provenance now recorded in
  `experiments.config_json` (`evaluation/runner.py`): `git_sha` (via `_git_sha()`
  — reads `GIT_SHA` env, falls back to `git rev-parse`), `reranker_model`,
  `rerank_provider`, `retrieval_pool`, `hhem_threshold`, and a per-dataset
  `corpus` fingerprint (`_corpus_fingerprint()` → n_chunks, n_passage_chunks,
  granularity). **Action for final runs:** export `GIT_SHA=$(git rev-parse HEAD)`
  when invoking `run-experiment` inside the container (no `.git` is mounted there).
- [ ] **C3** *(optional — decide)* Persist System B's accumulated `all_retrieved_ids`
  (currently dropped at the end of `system_b.py:answer`) if an "evidence ever seen"
  ceiling for B is wanted; final-context-only is what's stored today.
- [ ] **C6** *(decision)* Accuracy denominator includes `is_correct IS NULL` stub rows
  (CLAUDE.md known gap #4) and stubs carry zero cost. Either fix the denominator or
  adopt + state a "failures count as wrong" policy (defensible for predictability).
- [ ] **C7** Exact-pin `litellm` in `api/requirements.txt` for the final phase —
  its pricing map changes between versions and feeds `cost_usd` (RQ2).
- [ ] **C8** Verify index granularity purity: no mixed `<url>` and `<url>#p<i>`
  chunk IDs for multihop (ingest is additive; `index_corpus` indexes ALL chunks).
  If mixed → wipe Postgres chunks + OpenSearch index, re-ingest, re-index.
  *Now auto-flagged*: every new experiment's `config_json.corpus[ds].granularity`
  reads `"mixed"` when both granularities coexist (C2 fingerprint) — but acting on
  the flag (the wipe/re-ingest) is still manual.

**Post-hoc safe (answers stored; recomputable over historical runs):**
- [x] **C1** *(done, commit pending)* `evaluation/metrics.py:token_f1` — SQuAD-style
  token-overlap F1 over post-marker text, with refusal-equivalence so null-type is
  meaningful (entity-suffix stripping intentionally omitted to keep it a clean
  lexical metric). Persisted as `metrics.avg_token_f1` (migration `0003_token_f1`),
  surfaced in `compute-metrics`, `metrics-by-type`, and the `/metrics` + `/by-type`
  API routes. Recomputable over historical experiments via `compute-metrics`.
  **Action for the user:** run `alembic upgrade head` before the next `compute-metrics`.
  Unblocks slide wording W3.
- [ ] **C4** Latency aggregation (avg/p50/p95 per system) — notebook minimum,
  optional `metrics` column.
- [ ] **C5** `rescore` CLI command to re-score historical runs under the current
  `contains_match` (metric changed mid-project; stored `is_correct` is frozen at
  run-time code) — or decide headline numbers cite fresh runs only.

### N — Analysis (in `notebooks/analysis.py`)
All five added as marimo cells after the existing per-type cell, plus a shared
helper cell (imports the canonical `metrics.py` scoring so notebook numbers match
`compute-metrics`; defines `kendall_tau_b`, `bootstrap_ci`, `pareto_frontier`,
`covered`, `agreement`). Math + exact cell logic verified standalone on synthetic
data; the live versions need a populated Postgres. *(All done, commit pending.)*
- [x] **N1** Rank stability: `kendall_tau_b` (tie-aware, hand-rolled — no scipy dep)
  over per-system accuracy across experiments, labelled by model. Degrades to a
  "need ≥2 experiments" message until the matrix is run. (Found+fixed a diagonal
  duplicate-column bug here via the cell-logic test.)
- [x] **N2** Per-system accuracy bootstrap CI + latency p50/p95/mean/std + cost
  mean/std/total/per-correct (`variance_tbl`). Works on a single experiment.
- [x] **N3** Retrieval ceiling + failure attribution (`ceiling` + stacked-bar):
  coverage = fraction with ≥1 gold article retrieved (the ceiling); every error
  partitioned into `err_retrieval` (no evidence) vs `err_generation` (had evidence,
  still wrong) — verified `err_retrieval + err_generation == 1 − accuracy`.
  Null-type queries (no gold) excluded. NOTE: uses each run's persisted
  `retrieved_chunk_ids`, whose semantics differ per system (A top-5; B final-step
  top-5; F full fused; F-tuned 10) — so "coverage" is *evidence in the answering
  context*, not *ever seen* (that needs C3 for System B).
- [x] **N4** Metric agreement matrix + heatmap (`agree_mat`): pairwise agreement
  rate over contains / exact / tokenF1≥.5 / CRAG-good, recomputed under current
  scoring code (sidesteps stale stored `is_correct`, C5). CRAG cells use judged
  runs only.
- [x] **N5** Pareto frontier overlay (`fig_pareto`): marks non-dominated
  (accuracy, cost-per-correct) points over the scatter.

### P — Final-run protocol (procedure, no code)
- [ ] **P1** One wipe + re-ingest + re-index; freeze git SHA; run all 12
  (4 systems × 3 models) back-to-back on the same machine against the same index build.
- [ ] **P2** ONE fixed seeded stratified sample reused across all 12 runs
  (identical `query_ids` — verify via `config_json.selection` across experiments).
- [ ] **P3** Record hardware (CPU, RAM) — embedder/reranker/HHEM are CPU-bound and
  shape latency numbers.
- [ ] **P4** Judge pass post-hoc with a fixed `JUDGE_MODEL` independent of generation,
  same judge for all 12 experiments.

### W — Slides / dissertation wording (no code)
- [ ] **W1** "implement S1–S4 in a single LangGraph framework" → "in a single
  benchmark harness with a shared system interface" (only S2/B uses LangGraph).
- [ ] **W2** "via LiteLLM proxy" → "via the LiteLLM SDK" (no proxy container, by
  documented decision).
- [x] **W3** *(unblocked)* Token F1 now exists (C1), so the slides may keep it —
  but only report numbers once `alembic upgrade head` + `compute-metrics` have
  populated `avg_token_f1`. Until then the column is NULL.
- [ ] **W4** Objective 3: add the **null** question type; fix "stratified …
  stratified" and "a … sample sets".
- [ ] **W5** "4 × 3 experimental sample sets" → "12 runs over one fixed stratified
  sample" (plural implies different samples per cell → breaks comparability).
- [ ] **W6** RQ3 "model strength" → "across heterogeneous budget-class models", or
  add one stronger anchor model run to claim a strength axis.
- [ ] **W7** Present A/B/F as the controlled single-variable comparison; F-tuned as
  a stacked engineering system (it also changes prompt + context budget), so it
  doesn't undermine the Gap-1 confound claim.
- [ ] **W8** Citations: pin down Shi et al. 2024; check Asai year; soften/re-cite
  Gap 3 (Gao/Barnett indirect); verify Ammann quote vs PDF.
- [ ] **W9** Scope cross-provider latency claims (serving infrastructure confound;
  systems run sequentially so time-of-day load differs per system).

### D — Docs housekeeping (separate commit; low priority)
- [ ] **D1** CLAUDE.md drift: retrieval is hybrid+rerank (not plain kNN); B1/B3/B5
  removed from `SYSTEM_REGISTRY`; passage granularity is now the ingest default;
  config defaults changed (`litellm_model` = Nova Lite, `hhem_threshold` = 0.10);
  `embed_one` is not lru_cached (only the model is).
- [ ] **D2** README drift: "Implementation status" table calls `hybrid_search` a
  stub (it isn't); retrieval-eval article-granularity caveat predates passage
  chunking.

---

## 6. Bottom line

Done so far: **C1** (token F1), **C2** (provenance + corpus fingerprint), and the
full **N-series** (N1 rank stability, N2 variance/CIs, N3 retrieval ceiling +
failure attribution, N4 metric agreement, N5 Pareto frontier) in the notebook.
A4/O6 now have real implementation behind them; RQ3/RQ4 have their analysis code.

Remaining to *claim* the gated items is now mostly **running**, not building:
- A2/O3, RQ3, RQ4 → run the 4×3 matrix on a frozen pipeline (P1/P2), then the
  N1/N2 cells populate. Set `GIT_SHA=$(git rev-parse HEAD)` per run (C2).
- A4/O6 → `alembic upgrade head` + `compute-metrics` (token F1 column) and a
  `judge` pass (CRAG column), then the N3/N4 cells render over real data.
- A3/O5 → already claimable per-experiment; N5 frontier + the matrix complete it.

Claim confidently now: A1, A3 (modulo W2), O1, O5, RQ1, RQ2-cost.
Still factually wrong as currently worded (slide-only edits): "single LangGraph
framework" (W1), "LiteLLM proxy" (W2).

Verification status: all added Python compiles; `token_f1` and every N-cell's
math + pandas logic were unit-tested on synthetic data. Nothing here has been run
against the live Postgres/OpenSearch/Bedrock stack — that's the user's environment.
