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
- [ ] **C2** Record provenance in `experiments.config_json` (`evaluation/runner.py`):
  git SHA, `reranker_model`, `rerank_provider`, `retrieval_pool`, `hhem_threshold`,
  chunk granularity / corpus fingerprint (n_chunks). Without this the DB cannot
  prove environment constancy across the 4×3 matrix.
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

**Post-hoc safe (answers stored; recomputable over historical runs):**
- [ ] **C1** Implement token-level F1 in `evaluation/metrics.py` (SQuAD-style, over
  post-marker answer text), surface in `compute-metrics` + `metrics-by-type`.
  Decide: persisted `metrics` column (migration 0003) vs notebook-only.
  Required by A2/O4/A4 slides; enables comparison with Ammann et al.'s answer-F1.
- [ ] **C4** Latency aggregation (avg/p50/p95 per system) — notebook minimum,
  optional `metrics` column.
- [ ] **C5** `rescore` CLI command to re-score historical runs under the current
  `contains_match` (metric changed mid-project; stored `is_correct` is frozen at
  run-time code) — or decide headline numbers cite fresh runs only.

### N — Analysis (in `notebooks/analysis.py`)
- [ ] **N1** Rank stability across models: Kendall's τ / Spearman over system
  rankings per metric (RQ3/RQ4 deliverable).
- [ ] **N2** Cost + latency variance per strategy; bootstrap CIs for accuracy and
  cost-per-correct (currently single deterministic run, no uncertainty anywhere).
- [ ] **N3** Retrieval ceiling / failure attribution: accuracy conditioned on gold
  evidence coverage in retrieved context; baseline ceiling = System A retrieval or
  `retrieval-eval`. Note per-system semantics of `retrieved_chunk_ids` differ
  (A: top-5 used; B: final-step top-5; F: full fused list; F-tuned: 10 used).
- [ ] **N4** Metric agreement: contains vs exact vs token-F1 vs CRAG — agreement %,
  correlation, per question type (A4/O6 deliverable).
- [ ] **N5** Pareto frontier derivation: cheapest strategy per accuracy level
  (extends the existing cost-accuracy scatter).

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
- [ ] **W3** Keep "Token F1" on slides only if C1 is implemented; else say "exact match".
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

## 6. Bottom line (as of audit date)

Claim confidently now: A1, A3 (modulo W2), O1, O5, RQ1, RQ2-cost.
Claim only after action items: A2/O3 (C1 + P1/P2), A4/O6 (C1 + N3/N4),
RQ3 (P1/P2 + N1 + W6), RQ4 (N1/N2 + P1).
Factually wrong as currently worded: Token F1 (3 slides), "single LangGraph
framework", "LiteLLM proxy".
