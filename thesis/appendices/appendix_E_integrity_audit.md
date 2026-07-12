# Appendix E — Independent Integrity / QA Audit

> Appendix draft — content copied from `thesis/musique_matrix_analysis.md` §1 and related sections,
> plus `DISSERTATION_AUDIT.md` §5d; renumber/reformat at Word conversion.

## E.1 Data integrity audit — MuSiQue arm (E1–E3), PASS on all seven checks

Copied verbatim from `thesis/musique_matrix_analysis.md` §1:

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Sample identity | **PASS** | Identical 150 `query_ids` byte-for-byte across E1/E2/E3 and across all 8 systems within each (all pairwise set-differences = 0) |
| 2 | Provenance constancy | **PASS** | All config identical except `model`. E3's SHA drift (`12f2a49`→`ec457dc`) diff-verified: only the `--resume-id` CLI flag + a pre-billing skip; `_run_one` (execution/scoring) moved verbatim, unchanged |
| 3 | Completeness | **PASS** | 1,200 rows each; 150/system; 0 NULL answers; 0 duplicate (system, query) pairs |
| 4 | Stratification | **PASS** | Exactly 78 / 45 / 27 by hop count |
| 5 | Scoring sanity | **PASS** | Spot-checks: alias matching and date-range normalisation working; incorrect rows are genuine misses (wrong entity, hallucination, refusal); no false positives/negatives found |
| 6 | Nova degradation evidence | **PASS** | F-family `n_steps` = 1.35–1.41 on Nova vs 3.54–3.68 on DeepSeek/Qwen; B ≈ 4 everywhere |
| 7 | Contamination | **PASS** | Stale exp 44 (`final-musique-nova-full`, old `top_k=10` config) identified — **must be excluded from all analysis**. Exp 54 = clean resumable MultiHop partial (515 rows). Nothing else conflicts |

The report's own verdict (§ Executive summary):

> **Verdict: the MuSiQue arm is clean, verified, and internally consistent — findings are correct with
> the phrasing qualified below.** Data integrity passes all 7 checks; every reported number traces to
> the raw runs.

## E.2 The 1 failed run (MultiHop-RAG arm)

Copied from `thesis/musique_matrix_analysis.md` header (Part I/II summary):

> **Part II (§9–§10): MultiHop-RAG arm, E4–E6 (ids 54, 56, 57) · 3 models × 8 systems × 200 queries =
> 4,800 runs · 1 failure (0.02%) — completed 2026-07-11 at SHA `d03dd3b`. Matrix total: 8,400 runs,
> $24.59 LLM spend.**

And, with the specific identifying detail, copied from the Part II header (§ preceding §9):

> **1 failure** (exp 57, System A, qid 175 — clean NULL stub, scored wrong per the crash-is-wrong
> policy; 0.02%).

Cross-referenced in `DISSERTATION_AUDIT.md` §5d:

> The frozen final matrix is done: **8,400 runs, 1 failure (0.01%), $24.59 LLM spend.**

(Note: the two source documents state the same single failure at two slightly different rounded
percentages — 0.02% in `musique_matrix_analysis.md` Part II header, 0.01% in `DISSERTATION_AUDIT.md`
§5d, both computed against 4,800 or 8,400 as the denominator respectively; both figures are copied
verbatim rather than reconciled, since the underlying count — 1 failed run — is identical in both.)

## E.3 Nova `max_tokens` overshoot — the 10-row / 6-query-id anomaly

Copied verbatim from `thesis/musique_matrix_analysis.md` §3.5 (Anomaly scan):

> `max_tokens=800` cap held for every single-call system (A max 582). **10 Nova rows across 6 query ids
> exceeded the cap on single calls** (max 2,340 tokens; qids 17845/17853/17868/17873) — LiteLLM likely
> dropped the param for Nova's API shape (`drop_params=True`). Benign (Nova near-free, correctness
> unaffected); footnote in the write-up.

Note on the query-id count: the source text lists **4 explicit query ids**
(17845/17853/17868/17873) while stating "6 query ids" overshot the cap in total — the remaining 2 ids
are not individually enumerated in the source document. Copied as-is; do not infer or invent the
missing 2 ids.

Also listed in the pre-print verification checklist (`thesis/musique_matrix_analysis.md` §8):

> Nova single-call `max_tokens` overshoot (10 rows across 6 query ids) — one-line footnote.

## E.4 Contamination check — excluded stale experiments

Copied from E.1, check #7 above: experiment 44 (`final-musique-nova-full`, run under an old
`top_k=10` configuration) was identified as contaminating and **excluded from all analysis**.
Experiment 54 was identified as a clean, resumable MultiHop-RAG partial run (515 rows) later
completed under the final SHA. No other conflicting historical experiments were found.

## E.5 Scoring-anomaly checks (sample identity, config constancy, completeness — cross-referenced)

These correspond to checks #1–#5 of E.1 above (sample identity, provenance/config constancy,
completeness, stratification, scoring sanity). No additional scoring-anomaly findings beyond those
listed in E.1 were located in the source documents.

## E.6 SHA / provenance audit note for the appendix (copied from `thesis/musique_matrix_analysis.md` §10 closing paragraph)

> One integrity audit item for the appendix: E4–E6 ran at SHA `d03dd3b` (Part I at
> `12f2a49`/`ec457dc`); the intervening commits touched thesis prose and the resume/billing fix only —
> no retrieval, scoring, or generation semantics — and the budget/prompt/config snapshot is
> field-identical across all six experiments (verified in `config_json`).

## E.7 Refuted pilot claim — flagged as an integrity/methods finding, not a data-quality defect

Copied from `thesis/musique_matrix_analysis.md` §4.1 (the correction that must propagate to all
documents):

| Pilot claim (n=50, DeepSeek, pre-final config) | n=150 verdict |
|---|---|
| "Dense-only wins on MuSiQue; B-minus (0.640) is the champion; the retriever effect is dataset-dependent and reverses on MuSiQue" | **REFUTED.** Hybrid wins all 24 cells, pooled p=5.5×10⁻⁵. The pilot reversal was small-sample noise compounded by config drift (`top_k` 10→20, prompt softening, `max_tokens` cap; the reranker did NOT change — pilots also used Cohere). |
| "F-seq ≫ F on deep hops (4-hop 0.444 vs 0.111, 4×)" | **Not replicated.** F-seq>F is directional-only; its edge concentrates at 2-hop; 3/4-hop are statistical ties. |

> **Reframe for the thesis:** the pilot-vs-final reversal is itself a *finding* — a documented,
> quantified case study in the unreliability of small-sample RAG evaluation (ties directly into RQ4
> and the field-wide single-run critique). Affected artefacts still carrying the stale claim:
> `DISSERTATION_AUDIT.md §5c`, Chapter 3/4 pilot boxes, `RELATED_WORK.md §8` framing, session memory.
> **None of the refuted claims may survive as live claims.**

This is included in the integrity appendix because it demonstrates the audit process caught and
corrected a finding between the pilot and final runs, rather than allowing a stale number to propagate
silently.

## E.8 Verification method statement

Copied from the closing note of `thesis/musique_matrix_analysis.md`:

> *Compiled from: integrity audit (27 checks), statistical verification (independently cross-checked;
> exact binomial sign tests, Kendall τ-b, failure attribution over 3,600 runs), and primary-source
> literature verification (arXiv PDF/HTML extraction). All numbers trace to Postgres experiments
> 50/51/53 at the frozen SHAs above.*

Also, from the report header:

> **Verification:** three independent audits (data-integrity; statistics, itself independently
> cross-checked; literature), compiled 2026-07-05. Metrics table verified to match recomputation from
> raw runs exactly in all 24 cells.

---

**Not found in sources:** the source document states "27 checks" were run as part of the integrity
audit, but only enumerates 7 in the formal table (§1, reproduced as E.1 here); the remaining ~20
checks referenced by that summary count are not individually itemised in
`thesis/musique_matrix_analysis.md`, `DISSERTATION_AUDIT.md`, or `CLAUDE.md`. The 2 unnamed Nova query
ids referenced in E.3 (source says "6 query ids," only 4 are listed) were also not found named
anywhere in the reviewed sources.
