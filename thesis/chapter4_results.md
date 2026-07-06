# Chapter 4 — Results

> **Status: MuSiQue populated (verified n=150); MultiHop pending.** The MuSiQue arm of the final matrix
> (E1–E3: DeepSeek-V3, Qwen3-32B, Nova Lite; 3,600 runs; 0 failures) is complete, verified by three
> independent audits, and its numbers are filled in below. The MultiHop-RAG arm (E4–E6) has **not** yet
> run — cells that need it are marked `[MultiHop — E4–E6 pending]` and are **not** invented. Full
> statistical backing (paired tests, effect sizes, behavioural sub-studies) lives in
> `thesis/musique_matrix_analysis.md`; this chapter reports the headline numbers and the interpretation.
>
> **Voice note:** this is a data-backed draft skeleton — the analysis paragraphs state the findings and
> the required framing, but must be expanded into your own prose before submission. Resolve remaining
> `[CONFIRM]`/`[INSERT]` markers (MultiHop N, hardware) as they become available.

**Marking hook: analysis of findings is the highest-weighted criterion (/30).** Every section maps to one
research question and cites the verified evidence (experiment ids 50/51/53; frozen SHA `12f2a49`).

---

## 4.1 Overview of the executed evaluation

The evaluation comprises **two studies over one frozen substrate**, reported separately because they
manipulate different variables (Study 1 = orchestration, retriever fixed; Study 2 = retriever, the full
4×2 factorial). **System nomenclature** (codes in tables, full names in prose — Table 3.1): **A** =
Single-pass RAG · **B** = Iterative RAG · **F** = Parallel decomposition · **F-seq** = Sequential
decomposition (Self-Ask); each has a dense-kNN-only twin **A-/B-/F-/F-seq-minus**.

**Provenance (MuSiQue arm, confirmed from `experiments.config_json`):**

| Field | Value |
|---|---|
| Git SHA | `12f2a49` (E3 `ec457dc`, verified inert — resume-logic only) |
| Models | Qwen3-32B · DeepSeek-V3 · Nova Lite (AWS Bedrock, LiteLLM SDK, temp 0) |
| Dataset / N / seed | MuSiQue **150** (78 2-hop / 45 3-hop / 27 4-hop), seed 42; MultiHop-RAG `[E4–E6 pending]` |
| Reranker | Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`, eu-central-1) |
| Answer-context budget | 20 chunks, uniform (`top_k=20` for A/A-minus; `fused_answer_top_k=20` for the rest) |
| Identical `query_ids` across all cells | **confirmed** (integrity audit, all 24 cells) |
| Runs / failures | 3,600 / **0** |

**Model panel (RQ3/RQ4 scope).** Heterogeneous cost-efficient models spanning a capability gradient —
Nova Lite (weak) < Qwen3-32B (mid) < DeepSeek-V3 (strong-but-cheap) — not a frontier panel; cross-model
claims are framed accordingly (§3.5). One panel property is itself a result: **Nova Lite cannot execute
the decomposition systems** — its structured-output decoder parse-fails on ~85% of queries, so F/F-seq
degrade to naive retrieval (measured n_steps 1.4 vs 3.5–3.7 on the other models). Nova's F-family cells
are reported as this degradation (≈ Nova A), not as decomposition, and constitute the robustness finding
(§4.6).

---

## 4.2 Study 1 — Accuracy by orchestration strategy — RQ1

*Evidence: `compute-metrics`, `metrics-by-type` (exp 50/51/53). Metric: alias-aware containment accuracy.*

**Table 4.1 — Containment accuracy, 8 systems × 3 models, MuSiQue (n=150).** (Nova F-family marked † =
decompose parse-fails → degraded to ≈ Nova A; §4.6.)

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A (naive) | 0.487 | 0.473 | 0.273 |
| A-minus | 0.420 | 0.420 | 0.260 |
| **B (iterative)** | **0.513** | **0.527** | **0.347** |
| B-minus | 0.447 | 0.467 | 0.307 |
| F (parallel decomp.) | 0.453 | 0.440 | 0.300 † |
| F-minus | 0.427 | 0.420 | 0.260 † |
| F-seq (sequential decomp.) | 0.480 | 0.480 | 0.320 † |
| F-seq-minus | 0.433 | 0.467 | 0.260 † |

*(MultiHop-RAG equivalent: `[E4–E6 pending]`.)*

**Table 4.2 — Accuracy by hop count, MuSiQue, DeepSeek-V3** (2-hop n=78 / 3-hop n=45 / 4-hop n=27):

| System | 2-hop | 3-hop | 4-hop | Overall |
|---|---|---|---|---|
| A | 0.538 | 0.444 | 0.407 | 0.487 |
| B | 0.577 | 0.533 | 0.296 | 0.513 |
| F | 0.500 | 0.444 | 0.333 | 0.453 |
| F-seq | 0.577 | 0.422 | 0.296 | 0.480 |

**Figure 4.1** — grouped bar chart, accuracy by system grouped by hop (notebook N2 / by-type).

**Analysis (expand into own prose; the findings and framing are fixed):**
- **Iteration (B) is the best orchestration** — rank-1 in all three models (0.513 / 0.527 / 0.347). B>A is
  significant pooled across models (paired sign test p=.019; individually significant only on Nova,
  p=.027; directional on DeepSeek/Qwen). This is consistent with IRCoT/Adaptive-RAG on MuSiQue; the
  smaller margin here is expected because A is a far stronger single-pass baseline (hybrid+rerank,
  20-chunk context, small pool) than the BM25-top-k baselines those papers improve on (§6 / analysis §6).
- **Parallel decomposition (F) does *not* beat naive (A)** — F ≤ A on 2 of 3 models (pooled slightly
  favours A). Interpret via the *dead-bridge* problem: F issues bridge sub-questions ("the spouse of
  *that director*") blind, so later-hop retrievals are weak. **This is a novel finding** — no published
  precedent in either direction — and stands in tension with Ammann et al. (2025), who found
  decomposition helps on MultiHop-RAG; the MultiHop arm (E4–E6) is the reconciling test.
- **Sequential vs parallel decomposition (F-seq vs F):** F-seq > F directionally in all three models but
  **not significantly** (pooled p=.203); its edge concentrates at 2-hop. The bridge-resolution *mechanism*
  is nonetheless validated (§4.3 / §7.9 of the analysis: F-seq assembles complete gold evidence on
  100/150 vs F's 75) — the generator fails to convert the better evidence.
- **B vs F-seq:** B leads in all three models (directional, not significant); both resolve bridges — B
  adaptively, F-seq by pre-decomposition — at differing cost (§4.5).

---

## 4.3 Study 2 — Retriever × orchestration (novel contribution) — corrected from pilot

> **⚠ Correction of an earlier pilot claim.** A preliminary n=50 study (DeepSeek only, pre-final config)
> reported that *dense-only retrieval wins on MuSiQue and B-minus (0.640) is the best system*. **This did
> not replicate at n=150 and is refuted** (below). The reversal is retained as a *methodological finding*
> about small-sample RAG evaluation (§4.7 / analysis §4.1), not as a result.

*Evidence: `compute-metrics`, per-query paired sign tests (analysis §3.1).*

**Table 4.3 — Retriever effect (containment accuracy), MuSiQue, DeepSeek-V3.** (Same direction on Qwen &
Nova — Table 4.1; per-cell numbers for all models in the analysis report.)

| Orchestration | hybrid+rerank | dense-only | Δ (hybrid − dense) |
|---|---|---|---|
| naive (A / A-minus) | 0.487 | 0.420 | **+0.067** |
| iterative (B / B-minus) | 0.513 | 0.447 | **+0.066** |
| parallel decomp. (F / F-minus) | 0.453 | 0.427 | +0.026 |
| sequential decomp. (F-seq / F-seq-minus) | 0.480 | 0.433 | +0.047 |

**Finding (verified): hybrid retrieval beats dense-only, consistently but modestly.** All **24** cells
(8 systems × 3 models) favour hybrid — zero reversals; hybrid also retrieves strictly more gold (higher
recall@5) in every cell. The effect is small per cell (~3–7 pts) and **no single (model, orchestration)
contrast reaches significance** at n=150, but it is **highly significant pooled** (paired sign test:
overall b=212, c=136, **p=5.5×10⁻⁵**; per-model DeepSeek p=.006, Nova p=.030, Qwen p=.059 marginal).

**Required phrasing:** report this as *directionally universal and significant when pooled* — never claim
per-cell significance.

**Figure 4.5** — clustered bars: hybrid vs dense-only accuracy per orchestration (DeepSeek), MuSiQue.

**Analysis (expand):**
- *Consistency with the literature:* BEIR (Thakur et al. 2021) establishes that no retriever dominates
  across datasets; a hybrid/rerank pipeline outperforming dense-only is unremarkable *in general*.
- *The MuSiQue nuance:* MuSiQue's hard distractors are **BM25-mined with intermediate answers masked**
  (Trivedi et al. 2022) — adversarial to lexical retrieval *by construction*. Hybrid winning **despite**
  this implies the cross-encoder reranker filters the lexical noise the BM25 arm surfaces. **State this as
  the study's interpretation** — no direct published evidence exists for reranker robustness to
  BM25-adversarial distractors (analysis §6.3).
- *Dataset dependence* — the headline of Study 2 — **cannot be concluded until MultiHop (E4–E6) runs.**
  The pilot's cross-dataset contrast (hybrid +0.20 on news vs flat on MuSiQue) is exactly the claim the
  final MultiHop arm must confirm or revise. `[E4–E6 pending]`

---

## 4.4 Answer-quality metrics and the metric audit — A4 / O6

*Evidence: `compute-metrics` (`avg_token_f1`, `accuracy_exact`); N4 agreement matrix. CRAG judge omitted
by design (disclose).*

**Table 4.4 — Answer-quality metrics, MuSiQue, DeepSeek-V3.**

| System | Containment (primary) | Exact match | Token F1 |
|---|---|---|---|
| A | 0.487 | 0.000 | 0.444 |
| B | 0.513 | 0.000 | 0.477 |
| F | 0.453 | 0.000 | 0.416 |
| F-seq | 0.480 | 0.000 | 0.463 |

**Analysis (expand):** the secondaries track the primary ordering (B ≥ F-seq > A > F on token-F1), so no
finding rests on one metric. **Exact-match is 0.000 across the board** — MuSiQue gold answers are rarely
reproduced as an exact normalised string, which is precisely why containment (and token-F1) are the
appropriate metrics and why a strict EM would be uninformative here; report this as a metric-choice
justification, not a null result. Containment is stricter than the official MultiHop-RAG word-intersection
scorer (analysis §6), so reported accuracy is conservative. *(Full metric-agreement heatmap: N4.)*

---

## 4.5 Cost and latency — RQ2

*Evidence: N2 `variance_tbl`; N5 `fig_pareto`. `cost_usd` = billed Bedrock LLM cost only; the Cohere
reranker is a separately-metered per-retrieval charge (hybrid systems only), reported separately.*

**Table 4.5 — Cost-per-correct, MuSiQue (selected cells; full 24-cell grid in analysis §5).**

| Cell | Accuracy | $/correct | Pareto? |
|---|---|---|---|
| Nova-A | 0.273 | $0.00083 | ✓ (cheapest) |
| Qwen-A | 0.473 | $0.00128 | ✓ |
| Qwen-F-seq | 0.480 | $0.00230 | ✓ |
| **Qwen-B** | **0.527** | $0.00416 | ✓ (max accuracy) |
| DeepSeek-A | 0.487 | $0.00425 | dominated |
| DeepSeek-B | 0.513 | $0.01710 | dominated |

**Figure 4.4** — Pareto frontier: accuracy vs cost-per-correct across all 24 cells (N5). **Headline cost figure.**

**Finding (verified): the most expensive model is never the rational choice.** The Pareto frontier is
**Nova-A → Nova-F-seq → Qwen-A → Qwen-F-seq → Qwen-B** — **every DeepSeek cell is dominated.** Qwen-B beats
DeepSeek-B on accuracy (0.527 vs 0.513) at **24% of the cost** ($0.0042 vs $0.0171 per correct answer).
Within every model, B costs ~4× A per correct answer — iteration buys accuracy but not cheaply. This is
the Gap-2 contribution: cost-per-correct is unreported in the surveyed literature (analysis §6).

*Latency: report p50/p95 per system (B and F-seq highest, ~7–12 s; A/-minus ~1–3 s). Cross-model latency
carries a serving-infra confound — compare within a model (analysis §3.5).*

---

## 4.6 Cross-model rank stability and orchestration robustness — RQ3 / RQ4

*Evidence: N1 `kendall_tau_b`; per-system accuracy (exp 50/51/53).*

**Table 4.6 — System ranking by accuracy, per model (MuSiQue).**

| Rank | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| 1 | B | B | B |
| 2 | A | F-seq | F-seq |
| 3 | F-seq | A | B-minus |
| 4 | F | B-minus / F-seq-minus (tie) | F |
| … | (A-minus last) | (A-minus / F-minus last) | (A-minus / F-minus / F-seq-minus tie last) |

**Kendall τ-b across models:** DeepSeek↔Qwen **0.741**, DeepSeek↔Nova **0.643**, Qwen↔Nova **0.706** —
all three pairwise correlations significant.

**Findings (verified):**
- **Rankings are stable across the capability gradient (RQ3/RQ4):** B is rank-1 in all three models;
  F-seq is the best decomposition variant in all three; the four `-minus` twins cluster at the bottom.
  τ-b 0.64–0.74 (all significant) → the orchestration ranking generalises across cost-efficient models —
  a deployable conclusion, and evidence against single-model RAG benchmarking.
- **Orchestration robustness (a distinct RQ4 result): decomposition presupposes reliable structured
  output.** Nova Lite's decomposer parse-fails (~85% of queries), collapsing F/F-seq to naive retrieval;
  iterative free-text reformulation (B) still works on Nova (rank-1). **The model-robust multi-hop
  strategy is iteration, not decomposition** — connecting to the System-B routing redesign (§3.3) that
  fixed the same JSON-fragility for B.
- With n=150 the single-digit-query noise of the pilots is materially reduced; report bootstrap CIs and
  state where cells overlap (analysis §3).

---

## 4.7 Summary of findings

Per research question (MuSiQue arm; MultiHop completes RQ1/RQ3 and the Study-2 dataset contrast):

- **RQ1 (accuracy by orchestration + hop):** Iteration (B) is best in all models; sequential
  decomposition (F-seq) is the best decomposition variant but only directionally > parallel (F); parallel
  decomposition does **not** beat naive — a novel result.
- **RQ2 (cost per correct):** Qwen3-32B + iteration is the rational configuration; every frontier-cost
  (DeepSeek) cell is Pareto-dominated.
- **RQ3 (ranking across models):** Stable — τ-b 0.64–0.74, B rank-1 throughout. `[MultiHop confirms]`
- **RQ4 (predictability / robustness):** Rankings generalise; and decomposition orchestration is
  model-fragile (Nova collapse) while iteration is robust.
- **Novel — retriever × orchestration:** Hybrid > dense-only, consistently but modestly (pooled
  p=5.5×10⁻⁵), on MuSiQue. The **dataset-dependence** headline awaits MultiHop `[E4–E6 pending]`.
- **Behavioural (analysis §7):** the agent's self-termination is a valid confidence signal (early-stop
  20–61 pts more accurate than budget-forced), yielding a zero-cost abstention policy; generation, not
  retrieval, is the dominant error source (93–99% coverage; 87–99% of errors have gold in context), and
  accuracy degrades as needed evidence sits deeper in the 20-chunk context.
- **Methodological:** the n=50→n=150 reversal of the retriever finding is quantified evidence that
  small-sample RAG comparisons mislead (feeds RQ4; §4.3 correction).

> These feed Chapter 5's per-RQ conclusions (Conclusions /20). Chapter 5 and the MultiHop-dependent rows
> require E4–E6.

---

## Figure/table → source index

| Artefact | Source |
|---|---|
| T4.1/T4.2 accuracy, F4.1 bars | `compute-metrics`, `metrics-by-type` (exp 50/51/53), N2 |
| T4.3 / F4.5 retriever 4×2 | `compute-metrics`, analysis §3.1 |
| T4.4 quality, F4.3 agreement | `compute-metrics`, N4 |
| T4.5 cost, F4.4 Pareto | N2 `variance_tbl`, N5 `fig_pareto`, analysis §5 |
| T4.6 rank stability | N1 `kendall_tau_b`, analysis §3.3 |
| Behavioural sub-studies | analysis §7 (termination, abstention §7.8, position §7.9) |
