# Chapter 4 — Results

> **Status: complete prose draft for your review — not a hand-in.** All MuSiQue numbers are the
> verified final-matrix values (n=150 per cell, 3,600 runs, zero failures; every figure spot-checked
> against the raw data). The MultiHop-RAG arm has not yet been run: §4.8 is a stub, and the two
> sentences flagged `[MultiHop]` must be completed or the scope narrowed before submission. Before
> submitting: rewrite in your own voice; move the statistical machinery references to the appendix;
> renumber tables/figures per institutional style.

---

## 4.1 The executed evaluation

The evaluation reported here executed the full 4×2 factorial of Chapter 3 — four orchestration
strategies, each with its dense-only twin — under three language models on the MuSiQue benchmark:
3,600 runs in total (8 systems × 150 questions × 3 models), with no failed runs. All cells share the
frozen configuration described in §3.8: one commit, one index build, one seeded stratified sample
(78 two-hop, 45 three-hop, 27 four-hop questions; seed 42), identical query identifiers in every
cell, the Cohere reranker, and a uniform twenty-passage answer budget. Total generation cost for the
matrix was approximately $7. An independent integrity audit confirmed sample identity across all
cells, configuration constancy, completeness, and the absence of scoring anomalies; the audit and the
full statistical tables are reproduced in Appendix `[X]`. `[MultiHop: the corresponding paragraph for
the news-corpus arm is added when E4–E6 complete.]`

Because every system answered the same questions, all comparisons in this chapter are paired, and
significance statements use exact paired sign tests unless stated otherwise. Two phrasing conventions
are applied throughout, following the statistical results rather than preceding them: effects that
are consistent in direction but individually underpowered are described as *directional*, and pooled
tests are reported as pooled, never as per-cell findings.

## 4.2 Accuracy by orchestration strategy (RQ1)

Table 4.1 reports containment accuracy for all eight systems under each model.

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A (single-pass) | 0.487 | 0.473 | 0.273 |
| A-minus | 0.420 | 0.420 | 0.260 |
| B (iterative) | 0.513 | 0.527 | 0.347 |
| B-minus | 0.447 | 0.467 | 0.307 |
| F (parallel decomp.) | 0.453 | 0.440 | 0.300 † |
| F-minus | 0.427 | 0.420 | 0.260 † |
| F-seq (sequential decomp.) | 0.480 | 0.480 | 0.320 † |
| F-seq-minus | 0.433 | 0.467 | 0.260 † |

*Table 4.1 — containment accuracy, MuSiQue, n=150 per cell. † Nova Lite's decomposition systems
degraded to single-retrieval behaviour (§4.6) and approximate its System A rather than genuine
decomposition.*

Three results emerge.

Iterative retrieval is the strongest orchestration under every model. B ranks first in all three
columns, and the improvement over single-pass A is significant when pooled across models (56
discordant pairs favouring B against 33 favouring A; p = .019), though within individual models it
reaches significance only for Nova Lite (p = .027) and is directional for DeepSeek-V3 and Qwen3-32B.
The direction is consistent with the iterative literature reviewed in Chapter 2; the margin — two to
seven points — is considerably smaller than the gains reported there, and §4.5 and §4.7 argue this is
a property of the baseline: against a single-pass system equipped with hybrid retrieval, reranking
and a twenty-passage context, there is simply less for iteration to repair.

Parallel decomposition does not improve on single-pass retrieval. F trails A under DeepSeek-V3 and
Qwen3-32B and the pooled comparison marginally favours A (23 pairs to 29). The mechanism is visible
in the design: F must phrase later-hop sub-questions before earlier hops are resolved, so a
sub-question such as "the spouse of *that director*" goes to the retriever with its key entity
unresolved. No published study was found reporting this negative result — or its converse — under
matched conditions, and it stands in tension with the decomposition gains of Ammann, Golde and Akbik
(2025) on MultiHop-RAG, obtained on a different benchmark with a different retrieval stack.
`[MultiHop: the E4–E6 arm tests directly whether the discrepancy is dataset-driven.]`

Sequential decomposition improves on parallel decomposition in direction but not significantly.
F-seq exceeds F under all three models (51 discordant pairs to 38 pooled; p = .203) and is the
best-performing decomposition variant everywhere, but the evidence at this sample size does not
support a stronger claim. Notably, the mechanism *is* demonstrable even where the accuracy gain is
not: F-seq assembled the complete gold evidence set for 100 of 150 questions against F's 75 — the
bridge-resolution step retrieves what parallel decomposition misses — yet much of the advantage is
lost at generation (§4.7).

Accuracy declines with hop depth for every system and model. Under DeepSeek-V3, for example, System B
falls from 0.577 on two-hop questions to 0.296 on four-hop; the full per-hop tables are in Appendix
`[X]`. The four-hop stratum (n = 27) is small, and per-stratum differences there should be read as
indicative only.

## 4.3 The retrieval pipeline and its interaction with orchestration (Study 2)

Table 4.2 reports the paired effect of the retrieval pipeline — hybrid retrieval with reranking
against dense-only retrieval — for each orchestration under DeepSeek-V3; the same pattern holds under
the other two models.

| Orchestration | Hybrid | Dense-only | Difference |
|---|---|---|---|
| Single-pass (A) | 0.487 | 0.420 | +0.067 |
| Iterative (B) | 0.513 | 0.447 | +0.066 |
| Parallel decomposition (F) | 0.453 | 0.427 | +0.026 |
| Sequential decomposition (F-seq) | 0.480 | 0.433 | +0.047 |

*Table 4.2 — containment accuracy by retriever, MuSiQue, DeepSeek-V3, n=150 per cell.*

The hybrid pipeline outperformed its dense-only ablation in all twenty-four cells of the matrix —
every orchestration, under every model — and retrieved strictly more gold evidence in every cell as
well. The per-cell differences are modest, between three and seven points, and no single cell reaches
significance on its own; pooled across the matrix, however, the effect is unambiguous (212 discordant
pairs favouring hybrid against 136; p = 5.5 × 10⁻⁵). Pooled within models, the effect is significant
for DeepSeek-V3 (p = .006) and Nova Lite (p = .030) and marginal for Qwen3-32B (p = .059). The
appropriate summary is that the pipeline's benefit is directionally universal and statistically
secure in aggregate, while being too small to certify in any single configuration — a distinction
this chapter maintains deliberately.

The result carries an interpretive subtlety specific to MuSiQue. As described in §3.6, MuSiQue's
distractor paragraphs are mined with BM25, making them adversarial to lexical retrieval by
construction; one might therefore have expected the lexical arm of the hybrid pipeline to be a
liability here. That the full pipeline nonetheless wins every cell suggests the cross-encoder
reranker filters the lexical noise that the BM25 arm admits. No published evaluation of reranker
robustness to BM25-mined distractors was found, so this explanation is offered as this study's
interpretation rather than as an established mechanism.

These findings correct an earlier result from this project's own pilot phase, and the correction is
reported deliberately. A fifty-question pilot on DeepSeek-V3, run before the final configuration was
frozen, had shown the opposite pattern — dense-only retrieval apparently outperforming hybrid, with
the dense-only iterative system the best of all systems tested. At n = 150 under paired testing the
reversal disappears in every cell. The pilot conclusion is attributable to sampling noise compounded
by configuration drift between pilot and final runs, and its failure to replicate is itself one of
the study's findings on evaluation reliability, taken up under RQ4 in §4.6 and in Chapter 5.

## 4.4 Secondary metrics and the metric audit

Token-level F1 tracks the containment ordering throughout (DeepSeek-V3: A 0.444, B 0.477, F 0.416,
F-seq 0.463), so no ranking claim in this chapter depends on the choice between the primary metric
and its lexical secondary. Exact match is zero across all systems: MuSiQue gold answers are almost
never reproduced verbatim by instruction-tuned generators, which produce sentences rather than spans.
This is precisely the failure of strict exact match as an instrument for generative systems that
motivated the containment convention (§3.7), and it is reported here as evidence for that metric
choice rather than as a finding about the systems. The full metric-agreement analysis appears in
Appendix `[X]`.

## 4.5 Cost (RQ2)

Table 4.3 reports cost per correct answer — total billed generation cost divided by correct answers —
for selected configurations; the full twenty-four-cell table is in Appendix `[X]`.

| Configuration | Accuracy | Cost per correct answer |
|---|---|---|
| Nova Lite — A | 0.273 | $0.0008 |
| Qwen3-32B — A | 0.473 | $0.0013 |
| Qwen3-32B — F-seq | 0.480 | $0.0023 |
| Qwen3-32B — B | 0.527 | $0.0042 |
| DeepSeek-V3 — A | 0.487 | $0.0042 |
| DeepSeek-V3 — B | 0.513 | $0.0171 |

*Table 4.3 — cost-effectiveness of selected configurations, MuSiQue. Costs are billed generation
charges; the reranker is metered separately and identically for all hybrid systems.*

The accuracy–cost frontier is occupied entirely by the two cheaper models: in order of increasing
cost and accuracy, Nova-A, Nova-F-seq, Qwen-A, Qwen-F-seq, Qwen-B. Every DeepSeek-V3 configuration is
dominated — most strikingly, Qwen3-32B's iterative system exceeds DeepSeek-V3's on accuracy (0.527
against 0.513) at roughly a quarter of the cost per correct answer. Within each model, iteration
costs roughly four times as much per correct answer as single-pass retrieval; whether that premium is
worthwhile depends on how the deployment values its two-to-seven accuracy points. Sequential
decomposition occupies a consistent middle ground and appears on the frontier under both cheaper
models. The general conclusion for RQ2 is that model choice dominates orchestration choice in the
economics: selecting the mid-tier model with iteration outperforms selecting the strongest model at
any orchestration, on both axes at once.

Latency follows call structure: the single-pass systems answer in one to three seconds at the 95th
percentile, the iterative and sequential systems in seven to twelve. Cross-model latency comparisons
are confounded by serving infrastructure and are not made.

## 4.6 Rank stability and orchestration robustness (RQ3, RQ4)

Ranking the eight systems by accuracy within each model and correlating the rankings across models
yields Kendall τ-b of 0.741 (DeepSeek–Qwen), 0.643 (DeepSeek–Nova) and 0.706 (Qwen–Nova), each
significant at the 5% level. The ordering is stable in its essentials: the iterative system ranks
first under every model, sequential decomposition is the best decomposition variant under every
model, and the four dense-only twins occupy the bottom of the ranking almost everywhere. For RQ3,
the practical reading is that an orchestration choice made on one cost-efficient model carries to its
neighbours; single-model evaluations, ubiquitous in the literature, would not have revealed whether
this was so. `[MultiHop: cross-dataset stability is added when E4–E6 complete.]`

The robustness half of RQ4 is answered by a failure the design converted into a measurement. Nova
Lite could not execute the decomposition systems: its structured decomposition output failed to parse
on roughly 85% of questions, and F and F-seq consequently collapsed to single-retrieval behaviour
(mean retrieval counts near 1.4, against 3.5–3.7 for the same systems under the larger models). The
iterative system, whose free-text routing assumes nothing about structured output, ran at full
capability on the same model and ranked first. The deployable lesson is that decomposition
orchestration presupposes reliable structured generation — an assumption that quietly fails below a
capability threshold — whereas iterative reformulation degrades gracefully; for small-model
deployments, iteration is the robust choice.

The reliability half of RQ4 is answered by the pilot reversal reported in §4.3: conclusions drawn at
n = 50 inverted at n = 150 under paired testing. Single-run, small-sample evaluation — the field's
prevailing practice, as Chapter 2 documents — was demonstrably capable of producing a confident,
wrong conclusion within this very study.

## 4.7 The behaviour of the agentic systems

The persisted trajectories permit an analysis of how the multi-step systems behave, beyond what they
score. Four observations follow; the supporting tables are in Appendix `[X]`.

**Self-termination is a confidence signal.** The iterative system stops early when its routing step
elects to answer, and is otherwise forced to answer at the five-step budget. Early-stopped answers
are dramatically more accurate than budget-forced ones in every cell: under DeepSeek-V3, 0.839
against 0.429; under Qwen3-32B, 0.706 against 0.292; under Nova Lite, 0.577 against 0.224. Reaching
the budget is a signal that the question is hard, not that the search was thorough.

**The signal supports a free abstention policy.** Re-scoring the stored runs under a policy that
answers only when the agent self-terminates raises the precision of delivered answers by seventeen
to fifty-one points, at a coverage cost that varies sharply by model. Qwen3-32B offers the practical
operating point: it self-terminates on 57% of questions, retaining 76% of its correct answers while
filtering out roughly two-thirds of its errors. The policy costs nothing — the signal is emitted
anyway. The converse policy, redirecting budget-forced questions to the cheap single-pass system,
performs *worse* than simply letting the agent answer: the single-pass system does even worse on
exactly those questions. Budget exhaustion identifies questions that are intrinsically hard for
every strategy, not questions the agent in particular has failed.

**Errors are overwhelmingly generation-side.** With the twenty-passage budget, at least one gold
passage was present in the answering context for 93–99% of questions in every system, and between
87% and 99% of all errors occurred with gold evidence already in context. No system answered
correctly without gold evidence present — there were no lucky parametric guesses in 1,200 audited
runs. On this benchmark, at this context size, retrieval is close to solved and generation is the
binding constraint — which explains at once why orchestration gains are modest (§4.2) and why F-seq's
superior evidence assembly fails to convert fully into accuracy.

**Where generation fails is positional.** Restricting attention to runs where the complete gold
evidence set was in context — retrieval fully controlled — accuracy falls from 0.727 when the deepest
needed passage sits in the top five context positions to 0.469 when it sits in the bottom five, a
twenty-six point decline that persists within a single hop stratum. The pattern is consistent with
the position-sensitivity literature (Liu et al., 2023), with one caveat stated plainly: context
position here is assigned by the reranker rather than randomised, so position and question difficulty
are partially confounded, and the finding is correlational. A randomised-position replication is
identified as future work.

## 4.8 The MultiHop-RAG arm

`[Pending — E4–E6. This section will report the same tables and contrasts on the news benchmark,
completing Study 2's dataset comparison and testing whether the decomposition result of §4.2 and the
retriever result of §4.3 transfer to a lexically-conventional corpus. If the scope is narrowed to
MuSiQue-only instead, delete this section and adjust §§1.5, 3.6 and 5 accordingly.]`

## 4.9 Summary

On MuSiQue, under three cost-efficient models: iterative retrieval is the strongest and most robust
orchestration, significantly better than single-pass when pooled; parallel decomposition does not
improve on a strong single-pass baseline, a negative result without published precedent; sequential
decomposition improves on parallel directionally and demonstrably assembles better evidence, but the
gain is largely lost at generation. The hybrid retrieval pipeline beats its dense-only ablation in
all twenty-four cells — modestly per cell, decisively in aggregate — despite the benchmark's
lexically-adversarial construction. System rankings are stable across models; the accuracy–cost
frontier is owned by the mid-tier model, and the strongest model is never the economical choice. The
agent's self-termination decision is a usable confidence signal enabling a zero-cost abstention
policy; residual errors are dominated by generation failing over evidence it already holds,
concentrated where that evidence sits deep in the context. A pilot-scale version of this evaluation
reached the opposite conclusion about retrievers before reversing at full sample size — a caution
about small-sample evaluation that the field's prevailing practice does not currently heed.
