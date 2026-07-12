# Chapter 4 — Results and Discussion

> **Status: complete prose draft for your review — not a hand-in. BOTH arms now populated.** MuSiQue
> numbers are the verified final-matrix values (n=150 per cell, 3,600 runs, zero failures); MultiHop-RAG
> numbers (§4.8) are the completed E4–E6 arm (n=200 per cell, 4,800 runs, one failure, 0.02%). Before
> submitting: rewrite in your own voice; move the statistical machinery references to the appendix;
> renumber tables/figures per institutional style.

---

## 4.1 The executed evaluation

The evaluation reported here executed the full 4×2 factorial of Chapter 3 — four orchestration
strategies, each with its dense-only twin — under three language models on the MuSiQue benchmark:
3,600 runs in total (8 systems × 150 questions × 3 models), with no failed runs. All cells share the
frozen configuration described in §3.9: one commit, one index build, one seeded stratified sample
(78 two-hop, 45 three-hop, 27 four-hop questions; seed 42), identical query identifiers in every
cell, the Cohere reranker, and a uniform twenty-passage answer budget. The MultiHop-RAG arm executed
the same factorial over 200 seeded stratified questions (64 inference, 67 comparison, 45 temporal,
24 null; seed 42): 4,800 further runs, of which one failed (0.02% of that arm; 0.01% of the full
matrix) and is scored as incorrect under
the declared policy. Total generation cost for the complete 8,400-run matrix was approximately $25.
An independent integrity audit confirmed sample identity across all cells, configuration constancy,
completeness, and the absence of scoring anomalies; the audit and the full statistical tables are
reproduced in Appendices D and E.

All comparisons in this chapter follow the statistical plan of §3.8: paired exact sign tests,
seeded bootstrap confidence intervals on every accuracy cell, and Holm–Bonferroni correction within
the two pre-specified contrast families, with the directional / nominal / corrected phrasing
conventions defined there. Applying the correction to the executed matrix frames the chapter at the
outset: of the six pooled orchestration contrasts, **none** remains significant at the 5% level —
the strongest (parallel decomposition over single-pass on MultiHop-RAG, p = .011) narrowly misses
its corrected threshold of .0083 — whereas the four pooled retriever contrasts **all survive** with
room to spare (p ≤ 5.5 × 10⁻⁵ against thresholds of .0125–.05). Orchestration conclusions in this
chapter therefore rest on the *consistency* of direction across models and the *coherence* of the
cross-dataset pattern, and are labelled nominally significant where p < .05 uncorrected; the
retriever conclusions rest on corrected significance outright.

## 4.2 Accuracy by orchestration strategy (RQ1)

Table 4.1 reports containment accuracy for all eight systems under each model.

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A (single-pass) | 0.487 [.41–.57] | 0.473 [.39–.55] | 0.273 [.20–.35] |
| A-minus | 0.420 [.35–.50] | 0.420 [.34–.50] | 0.260 [.19–.33] |
| B (iterative) | 0.513 [.43–.59] | 0.527 [.45–.61] | 0.347 [.27–.43] |
| B-minus | 0.447 [.37–.53] | 0.467 [.39–.55] | 0.307 [.23–.38] |
| F (parallel decomp.) | 0.453 [.37–.53] | 0.440 [.36–.52] | 0.300 [.23–.37] † |
| F-minus | 0.427 [.35–.51] | 0.420 [.34–.50] | 0.260 [.19–.33] † |
| F-seq (sequential decomp.) | 0.480 [.40–.56] | 0.480 [.40–.56] | 0.320 [.25–.39] † |
| F-seq-minus | 0.433 [.35–.51] | 0.467 [.39–.55] | 0.260 [.19–.33] † |

*Table 4.1 — containment accuracy with bootstrap 95% CIs, MuSiQue, n=150 per cell. † Nova Lite's
decomposition systems degraded to single-retrieval behaviour (§4.6) and approximate its System A
rather than genuine decomposition. Note that the per-cell intervals overlap heavily across
orchestrations within a model — the paired analyses below, not the marginal intervals, carry the
comparative claims.*

*Figure 4.2 — the same cells graphically, grouped by system with CI whiskers.
[File: `thesis/figures/figure_4_2_musique_accuracy.png`.]*

Three results emerge.

Iterative retrieval is the strongest orchestration under every model. B ranks first in all three
columns, and the improvement over single-pass A is nominally significant when pooled across models
(56 discordant pairs favouring B against 33 favouring A; p = .019), though within individual models
it reaches nominal significance only for Nova Lite (p = .027), is directional for DeepSeek-V3 and
Qwen3-32B, and — like every orchestration contrast in this study — does not survive the Holm
correction described in §4.1. The claim that iteration leads on this dataset therefore rests on its
rank-1 position under all three models rather than on any corrected test.
The direction is consistent with the iterative literature reviewed in Chapter 2; the margin — two to
seven points — is considerably smaller than the gains reported there, and §4.5 and §4.7 argue this is
a property of the baseline: against a single-pass system equipped with hybrid retrieval, reranking
and a twenty-passage context, there is simply less for iteration to repair.

Parallel decomposition does not improve on single-pass retrieval. F trails A under DeepSeek-V3 and
Qwen3-32B and the pooled comparison marginally favours A (23 pairs to 29). The mechanism is visible
in the design: F must phrase later-hop sub-questions before earlier hops are resolved, so a
sub-question such as "the spouse of *that director*" goes to the retriever with its key entity
unresolved. No published study was found in the literature survey conducted for this study (Appendix
A) reporting this negative result — or its converse — under
matched conditions, and it stands in tension with the decomposition gains of Ammann, Golde and Akbik
(2025) on MultiHop-RAG, obtained on a different benchmark with a different retrieval stack. The
MultiHop-RAG arm of this study (§4.8) resolves the discrepancy directly: on that benchmark, F is the
*best* system under both capable models — the negative result is a property of MuSiQue's sequentially
dependent hops, not of parallel decomposition in general.

Sequential decomposition improves on parallel decomposition in direction but not significantly.
F-seq exceeds F under all three models (51 discordant pairs to 38 pooled; p = .203) and is the
best-performing decomposition variant everywhere, but the evidence at this sample size does not
support a stronger claim. Notably, the mechanism *is* demonstrable even where the accuracy gain is
not: F-seq assembled the complete gold evidence set for 100 of 150 questions against F's 75 — the
bridge-resolution step retrieves what parallel decomposition misses — yet much of the advantage is
lost at generation (§4.7).

Accuracy declines with hop depth for every system and model. Under DeepSeek-V3, for example, System B
falls from 0.577 on two-hop questions to 0.296 on four-hop; the full per-hop tables are in Appendix
C. The four-hop stratum (n = 27) is small, and per-stratum differences there should be read as
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
pairs favouring hybrid against 136; p = 5.5 × 10⁻⁵ — a value that survives the Holm correction over
the retriever-contrast family with an order of magnitude to spare). Pooled within models, the effect
is significant for DeepSeek-V3 (p = .006) and Nova Lite (p = .030) and marginal for Qwen3-32B
(p = .059). The appropriate summary is that the pipeline's benefit is directionally universal and
multiplicity-robust in aggregate, while being too small to certify in any single configuration — a
distinction this chapter maintains deliberately.

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
choice rather than as a finding about the systems. The recorded secondary-metric values for both
arms are collected in Appendix C.

## 4.5 Cost (RQ2)

Table 4.3 reports cost per correct answer — total billed generation cost divided by correct answers —
for selected configurations; the full twenty-four-cell table is in Appendix C.

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

*Figure 4.1 — accuracy against cost per correct answer (log scale) for the hybrid systems under all
three models, both datasets, with the Pareto frontier drawn.
[File: `thesis/figures/figure_4_1_pareto.png`.]*

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
this was so. The MultiHop arm sharpens the answer in an important way: rankings are stable across
models *within* each dataset, but they do **not** transfer *across* datasets — the systems that rank
first on the two benchmarks are different (§4.8), so the decision-relevant variable is the benchmark's
question structure, not the model.

The robustness half of RQ4 is answered by a failure the design converted into a measurement. Nova
Lite could not execute the decomposition systems: its structured decomposition output failed to parse
on roughly 85% of questions, and F and F-seq consequently collapsed to single-retrieval behaviour
(mean retrieval counts near 1.4, against 3.5–3.7 for the same systems under the larger models). The
iterative system, whose free-text routing assumes nothing about structured output, ran at full
capability on the same model and ranked first. The deployable lesson is that decomposition
orchestration presupposes reliable structured generation — an assumption that quietly fails below a
capability threshold — whereas iterative reformulation, which assumes nothing about output structure, keeps functioning; for small-model
deployments, iteration is the robust choice.

The reliability half of RQ4 is answered by the pilot reversal reported in §4.3: conclusions drawn at
n = 50 inverted at n = 150 under paired testing. Single-run, small-sample evaluation — the field's
prevailing practice, as Chapter 2 documents — was demonstrably capable of producing a confident,
wrong conclusion within this very study.

## 4.7 The behaviour of the agentic systems

The persisted trajectories permit an analysis of how the multi-step systems behave, beyond what they
score. Four observations follow; the supporting tables are in Appendices C and D.

Self-termination proves to be a confidence signal. The iterative system stops early when its routing step
elects to answer, and is otherwise forced to answer at the five-step budget. Early-stopped answers
are dramatically more accurate than budget-forced ones in every cell: under DeepSeek-V3, 0.839
against 0.429; under Qwen3-32B, 0.706 against 0.292; under Nova Lite, 0.577 against 0.224. Reaching
the budget is a signal that the question is hard, not that the search was thorough.

The same signal supports an abstention policy that costs nothing. Re-scoring the stored runs under a policy that
answers only when the agent self-terminates raises the precision of delivered answers by seventeen
to fifty-one points, at a coverage cost that varies sharply by model. Qwen3-32B offers the practical
operating point: it self-terminates on 57% of questions, retaining 76% of its correct answers while
filtering out roughly two-thirds of its errors. The policy costs nothing — the signal is emitted
anyway. The converse policy, redirecting budget-forced questions to the cheap single-pass system,
performs *worse* than simply letting the agent answer: the single-pass system does even worse on
exactly those questions. Budget exhaustion identifies questions that are intrinsically hard for
every strategy, not questions the agent in particular has failed.

The errors that remain are overwhelmingly generation-side. With the twenty-passage budget, at least one gold
passage was present in the answering context for 93–99% of questions in every system, and between
87% and 99% of all errors occurred with gold evidence already in context. No system answered
correctly without gold evidence present — there were no lucky parametric guesses in 1,200 audited
runs. On this benchmark, at this context size, retrieval is close to solved and generation is the
binding constraint — which explains at once why orchestration gains are modest (§4.2) and why F-seq's
superior evidence assembly fails to convert fully into accuracy.

Where generation fails turns out to be positional. Restricting attention to runs where the complete gold
evidence set was in context — retrieval fully controlled — accuracy falls from 0.727 when the deepest
needed passage sits in the top five context positions to 0.469 when it sits in the bottom five, a
twenty-six point decline that persists within a single hop stratum. The pattern is consistent with
the position-sensitivity literature (Liu et al., 2024), with one caveat stated plainly: context
position here is assigned by the reranker rather than randomised, so position and question difficulty
are partially confounded, and the finding is correlational. A randomised-position replication is
identified as future work.

## 4.8 The MultiHop-RAG arm and the cross-dataset comparison

Table 4.4 reports containment accuracy for the full factorial on MultiHop-RAG.

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A | 0.830 [.78–.88] | 0.820 [.77–.87] | 0.785 [.73–.84] |
| A-minus | 0.670 [.61–.74] | 0.685 [.62–.75] | 0.675 [.61–.74] |
| B | 0.825 [.77–.88] | 0.845 [.80–.90] | 0.755 [.70–.81] |
| B-minus | 0.760 [.70–.82] | 0.775 [.72–.83] | 0.760 [.70–.82] |
| F | **0.855** [.81–.90] | **0.875** [.83–.92] | 0.770 [.71–.83] † |
| F-minus | 0.785 [.73–.84] | 0.790 [.74–.85] | 0.695 [.63–.76] † |
| F-seq | 0.845 [.79–.90] | 0.845 [.80–.90] | 0.775 [.72–.83] † |
| F-seq-minus | 0.715 [.65–.78] | 0.725 [.67–.79] | 0.710 [.65–.77] † |

*Table 4.4 — containment accuracy with bootstrap 95% CIs, MultiHop-RAG, n=200 per cell. † Nova Lite's
decomposition systems degraded to near-single-retrieval behaviour again (mean retrieval counts
1.45–1.57 against 3.4–3.7 under the larger models), replicating the robustness finding of §4.6 on a
second dataset. The A/A-minus intervals are disjoint under every model — the retriever effect on this
dataset is visible even in the marginal intervals, unlike any orchestration contrast.*

*Figure 4.3 — the same cells graphically, grouped by system with CI whiskers.
[File: `thesis/figures/figure_4_3_multihop_accuracy.png`.]*

Absolute accuracy is far higher than on MuSiQue — 0.83 rather than 0.49 for the single-pass baseline —
consistent with MultiHop-RAG's less adversarial construction. Three results carry the chapter's
argument to its conclusion.

The first result is that the orchestration ranking inverts. Parallel decomposition, which failed to improve on single-pass
retrieval on MuSiQue, is the *best* system on MultiHop-RAG under both capable models: nominally
significantly better than A pooled across them (26 discordant pairs to 10; p = .011, nominally
significant under Qwen3-32B alone at p = .019), and better than the iterative system at the
conventional boundary (33 pairs to 18; p = .049) — at roughly a third of B's cost per correct answer.
As §4.1 disclosed, neither contrast survives the six-test Holm correction (the F-over-A test misses
its .0083 threshold narrowly); what cannot be attributed to multiplicity is the *pattern* — F ahead
of A and of B under every model on this dataset, the exact mirror of its position on MuSiQue. Iteration, first-ranked everywhere on MuSiQue, does
not beat single-pass retrieval here at all (28 pairs to 31; p = .80). Sequential decomposition's
directional advantage over parallel also reverses (24:31 in F's favour). The mechanism proposed in
§4.2 explains both halves at once: parallel decomposition's weakness is that later-hop sub-questions
are phrased before earlier hops resolve, which is fatal when hops depend on one another (MuSiQue's
construction) and irrelevant when a question's evidence requirements are largely independent
(MultiHop-RAG's inference and comparison questions), where the extra sub-question retrievals are pure
coverage gain. This resolves the tension with Ammann, Golde and Akbik (2025) noted in §4.2: their
decomposition gains on MultiHop-RAG replicate in direction here, and the contradiction with the
MuSiQue result is dataset structure, not method.

The second is that the retriever effect is much larger on news, and no longer needs pooling. The hybrid pipeline
beats its dense-only twin by up to sixteen points on MultiHop-RAG (A vs A-minus: +0.160, +0.135,
+0.110 across the three models), and the paired effect is individually significant within every model
(DeepSeek 115:30, p = 6×10⁻¹³; Qwen 129:47, p = 5×10⁻¹⁰; Nova 99:49, p = 5×10⁻⁵) — where MuSiQue
required pooling the entire matrix to secure a three-to-seven point effect. The by-type breakdown
localises the mechanism precisely: the advantage concentrates in **comparison questions**, which name
their sources ("the TechCrunch article on…"), where the single-pass system scores 0.791 with the
hybrid retriever against 0.373 without it under DeepSeek-V3 — a forty-two point gap — while
inference-type questions show essentially no retriever effect at all (0.938 against 0.938); Figure
4.4 shows the contrast. Lexical
retrieval earns its keep exactly where queries contain lexical anchors, and contributes little where
they do not. Null questions, finally, show no over-answering: every system scores 0.83–0.96 on the
unanswerable set, and the iterative system is slightly *better* than single-pass there.

*Figure 4.4 — the retriever effect localised by question type (single-pass system, DeepSeek-V3):
no effect on inference questions, a forty-two point gap on source-naming comparison questions.
[File: `thesis/figures/figure_4_4_retriever_by_type.png`.]*

The third is economic. The MultiHop-RAG accuracy–cost frontier collapses to two points: Nova-A
(0.785 at $0.0006 per correct answer) and Qwen-F (0.875 at $0.0015). Qwen3-32B with parallel
decomposition dominates every other configuration on both axes simultaneously — including every
DeepSeek-V3 cell (DeepSeek-F reaches 0.855 at 3.8 times Qwen-F's cost) and Qwen's own iterative system
(2.6 times the cost for less accuracy). Taken together with §4.5, the mid-tier model owns the frontier
on both datasets; what changes between datasets is which orchestration accompanies it.

## 4.9 Summary

The completed matrix — eight systems, three models, two datasets, 8,400 runs — supports one central
conclusion with several parts: which orchestration wins is a property of the dataset, not of the
method. On MuSiQue, whose hops depend sequentially on one another, iterative retrieval ranks first
under every model and parallel decomposition fails to beat single-pass retrieval; on MultiHop-RAG,
whose evidence requirements are largely independent, the ranking inverts — parallel decomposition is
the best and cheapest multi-query strategy, nominally significantly ahead of both single-pass and
iterative retrieval, while iteration no longer pays for itself. The two halves of the study differ in
statistical weight and are reported accordingly: the orchestration contrasts are nominally significant
and directionally unanimous but do not survive multiplicity correction, whereas the retriever result
is multiplicity-robust outright — the hybrid pipeline beats its dense-only ablation everywhere, with
an effect an order of magnitude larger on news (up to sixteen points, Holm-corrected significant in
every model, concentrated almost entirely in lexically-anchored comparison questions) than on the
lexically-adversarial MuSiQue (three to seven points, significant only pooled). Within a dataset, rankings are stable across the
cost-efficient model gradient; across datasets they are not — so benchmark structure, not model
choice, is the decision-relevant variable, and single-benchmark evaluations of RAG orchestration
generalise less than the literature assumes.

The remaining findings qualify and mechanise this picture. The mid-tier model owns the accuracy–cost
frontier on both datasets — the strongest model is never the economical choice, and the frontier
configuration is Qwen3-32B with iteration on MuSiQue and with parallel decomposition on MultiHop-RAG.
Decomposition presupposes reliable structured output and silently collapses on the weakest model,
on both datasets; iteration is the model-robust strategy. The iterative agent's self-termination
decision is a usable confidence signal enabling a zero-cost abstention policy. Residual errors are
dominated by generation failing over evidence already in context, concentrated where that evidence
sits deep in the window. And a pilot-scale version of this evaluation reached the opposite conclusion
about retrievers before reversing at full sample size — a caution about small-sample evaluation
practice that the results above show to be far from hypothetical.
