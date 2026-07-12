# Chapter 5 — Conclusions, Recommendations and Future Work

> **Status: complete draft for your review — not a hand-in.** Every claim below is established in
> Chapter 4 and traceable to the verified analysis record; no new results are introduced. Before
> submission: rewrite in your own voice; confirm the appendix letters once the appendices are final.

---

## 5.1 Answers to the research questions

**RQ1 — How does accuracy vary across orchestration strategies under matched conditions?** The answer
the study did not expect to give is that it depends on the benchmark more than on the strategy. On
MuSiQue, whose hops depend sequentially on one another, iterative retrieval ranked first under every
model, improving on single-pass retrieval by two to seven points (nominally significant pooled across
models, p = .019), while parallel decomposition failed to improve on single-pass retrieval at all. On
MultiHop-RAG, whose questions mostly demand independent pieces of evidence, the ranking inverted:
parallel decomposition was the best system under both capable models (nominally ahead of single-pass,
p = .011, and of iteration, p = .049), and iteration no longer paid for itself (p = .80 against
single-pass). No orchestration contrast survived multiplicity correction, so these are consistent
directions rather than certified effects; but the *pattern* — the same eight systems changing order
between two benchmarks, identically under three models — is the finding. A proposed mechanism
accounts for both halves: parallel decomposition must phrase later-hop sub-questions before earlier
hops resolve, which is fatal when hops depend on one another and harmless — indeed, pure retrieval
coverage — when they do not. Accuracy declined with hop depth for every system and model, and no
strategy escaped that gradient.

**RQ2 — What does each strategy cost per correct answer, and which configurations are
Pareto-efficient?** The economics are dominated by model choice, not orchestration choice. The
accuracy–cost frontier on both benchmarks is owned by the two cheaper models; every configuration of
the strongest model in the panel is dominated. On MuSiQue the frontier ends at the mid-tier model with
iteration, which beats the strongest model's iterative system on accuracy at roughly a quarter of the
cost per correct answer; on MultiHop-RAG the frontier collapses to two points, the smallest model
single-pass and the mid-tier model with parallel decomposition, the latter dominating everything else
on both axes at once. Within a model, iteration costs roughly four times as much per correct answer
as single-pass retrieval, and decomposition sits between the two. The whole 8,400-run matrix cost
about $25 in generation charges — a fact worth stating because it means evaluations of this kind are
affordable to any research group, and the literature's silence on cost (Chapter 2) is not explained
by cost being hard to measure.

**RQ3 — Are strategy rankings stable across models?** Within a dataset, yes: the eight-system
rankings correlate strongly across the three models (Kendall τ-b between 0.64 and 0.74, all
significant), the top-ranked orchestration is the same under every model, and the dense-only twins
occupy the bottom of the ranking almost everywhere. An orchestration choice made on one cost-efficient
model therefore carries to its neighbours. Across datasets, no: the systems that rank first on the
two benchmarks differ, under every model. The decision-relevant variable is the benchmark's question
structure, not the generator — a distinction that single-model, single-benchmark evaluations, which
is to say most of the literature, cannot observe.

**RQ4 — How predictable and robust are these systems?** Two results answer this. First, robustness:
the smallest model in the panel could not execute the decomposition systems — its structured
decomposition output failed to parse on roughly 85% of questions, on both datasets, collapsing both
decomposition variants to single-retrieval behaviour — while the iterative system, whose free-text
routing assumes nothing about structured output, ran at full capability on the same model.
Decomposition presupposes reliable structured generation; that assumption fails silently below a
capability threshold, and iteration is the strategy that degrades least. Second, predictability: a
fifty-question pilot within this project produced a confident retrieval conclusion — dense-only
retrieval beating the hybrid pipeline — that reversed in every cell at n = 150 under paired testing.
The study thereby documents, within itself, the failure mode of the field's prevailing single-run,
small-sample evaluation practice.

Alongside the four planned answers, one unplanned result deserves the summary. In contrast to the
orchestration effects, the retrieval-pipeline effect is multiplicity-robust: the hybrid pipeline beat
its dense-only ablation in all twenty-four cells on MuSiQue (pooled p = 5.5 × 10⁻⁵) and by up to
sixteen points on MultiHop-RAG, individually significant within every model, with the advantage
concentrated almost entirely in comparison questions that name their sources — a forty-two point gap
where queries contain lexical anchors, and essentially none where they do not. The pipeline below the
orchestration matters more, more reliably, than the orchestration itself.

## 5.2 Contributions

The dissertation set out to make five contributions (§1.6); the results support all five, with the
statistical honesty each requires. It delivers the first controlled comparison of single-pass,
iterative, parallel-decomposition and sequential-decomposition orchestration on a frozen retrieval
substrate (Gap 1), and finds the comparison's outcome to be dataset-conditional. It makes cost per
correct answer a first-class reported metric (Gap 2), and finds it decisive: the frontier
configuration is never the strongest model. It provides controlled evidence in the cost-efficient
model tier (Gap 3), including the finding that a strategy family can silently stop working below a
capability threshold. It contributes a behavioural analysis showing the iterative agent's
self-termination decision is a usable, zero-cost confidence signal. And it demonstrates, with a
documented internal replication failure, why small-sample RAG evaluation is unsafe. The negative and
boundary results — decomposition failing on MuSiQue, iteration failing to pay on MultiHop-RAG,
orchestration effects that do not survive multiplicity correction — are reported with the same
prominence as the positive ones, which the literature reviewed in Chapter 2 rarely does.

## 5.3 Recommendations for practice

Five recommendations follow directly from the results, for a practitioner building multi-hop RAG on
cost-efficient models.

1. **Choose orchestration by the workload's hop structure, not by leaderboard results.** If questions
   resolve sequentially — each fact needed to phrase the next query — use iterative retrieval. If
   questions need several independent pieces of evidence, use parallel decomposition, which delivers
   more accuracy than iteration there at roughly a third of iteration's cost per correct answer. A
   benchmark result transfers only to workloads that share the benchmark's structure.
2. **Spend the model budget on a mid-tier model with the right orchestration, not a stronger model.**
   On both benchmarks, the mid-tier model with the dataset-appropriate strategy dominated every
   configuration of the strongest model on accuracy and cost simultaneously.
3. **For small models, prefer iteration.** Decomposition depends on structured output that small
   models cannot reliably produce, and the failure is silent — the system degrades to single-pass
   behaviour without erroring. Free-text control loops are the robust design at the bottom of the
   capability range.
4. **Use the agent's self-termination as a confidence signal.** Answering only when the iterative
   agent stops early filtered out roughly two-thirds of its errors while retaining three-quarters of
   its correct answers at the practical operating point, at zero additional cost. Budget exhaustion
   identifies intrinsically hard questions; routing them to a cheaper system makes results worse, not
   better.
5. **Keep the hybrid retrieval pipeline, especially where queries name things.** Its advantage is
   multiplicity-robust on both benchmarks, an order of magnitude larger where questions contain
   lexical anchors, and it did not become a liability even on a benchmark whose distractors were
   constructed adversarially to lexical retrieval.

## 5.4 Limitations

The limits of these claims are stated throughout the thesis and collected here. The orchestration
findings are nominally significant and directionally unanimous but do not survive multiplicity
correction at these sample sizes; they are offered as consistent evidence, not certified effects.
Two benchmarks cannot span "multi-hop question answering": both are English, public, and factoid, and
the dataset-dependence result itself warns against generalising from any small benchmark set — this
study's included. MuSiQue was evaluated in a pooled-distractor setting harder than its native form
and easier than open-domain retrieval, so absolute scores are not comparable to published numbers in
either setting. The model panel is confined to the cost-efficient tier by design; the frontier tier
may reorder the strategies. Correctness is measured by lexical containment with lexical secondaries —
deterministic and comparable, but blind to semantically correct paraphrase and to faithfulness, which
was descoped. The reranker's API cost is metered separately from generation cost and identically for
all hybrid systems, so it affects the level, not the ordering, of the cost analysis. The
position-sensitivity finding is correlational, since context position was assigned by the reranker
rather than randomised. And the proposed mechanism for the cross-dataset inversion — sequential hop
dependence versus independent evidence requirements — is an interpretation consistent with the data,
not a demonstrated cause.

## 5.5 Future work

Six extensions follow naturally. First, a randomised-position replication of the position-sensitivity
finding, shuffling gold-evidence position in the answering context to remove the reranker confound.
Second, an oracle-retrieval condition — answering over the gold passages directly — to bound the
generation ceiling that §4.7 suggests is the binding constraint. Third, adaptive routing between
iteration and parallel decomposition keyed on predicted hop dependence: the cross-dataset inversion
implies a classifier that detects whether a question's sub-questions are sequentially dependent could
capture the better strategy on both workload types, extending the Adaptive-RAG line from
"how much processing" to "which orchestration". Fourth, restoring faithfulness measurement, so that
the generation-side failures §4.7 localises can be separated into grounding failures and reasoning
failures. Fifth, sample sizes powered for the orchestration contrasts to survive multiplicity
correction — the observed discordant-pair rates make the required n estimable in advance. Sixth,
extending the matrix upward to frontier models and outward to further benchmarks, to test where the
dataset-dependence result itself stops generalising.

## 5.6 Concluding remark

The question this dissertation began with — which orchestration strategy should a practitioner choose
for multi-hop RAG? — turned out to be underspecified. Under matched conditions, at matched budgets,
across three models and 8,400 runs, no strategy is best in general: iteration wins where hops chain,
decomposition wins where evidence fans out, and the retrieval pipeline beneath both matters more,
more reliably, than either. The practical contribution of the study is not a winner but a decision
rule, priced in dollars per correct answer; its methodological contribution is a demonstration, from
inside a single project, that the field's customary evaluation practice — one model, one benchmark,
one small sample, no intervals — can manufacture confident conclusions that a controlled replication
reverses.
