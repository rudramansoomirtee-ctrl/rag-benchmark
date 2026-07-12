# Chapter 1 — Introduction

> **Status: complete draft for your review — not a hand-in.** Facts, gaps, RQs and contributions match
> the as-built project exactly, updated to the COMPLETED full matrix (both dataset arms, 8,400 runs);
> citations are drawn only from the verified base (WRITING_GUIDE §4). Before submission: (a) rewrite in
> your own voice — treat every sentence as replaceable, the *content* as fixed; (b) confirm the
> supervisor-agreed scope; (c) fill the one remaining `[CONFIRM]` marker (repository/archival statement).

---

## 1.1 Context

Large language models answer many questions fluently and some of them wrongly. Retrieval-augmented
generation (RAG) addresses the second problem by grounding the model's answer in documents retrieved
at query time (Lewis et al., 2020), and it has become the default architecture for question answering
over document collections that the model has not memorised. For single-fact questions the recipe is
straightforward: retrieve a handful of relevant passages, place them in the prompt, and ask the model
to answer from them.

Multi-hop questions break this recipe. A question such as *"What is the seat of the county sharing a
border with the county in which J. P. Hayes was born?"* cannot be answered from any single passage;
the evidence is distributed across documents, and the second retrieval depends on a fact established
by the first. Benchmarks constructed specifically to resist single-passage shortcuts — MultiHop-RAG
over news (Tang and Yang, 2024) and MuSiQue over Wikipedia paragraphs (Trivedi et al., 2022) — show
that standard single-pass RAG leaves a large fraction of such questions unanswered.

The research community's response has been a proliferation of *orchestration strategies*: control
logic wrapped around the same retrieve-and-generate primitives. Iterative approaches interleave
retrieval with reasoning, letting intermediate conclusions steer the next search (Trivedi et al.,
2023). Decomposition approaches split the question into single-hop sub-questions before retrieving —
either all at once (Ammann, Golde and Akbik, 2025) or sequentially, carrying each resolved answer into
the next sub-question (Press et al., 2023). Adaptive approaches try to route each question to the
cheapest strategy that can answer it (Jeong et al., 2024). Each family reports improvements over a
single-pass baseline. A practitioner deciding what to build, however, finds it surprisingly hard to
learn from this literature which strategy to choose, and at what price.

## 1.2 Problem statement

Three properties of the current evidence base make that decision hard.

First, published comparisons are confounded. Reported results on the same benchmark differ not only
in orchestration strategy but simultaneously in chunking scheme, embedding model, retriever, reranker,
prompt, and generator; no public leaderboard existed for MultiHop-RAG at the time of writing that would
normalise these choices (Appendix A). When a decomposition paper and an iteration paper each claim
gains over "standard RAG", the two baselines are rarely the same system, and the gains are not
comparable.

Second, cost is almost never part of the result. The strategies differ substantially in the number of
model calls they issue per question — an iterative agent may make ten times the calls of a single-pass
system — yet a survey of nineteen comparator papers conducted for this study (tabulated in
Appendix A) found dollar cost reported in only one appendix, and cost per correct answer reported nowhere.
For deployment decisions the relevant quantity is not accuracy alone but what each additional correct
answer costs.

Third, the evidence concentrates on models at the capability frontier, evaluated once, on samples small
enough that single-digit query differences move the conclusions. Whether the reported orderings of
strategies survive on the cost-efficient models that most deployments actually use — and whether they
are stable at all across models and samples — is largely untested.

## 1.3 Research gaps

This dissertation addresses three gaps that follow directly from the problems above.

**Gap 1 — no controlled comparison of orchestration strategies.** Existing studies change several
pipeline components at once. What is missing is an evaluation that freezes the retrieval substrate,
the prompts, the answer-context budget, and the generator, and varies *only* the orchestration
strategy, so that measured differences are attributable to orchestration alone.

**Gap 2 — cost is unaccounted.** No surveyed study reports cost per correct answer, and almost none
report dollar cost at all. An evaluation that records the billed cost of every model call, and relates
it to accuracy, does not yet exist for this family of systems.

**Gap 3 — the cost-efficient regime is untested.** Controlled evidence on models below the frontier —
where structured-output reliability, instruction following, and reasoning depth degrade unevenly — is
absent, leaving open whether strategy rankings transfer across the model tier that most production
systems occupy.

## 1.4 Aim, objectives and research questions

The aim of the study is to determine, under controlled and cost-accounted conditions, which
orchestration strategy a practitioner should choose for multi-hop retrieval-augmented generation on
cost-efficient language models. Five objectives operationalise the aim:

- **O1.** To implement four orchestration strategies — single-pass, iterative retrieval, parallel
  decomposition, and sequential decomposition — behind a single interface on one frozen retrieval
  substrate, each paired with a dense-retrieval-only ablation (a 4×2 factorial).
- **O2.** To evaluate the factorial under three language models spanning the cost-efficient tier, on
  two multi-hop benchmarks with deliberately different retrieval characteristics, over identical
  seeded question samples.
- **O3.** To record the billed cost of every model call and derive cost per correct answer and the
  accuracy–cost Pareto frontier across all configurations.
- **O4.** To quantify the statistical reliability of the comparisons — paired tests, bootstrap
  confidence intervals, and multiplicity correction — and the stability of strategy rankings across
  models, datasets, and sample sizes.
- **O5.** To derive evidence-based deployment recommendations from the resulting matrix, including
  the behaviour of the strategies when a model cannot support their assumptions.

The study is organised around four questions:

- **RQ1.** How does answer accuracy vary across orchestration strategies — single-pass, iterative
  retrieval, parallel decomposition, and sequential decomposition — when the retrieval substrate,
  prompts, context budget, and generator are held constant, and how does this vary with question depth
  (hop count)?
- **RQ2.** What does each strategy cost per correct answer, and which strategy–model configurations
  are Pareto-efficient in the accuracy–cost plane?
- **RQ3.** Is the ranking of strategies stable across language models spanning a capability gradient
  within the cost-efficient tier?
- **RQ4.** How predictable and robust are these systems — do rankings replicate across samples of
  practical size, and what happens to each strategy when a model cannot support the structured
  outputs it assumes?

## 1.5 Approach

The study is a controlled computational experiment. Eight systems form a 4×2 factorial: four
orchestration strategies — single-pass (A), iterative retrieval (B), parallel decomposition (F), and
sequential decomposition (F-seq) — each paired with a dense-retrieval-only twin that ablates the
hybrid retrieval pipeline, isolating the retriever's contribution for every orchestration. All eight
share one frozen substrate: the same hybrid BM25-plus-dense retriever with cross-encoder reranking,
the same embedding model, the same answer prompt, an identical twenty-chunk answer-context budget, and
deterministic decoding. The factorial is evaluated under three language models chosen to span a
capability gradient within the cost-efficient tier — Amazon Nova Lite, Qwen3-32B, and DeepSeek-V3 —
on two multi-hop benchmarks with deliberately different retrieval characteristics: MultiHop-RAG, whose
news corpus rewards lexical matching, and MuSiQue, whose distractors are constructed adversarially to
lexical retrieval (Trivedi et al., 2022; cf. Thakur et al., 2021 on the dataset-dependence of
retrievers). Every run records the billed cost of each model call; correctness is scored by a
deterministic containment metric with lexical secondaries; and every experiment persists a provenance
fingerprint — commit hash, configuration, and the exact query sample — so that all reported numbers
are reproducible from the released artefact. The final samples comprise 150 MuSiQue questions and
200 MultiHop-RAG questions, seeded and stratified, identical across every cell: 8,400 evaluated runs
in total.

## 1.6 Contributions

The dissertation makes five contributions.

1. **A controlled orchestration benchmark.** The first evaluation, to the author's knowledge, that
   compares single-pass, iterative, and both parallel and sequential decomposition strategies on a
   frozen retrieval substrate with a uniform context budget, across models and datasets, so that
   differences are attributable to orchestration alone (Gap 1).
2. **Cost per correct answer as a first-class metric.** Billed per-call cost is recorded for every
   run and aggregated into cost-per-correct and an accuracy–cost Pareto analysis across all
   strategy–model configurations (Gap 2).
3. **Empirical findings in the cost-efficient regime** (Gap 3), the central one being that *which
   orchestration wins is a property of the dataset*: iterative retrieval ranks first on MuSiQue's
   sequentially dependent hops while parallel decomposition fails there, yet on MultiHop-RAG the
   ranking inverts and parallel decomposition leads both single-pass and iterative retrieval — a
   nominally significant advantage, directionally unanimous across models though not surviving
   multiplicity correction — replicating the direction of Ammann, Golde and Akbik (2025) on their
   benchmark while showing it does not transfer. The retrieval pipeline's value is likewise
   dataset-dependent, and here the evidence is multiplicity-robust (large and per-model significant on
   news; small and pooled-only on the lexically-adversarial benchmark); residual errors are dominated
   by generation failures occurring while the gold evidence is already present in context.
4. **Behavioural analysis of agentic RAG.** The iterative agent's self-termination decision is shown
   to be a usable confidence signal: answering only when the agent stops early yields large precision
   gains at tunable coverage, at zero additional cost.
5. **A demonstration of small-sample unreliability.** A documented case in which conclusions drawn at
   n=50 reversed at n=150 under paired testing — quantifying, within one study, why the field's
   prevailing single-run, small-sample evaluation practice is unsafe.

All code, configurations, and run-level data are released for reproduction. `[CONFIRM: repository
access/archival statement per university requirements]`

## 1.7 Scope and delimitations

The study is deliberately bounded. The model panel is confined to the cost-efficient tier; frontier
proprietary models are excluded by design, since the contribution targets the regime where cost
matters, and cross-model claims are framed as rank stability across that tier rather than as absolute
capability comparisons. MuSiQue is evaluated in a pooled-distractor retrieval setting — harder than
its native per-question setting, easier than open-domain Wikipedia — so absolute scores are not
directly comparable to either, although within-study comparisons are unaffected. Correctness is
measured by deterministic lexical metrics; faithfulness and hallucination measurement are out of
scope and noted as future work. All experiments are in English, on public research datasets, with
low ethical risk.

## 1.8 Thesis structure

Chapter 2 reviews the literature on retrieval-augmented generation, multi-hop question answering,
orchestration strategies, and evaluation practice, and develops the three gaps. Chapter 3 describes
the experimental design: the systems, the frozen substrate, the datasets and sampling, the metrics,
and the reproducibility protocol. Chapter 4 reports and discusses the results of the two studies —
the orchestration comparison and the retrieval-pipeline factorial — together with the behavioural
analyses. Chapter 5 answers the research questions, states the contributions and limitations, derives
recommendations for practice, and outlines future work.
