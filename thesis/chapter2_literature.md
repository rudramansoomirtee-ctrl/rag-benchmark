# Chapter 2 — Literature Review

> **Status: complete draft for your review — not a hand-in.** Written from scratch against the verified
> citation base (WRITING_GUIDE §4); nothing is inherited from the earlier chap 2 document, whose
> references failed verification. Before submission: (a) rewrite in your own voice; (b) re-verify every
> reference marked ⚠ in the end-note against the published PDF; (c) attach the cost-survey table as an
> appendix (it evidences §2.5's central claim).

---

## 2.1 Retrieval-augmented generation and the multi-hop problem

Retrieval-augmented generation couples a parametric language model with a non-parametric document
index: at inference time, passages relevant to the query are retrieved and supplied to the model as
context, so that the answer can be grounded in text the model never memorised (Lewis et al., 2020).
The approach inherits a long lineage from open-domain question answering, where dense passage
retrieval had already displaced purely lexical search for single-fact questions (Karpukhin et al.,
2020). For such questions the pipeline is now routine. The interesting failures begin when no single
passage contains the answer.

Multi-hop questions require composing facts across documents: the identity established by one passage
becomes the subject of the next retrieval. Datasets built to study this — HotpotQA (Yang et al., 2018)
and its successors — initially suffered from a defect that flattered single-pass systems: many
"multi-hop" questions could be answered from one passage through lexical shortcuts. Two more recent
benchmarks address this directly and are used in this study. MuSiQue composes 2–4-hop questions from
verified single-hop components and, importantly, constructs its distractor paragraphs adversarially:
candidate distractors are mined with BM25 using the question with intermediate answers masked, so that
passages lexically similar to the question but useless for answering it surround the gold evidence
(Trivedi et al., 2022). MultiHop-RAG poses inference, comparison, temporal and null questions over a
news corpus, with gold evidence spread across two to four articles, and packages the corpus and
queries specifically for evaluating RAG pipelines rather than reading comprehension (Tang and Yang,
2024). The two benchmarks make usefully different demands on retrieval — a point taken up in §2.4.

## 2.2 Orchestration strategies for multi-hop RAG

If one retrieval pass cannot gather the evidence, the system must issue more than one query. The
literature has explored three broad families of control logic — what this dissertation calls
*orchestration* — differing in how the additional queries are produced.

**Iterative retrieval.** IRCoT interleaves retrieval with chain-of-thought reasoning: each reasoning
step conditions the next retrieval, and the loop continues until an answer emerges. On MuSiQue this
raised answer F1 from 29.4 (single retrieval) to 36.5 with a GPT-3-class model, with similar gains on
three other multi-hop datasets (Trivedi et al., 2023). Related designs vary what is fed back into the
next query: Iter-RetGen conditions retrieval on the previous draft answer (Shao et al., 2023), FLARE
triggers re-retrieval when the model's token confidence drops (Jiang et al., 2023), and ReAct
interleaves free-form actions with tool calls in a single trajectory (Yao et al., 2023). The common
claim is that *what to retrieve next depends on what has already been derived* — and the common cost
is more model calls per question, a point these papers rarely price.

**Question decomposition.** A second family splits the question before retrieving. Self-Ask prompts
the model to pose and answer explicit follow-up sub-questions sequentially, improving over
chain-of-thought on compositional 2-hop questions (Press et al., 2023); least-to-most prompting makes
the same sequential argument for reasoning tasks generally (Zhou et al., 2023). Decomposition can
instead be parallel: Ammann, Golde and Akbik (2025) decompose the question once, retrieve for every
sub-question independently, rerank the merged pool, and report retrieval MRR@10 gains of 36.7% and
answer-F1 gains of 11.6% on MultiHop-RAG. The parallel form is cheaper — one decomposition call and
concurrent retrievals — but it must phrase later-hop sub-questions before earlier hops are resolved,
so bridge entities can only be referred to descriptively. Whether that matters in practice is exactly
the kind of question the current evidence base cannot answer, because sequential and parallel
decomposition have not been compared under matched conditions. BeamAggR combines decomposition with
answer-level aggregation over reasoning trees and reports the strongest decomposition-family F1 on
MuSiQue open-domain (36.9 with GPT-3.5 and web search), though its aggregation operates on candidate
answers rather than on retrieval results, which limits its comparability to fusion-based designs
(Chu et al., 2024).

**Adaptive routing.** Adaptive-RAG trains a classifier to route each question to no-retrieval,
single-step, or multi-step processing, motivated by the observation that multi-step strategies waste
calls on easy questions. Its GPT-3.5 results on MuSiQue illustrate both halves of the argument:
multi-step processing raised containment accuracy from 23.6 to 31.6 over single-step, while
single-step retrieval actually *underperformed* no retrieval on exact match — retrieval noise can hurt
(Jeong et al., 2024). Adaptive-RAG's router, however, decides only *how much* processing a question
receives — none, one step, or many — not *which kind*: the choice between iterating and decomposing,
which the present study finds to be the consequential one, is outside its action space.

**The trained frontier.** A fourth line of work moves the control logic from the prompt into the
model's weights, and marks the boundary of this dissertation's scope. ReAct interleaves free-form
reasoning with tool actions in a single generated trajectory, so the model itself decides when to
search and what to search for (Yao et al., 2023); its flexibility is also its weakness for controlled
study, since the trajectory conflates reasoning, retrieval policy and answer generation in one
stream, and a malformed action derails the episode. Self-RAG trains the generator to emit reflection
tokens that trigger retrieval and critique its own drafts (Asai et al., 2024), and more recent work
optimises the search behaviour itself with reinforcement learning from answer rewards
(Jin, B. et al., 2025; Song et al., 2025). These
approaches require training access to the model and per-task reward engineering; the present study
confines itself to prompting-level orchestration precisely because that is the design space available
to a practitioner deploying commercial models behind an API — but the trained line matters here for a
second reason: it is where evaluation conventions such as containment-style scoring (Cover-EM) have
been carried forward, and its reported gains face the same confounding critique developed below.

Across all four families, reported gains are real but the comparisons are loose: each paper's
baseline is its own implementation, on its own retrieval stack, usually with one generator, and the
strategies are almost never run against each other by a third party under matched conditions. Two
further patterns recur. Gains are largest against weak baselines — IRCoT's improvements are measured
against single-retrieval BM25, Ammann, Golde and Akbik's against a dense-only retriever — leaving
open how much headroom remains over a well-engineered single-pass system. And each family is
typically evaluated on the benchmark family that suits it: iterative methods on sequentially
composed questions, decomposition methods on MultiHop-RAG's largely parallel evidence requirements —
a selection effect that, if real, would predict exactly the dataset-dependence this study sets out
to test.

## 2.3 The retrieval substrate: lexical, dense, hybrid, reranked

Orchestration sits on top of a retriever, and the retrieval literature gives clear reasons not to
treat that substrate as interchangeable. The BEIR benchmark evaluated retrievers zero-shot across
eighteen datasets and found no universal winner: BM25 proved "a robust baseline", reranking and
late-interaction models achieved the best average zero-shot quality, and dense bi-encoders — despite
their in-domain strength — often underperformed out of domain (Thakur et al., 2021). Production
practice has converged on hybrid designs that run lexical and dense retrieval in parallel and fuse
the rankings, commonly with reciprocal rank fusion, a rank-based method that ignores incomparable
score scales (Cormack et al., 2009), followed by a cross-encoder reranker that rescores query–passage
pairs jointly.

The components of that pipeline embody a quality–cost trade-off that is rarely made explicit.
Bi-encoder embedders are cheap at query time because passages are embedded once, offline, and a
query costs a single forward pass; the price is that query and passage never attend to each other.
Cross-encoders reverse the trade: scoring each query–passage pair jointly is what makes reranking
effective — the design traces to BERT-based passage re-ranking (Nogueira and Cho, 2019) — but it
cannot be precomputed, so it is applied only to a shallow candidate pool, and when consumed as a
commercial API it is metered per call. Embedders have meanwhile specialised: retrieval-tuned models
trained specifically for augmenting language models (Zhang et al., 2023) now displace general
sentence encoders in RAG stacks. Two things follow for any study in this space. The retrieval substrate is
itself a bundle of tuned choices, which strengthens the case for freezing it while orchestration
varies; and part of a RAG system's per-query cost sits in retrieval components that the cost
discussions of §2.4 — where they exist at all — do not count.

Two consequences matter for this dissertation. First, retriever quality and orchestration are
confounded in most published comparisons: a decomposition method evaluated over dense-only FAISS
retrieval (as in Ammann et al., 2025) and an iterative method evaluated over BM25 (as in Trivedi et
al., 2023) differ in two dimensions at once. Second, the benchmarks themselves interact with the
retriever choice: MuSiQue's distractors were *mined with BM25*, which raises the untested question of
whether lexical retrieval components are differentially harmed there — the kind of retriever–dataset
interaction BEIR's results predict in general but that no study has measured for multi-hop RAG
orchestration specifically.

## 2.4 Evaluation practice: metrics, samples, and what goes unreported

Answer-correctness conventions vary across this literature in ways that complicate comparison. SQuAD
exact match and token-level F1 remain the default for reading-comprehension-derived benchmarks;
containment-style accuracy — does the gold answer appear in the generated response — is used by
MultiHop-RAG's own evaluation (Tang and Yang, 2024), by Adaptive-RAG under the name Acc (Jeong et
al., 2024), and as Cover-EM in the reinforcement-learning line of work (§2.2). Containment tolerates
verbose generations that exact match rejects, which is why studies of instruction-tuned generators
favour it; its known weakness — crediting a lucky mention — argues for pairing it with stricter
lexical secondaries rather than replacing it.

A parallel evaluation tradition replaces lexical scoring with model judgement. RAGAS scores RAG
outputs on faithfulness and relevance dimensions using an LLM as the judge, without gold references
(Es et al., 2024), and LLM-as-judge protocols
are increasingly used for answer correctness itself. The appeal is semantic tolerance — a paraphrased
correct answer scores correctly — but the price is a non-deterministic instrument whose own errors
are correlated with the systems it judges, a costed dependency on another model, and reduced
comparability across studies that use different judges. For a benchmark whose purpose is controlled
comparison, deterministic metrics are the conservative choice, and that is the position this study
takes; the trade-off is acknowledged, since containment scoring under-credits semantically correct
paraphrase. A further generation-side result bears on evaluation design: language models use long
contexts unevenly, with accuracy degrading when the needed evidence sits in the middle of the
context window (Liu et al., 2024) — which implies that a RAG evaluation's context budget and
evidence ordering are experimental variables, not neutral plumbing, and motivates this study's fixed
context budget and its analysis of accuracy by evidence position.

More consequential than metric choice is what the literature does *not* report. Three omissions
stand out.

**Cost.** The orchestration families differ by an order of magnitude in model calls per question,
yet dollar cost is almost never reported. In a structured survey of nineteen comparator papers
conducted for this study (Appendix A), exactly one first-party dollar figure was found — an
appendix of the HippoRAG paper, reporting an online-retrieval cost of roughly $0.10 per thousand
queries for its method against $1–3 for iterative retrieval (Gutiérrez et al., 2024); the same
appendix shows the comparison reversing at indexing time, where HippoRAG costs $15 more per ten
thousand passages, which illustrates how partial even the rare cost reporting is — and *no* paper
reported cost per correct answer, the quantity a deployment decision actually turns on. Efficiency proxies appear occasionally
(step counts and relative latency in Adaptive-RAG; token counts in a few others), but the economic
axis is essentially absent.

**Controlled comparison.** Published numbers on the same benchmark are rarely comparable, because
chunking, embedder, retriever, reranker, prompt and generator all vary together; MultiHop-RAG has no
public leaderboard that would impose a common configuration. The nearest exception is FlashRAG, a
toolkit that re-implements many methods over one frozen retriever and generator (Jin, J. et al., 2025) —
but it reports only exact match and F1, prices nothing, evaluates a single generator, and does not
include MultiHop-RAG; the controlled-comparison idea has not yet been carried into the multi-hop,
cost-aware setting.

**Statistical reliability.** The prevailing practice is a single run on a single sample, without
confidence intervals; differences of a few points are routinely narrated as findings. Given that
several of the reported inter-method gaps are of the same magnitude as plausible sampling noise at
common sample sizes, the stability of published rankings — across samples, and across generators —
is largely an open question.

## 2.5 The gaps this study addresses

The threads above converge on three gaps.

**Gap 1.** No study compares single-pass, iterative, parallel-decomposition and
sequential-decomposition orchestration under matched conditions — one retriever, one prompt, one
context budget, one generator per cell — so the attribution question at the centre of the
orchestration literature remains open. The sequential-versus-parallel decomposition contrast, in
particular, has never been isolated, despite being a design fork every practitioner faces.

**Gap 2.** Cost per correct answer is unreported in the surveyed literature, and dollar cost nearly
so. No accuracy–cost frontier across orchestration strategies and models exists.

**Gap 3.** The evidence concentrates on frontier or research-scale models. Whether orchestration
rankings transfer to the cost-efficient commercial tier — where structured-output reliability and
reasoning depth degrade unevenly, and where most deployed systems operate — is untested, as is the
robustness of each strategy to a model that cannot support its assumptions.

This dissertation addresses the three gaps jointly: a factorial evaluation that freezes the retrieval
substrate and context budget, crosses four orchestration strategies with a retrieval-pipeline
ablation, prices every model call, and repeats the whole design across three cost-efficient models
and two benchmarks whose retrieval characteristics deliberately differ.

---

> **Verification end-note (remove before submission; action per WRITING_GUIDE §4).** All reference
> entries were verified against primary sources (arXiv/ACL Anthology/publisher pages) on 2026-07-12;
> see references_draft.md for per-entry status and source URLs. Corrections that affect this
> chapter's text and are already applied: the HippoRAG cost figure is for *online retrieval* (its
> indexing is dearer — now stated); FlashRAG's citable version is WWW 2025 (Jin, J. et al., 2025);
> Lost in the Middle is TACL 2024 (Liu et al., 2024); CRAG is NeurIPS 2024 D&B, not KDD. Claim-level
> spot-checks done earlier in the project: Trivedi et al. 2023 Table 4; Press et al. 2023 Tables
> 1/14; Jeong et al. 2024 Table 8; Thakur et al. 2021 abstract; Ammann et al. 2025 headline numbers;
> Chu et al. 2024 Table 1 (their "+8.5%" abstract claim does not reconcile with the table — cite
> table values only). All inline citations now have verified entries in references_draft.md; note
> the LLM-Embedder paper (Zhang et al., 2023) was retitled in a 2026 arXiv revision — pin the arXiv
> version if quoting from it.
