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
(Jeong et al., 2024). Trained routers, reflection tokens (Asai et al., 2024), and reinforcement-learned
search policies sit beyond the scope of prompting-level orchestration, but they mark the frontier the
prompting family is implicitly compared against.

Across all three families, reported gains are real but the comparisons are loose: each paper's
baseline is its own implementation, on its own retrieval stack, usually with one generator, and the
strategies are almost never run against each other by a third party under matched conditions.

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
al., 2024), and as Cover-EM in the reinforcement-learning line of work. Containment tolerates verbose
generations that exact match rejects, which is why studies of instruction-tuned generators favour it;
its known weakness — crediting a lucky mention — argues for pairing it with stricter lexical
secondaries rather than replacing it.

More consequential than metric choice is what the literature does *not* report. Three omissions
stand out.

**Cost.** The orchestration families differ by an order of magnitude in model calls per question,
yet dollar cost is almost never reported. In a structured survey of nineteen comparator papers
conducted for this study (Appendix `[X]`), exactly one first-party dollar figure was found — an
appendix of the HippoRAG paper, reporting roughly $0.10 per thousand queries for its method against
$1–3 for iterative retrieval (Gutiérrez et al., 2024) — and *no* paper reported cost per correct
answer, the quantity a deployment decision actually turns on. Efficiency proxies appear occasionally
(step counts and relative latency in Adaptive-RAG; token counts in a few others), but the economic
axis is essentially absent.

**Controlled comparison.** Published numbers on the same benchmark are rarely comparable, because
chunking, embedder, retriever, reranker, prompt and generator all vary together; MultiHop-RAG has no
public leaderboard that would impose a common configuration. The nearest exception is FlashRAG, a
toolkit that re-implements many methods over one frozen retriever and generator (Jin et al., 2024) —
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

> **Verification end-note (remove before submission; action per WRITING_GUIDE §4).** Verified against
> primary sources during this project: Lewis et al. 2020 author list; Trivedi et al. 2022 distractor
> construction; Trivedi et al. 2023 Table 4; Press et al. 2023 Table 1/14; Jeong et al. 2024 Table 8;
> Thakur et al. 2021 abstract; Ammann et al. 2025 headline numbers; Chu et al. 2024 Table 1 (their
> "+8.5%" abstract claim does not reconcile with the table — cite table values only). Still requiring
> PDF re-verification before print (⚠): Karpukhin et al. 2020 full details; Yang et al. 2018; Shao et
> al. 2023; Jiang et al. 2023; Yao et al. 2023; Zhou et al. 2023; Asai et al. 2024; Cormack et al.
> 2009; Gutiérrez et al. 2024 (HippoRAG cost figures); Jin et al. 2024 (FlashRAG scope).
