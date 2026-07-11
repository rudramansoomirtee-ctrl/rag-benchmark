# Chapter 3 — Methodology

> **Status: complete draft for your review — not a hand-in.** This is a prose rewrite of the earlier
> scaffold: same facts and design, academic register, internal-document citations replaced by primary
> sources or appendix pointers. Before submission: (a) rewrite in your own voice; (b) export the seven
> figures — Mermaid source for each lives in `thesis/figures/figure_3_1.mmd` … `figure_3_7.mmd` — to
> numbered images (paste each `.mmd` into mermaid.live, or run `mmdc -i figure_3_1.mmd -o figure_3_1.png`
> for each); (c) resolve the remaining `[CONFIRM]` markers (hardware; MultiHop sample when the second
> dataset is run); (d) attach the reproducibility appendix.

---

## 3.1 Research design

This study is a controlled computational experiment. Its aim is not to identify the best possible
retrieval-augmented generation pipeline, but to isolate the effect of two design choices on answer
quality and cost: the *orchestration strategy* — the control logic that decides how many retrieval
queries to issue and how to phrase them — and the *retrieval pipeline* that serves those queries.
Everything else is held fixed.

The design comprises two studies over one frozen substrate. In the first, the retriever is held
constant and four orchestration strategies are compared. In the second, every strategy is paired with
a deliberately weakened, dense-retrieval-only twin, producing a 4×2 factorial in which the retrieval
pipeline itself becomes a manipulated variable. Because the corpus, prompts, answer-context budget,
decoding parameters and evaluation harness are identical in every cell, measured differences within a
model are attributable to the manipulated factor rather than to incidental engineering choices. The
factorial is repeated under three language models and two benchmarks, so the stability of any
conclusion can be examined across generators and data distributions.

Determinism is enforced wherever a language model is called (temperature zero throughout); every
strategy is implemented behind a single programming interface and executed by one runner; and each
experiment records a provenance fingerprint — the exact code version, configuration, and query sample
— sufficient to reproduce it (§3.8).

*Figure 3.1 — experimental design overview. [Source: `thesis/figures/figure_3_1.mmd` — export before submission.]*

## 3.2 The controlled comparison

**Study 1: orchestration.** Four systems share the same retriever (§3.4), the same generator and
answer prompt, and the same answer-context budget of twenty passages. System A retrieves once and
answers over its top twenty. Systems B, F and F-seq issue several retrieval queries and fuse the
resulting ranked lists by reciprocal rank fusion, answering over the fused top twenty. With the
budget uniform across all four, the systems differ in a single respect: how the queries that fill
those twenty context slots are produced. B produces them sequentially, each conditioned on the
evidence already gathered; F produces them in parallel, by decomposing the question up front; F-seq
decomposes like F but resolves the sub-questions in order, substituting each resolved answer into the
next retrieval query. Two contrasts of particular interest follow. F against F-seq isolates parallel
versus sequential decomposition — a comparison the literature has not made under matched conditions
(Chapter 2) — and F-seq against B isolates pre-planned decomposition against free-form iterative
reformulation.

The twenty-passage budget was fixed after a sensitivity ablation on pilot data showed that the
optimum differs by strategy: iterative accumulation performed better with a narrower context, and
decomposition with a wider one. A single constant was adopted in preference to per-system tuning,
trading a few points of one system's accuracy for a comparison in which context size cannot explain
any difference. The ablation is reported in Appendix `[X]`.

**Study 2: the retrieval pipeline.** Each of the four systems has a twin — A-minus, B-minus, F-minus,
F-seq-minus — identical in every respect except that retrieval is restricted to dense
nearest-neighbour search alone: no lexical matching, no rank fusion, no reranking. The twins are
budget-matched to their parents, so each parent–twin difference isolates the retrieval pipeline's
contribution for that orchestration, and the pattern of differences across orchestrations and
datasets tests whether the pipeline's value is uniform or conditional. This factorial construction
replaced an earlier design in which several retrieval improvements were stacked into one system;
stacking confounds levers, and the revised design manipulates the retriever as a single factor
instead.

*Figure 3.2 — the single-variable orchestration comparison. [Source: `thesis/figures/figure_3_2.mmd`.]*

## 3.3 The eight systems

Table 3.1 fixes the nomenclature. Codes are used in tables and figures, names in prose; the
dense-only twins are tagged, and the hybrid retriever is otherwise implicit.

| Code | Name | Orchestration | Retriever |
|---|---|---|---|
| A | Single-pass RAG | one retrieval, one answer | hybrid |
| A-minus | Single-pass RAG (dense-only) | as A | dense kNN only |
| B | Iterative RAG | reformulation loop, at most five retrievals | hybrid |
| B-minus | Iterative RAG (dense-only) | as B | dense kNN only |
| F | Parallel decomposition | decompose once; retrieve sub-questions concurrently | hybrid |
| F-minus | Parallel decomposition (dense-only) | as F | dense kNN only |
| F-seq | Sequential decomposition | decompose; resolve hops in order | hybrid |
| F-seq-minus | Sequential decomposition (dense-only) | as F-seq | dense kNN only |

*Table 3.1 — system nomenclature: the 4×2 retrieval–orchestration factorial.*

**Single-pass RAG (A)** retrieves twenty passages for the question and answers over them in one model
call. It is the baseline every comparable study measures against, and the degenerate case to which
the other strategies reduce when no further queries are warranted.

**Iterative RAG (B)** is a bounded loop. At each step the model receives the original question and
the evidence gathered so far, and replies in one of two forms: an instruction to answer now, or a new
search query. The loop ends when the model elects to answer, or after five retrievals, whichever
comes first; a dedicated call with the shared answer prompt then produces the final response.
Evidence accumulates across iterations — routing and answering operate on the fused ranking of every
iteration's results — so a passage found early remains visible late. Free-text routing was adopted
after a typed, schema-constrained routing call proved unreliable on smaller models, which emit the
decision as prose rather than parseable structure; the free-text form behaves consistently across the
model panel and halves the per-iteration call count. B's nearest published relatives are the
iterative family discussed in Chapter 2, from which it differs chiefly in reformulating a search
query, rather than a chain-of-thought sentence or a draft answer, and in running on cost-efficient
models.

**Parallel decomposition (F)** makes one model call that splits the question into two to four
single-hop sub-questions, using a few-shot prompt. It retrieves for the original question and for
each sub-question concurrently over the shared retriever, fuses the ranked lists, and answers once
over the fused top twenty. A single-hop question yields an empty decomposition, in which case F
reduces to A. F follows the parallel decompose-and-rerank recipe of Ammann, Golde and Akbik (2025) in
outline, but is not a replication: the retriever, fusion procedure, decoding temperature and
generator all differ, which is material when interpreting any divergence between their findings and
this study's.

**Sequential decomposition (F-seq)** shares F's decomposer and answer prompt but resolves the ordered
sub-questions one at a time. Before each hop's retrieval, the answers already resolved are
substituted into the sub-question, so that a descriptive reference such as "the spouse of that
director" becomes a named entity; a small model call then answers the sub-question from its own
retrieved context, and the resolved fact carries forward. A hop that cannot be answered is marked
unknown and is not substituted, which prevents a failed hop from contaminating later queries. The
final answer is generated over the fused union of all hops' retrievals, with the resolved
intermediate facts supplied alongside the passages. This is the self-ask pattern of Press et al.
(2023) transposed onto a fixed retrieval substrate, and it exists to make the parallel-versus-
sequential contrast with F a single-variable comparison.

Both decomposition systems degrade deliberately rather than fail: if the decomposition call cannot be
parsed, the system proceeds with no sub-questions — that is, as System A — and the event is visible
in the recorded retrieval count. The fallback matters beyond robustness. On the weakest model in the
panel it converts a would-be crash into a measurable finding about which strategies small models can
execute at all (§3.5).

*Figures 3.3–3.5 — control flow of systems B, F and F-seq. [Sources: `thesis/figures/figure_3_3.mmd`,
`figure_3_4.mmd`, `figure_3_5.mmd`.]*

## 3.4 The shared retrieval substrate

Every system calls one retrieval entry point. Queries are embedded with a 768-dimensional
sentence-embedding model; the corpus is indexed for approximate nearest-neighbour search (HNSW,
cosine similarity) alongside a standard inverted index. A retrieval call runs lexical (BM25) and
dense searches in parallel and fuses the two rankings by reciprocal rank fusion — a rank-based method
chosen because it is indifferent to the incomparable score scales of its inputs (Cormack et al.,
2009) — into a candidate pool of forty. A cross-encoder reranker (Cohere Rerank 3.5) then rescores
query–passage pairs jointly and returns the top twenty. The pool is deliberately twice the final
budget, so the reranker selects rather than merely reorders. Multi-query systems fuse their per-query
rankings with the same reciprocal rank fusion, client-side.

The dense-only ablation used by the four twin systems bypasses this pipeline with a single switch:
retrieval returns the top twenty by dense similarity, with no lexical arm, no fusion and no
reranking. Because the switch applies per system rather than globally, ablated and full-pipeline
systems run side by side within the same experiment, on the identical query sample.

One presentation detail is shared by all systems because piloting showed it to be consequential:
each retrieved passage is rendered with its source and title metadata, not text alone. MultiHop-RAG's
comparison questions frequently identify an article by its publisher, which the passage text does not
expose; without the metadata such questions fail across every system even when retrieval succeeds.
Since the formatting is common to all systems, it cannot advantage any one strategy.

*Figure 3.6 — the retrieval substrate and the dense-only ablation path. [Source: `thesis/figures/figure_3_6.mmd`.]*

## 3.5 Models and inference

The three generators span a capability gradient within the cost-efficient tier: Amazon Nova Lite, a
small and inexpensive model; Qwen3-32B, mid-sized; and DeepSeek-V3, a large mixture-of-experts model
that is nonetheless inexpensive to serve. All are accessed through AWS Bedrock with deterministic
decoding and a fixed output-length ceiling. The billed cost of every call is read from the provider
response and persisted with the run, so the cost analysis rests on actual charges rather than
estimates; the reranker is metered separately by the provider, and its cost is reported separately
from generation cost. Frontier proprietary models are excluded by design: the study's contribution is
evidence in the regime where cost is a first-class constraint, and cross-model claims are framed as
rank stability across this gradient rather than as absolute capability comparisons.

The panel produced one methodological finding that shapes how results are reported, and it is stated
here in advance. Nova Lite cannot reliably execute the decomposition systems: its structured output
fails to parse for roughly five questions in six, at which point F and F-seq fall back to
single-retrieval behaviour. Rather than excluding those cells, the study runs the full symmetric
matrix and reports Nova's decomposition cells explicitly as degraded — they approximate Nova's
single-pass system — because the degradation itself is evidence for the robustness question in RQ4.

## 3.6 Datasets and sampling

Two multi-hop benchmarks are used, chosen because their retrieval characteristics differ in a way
that matters to Study 2.

**MultiHop-RAG** (Tang and Yang, 2024) provides 2,556 queries over a news corpus in four types —
inference, comparison, temporal, and null (unanswerable) — with gold evidence distributed across two
to four articles. The corpus is chunked into 256-token passages; gold evidence is keyed by article,
and retrieved passages are mapped to their parent article for retrieval scoring. Null questions are
retained throughout, because a strategy's tendency to answer the unanswerable is itself informative.

**MuSiQue** (Trivedi et al., 2022) provides 2–4-hop questions whose construction resists shortcut
answering. Its distractor paragraphs are mined with BM25 against the question with intermediate
answers masked — distractors adversarial to lexical retrieval by design. Each question ships with
roughly twenty candidate paragraphs; this study pools all ingested questions' paragraphs into a
single index of three thousand passages, so each query competes against the full pool rather than
against its own twenty. The pooled setting is harder than the dataset's native reading-comprehension
form and easier than open-domain retrieval over Wikipedia; absolute scores are therefore not directly
comparable to either, though within-study comparisons are unaffected. An integrity check confirmed
that every gold paragraph is present and indexed.

Queries are selected once per dataset as a seeded, stratified sample — by question type for
MultiHop-RAG, by hop count for MuSiQue — and the identical sample is reused by every system and every
model, so all comparisons are paired. The MuSiQue sample comprises 150 questions (78 two-hop, 45
three-hop, 27 four-hop; seed 42). The MultiHop-RAG sample comprises 200 questions
`[CONFIRM when the MultiHop arm is run]`. Pilot experiments used fifty-question samples and exhibited
sampling noise large enough to reverse conclusions between runs; the final sample sizes were raised
accordingly, and the pilot-to-final comparison is itself reported in Chapter 4 as evidence on the
reliability of small-sample evaluation.

## 3.7 Metrics and scoring

Correctness is reported through a layered set of measures with one deterministic primary.

The primary metric is **containment accuracy**: whether the normalised gold answer appears in the
normalised response. This matches the convention of the MultiHop-RAG benchmark for short factoid
answers (Tang and Yang, 2024) and the containment-style accuracy used elsewhere in the multi-hop
literature (Jeong et al., 2024). For MuSiQue, which supplies acceptable answer aliases, the matcher
scores against the gold and its aliases, with whole-word matching in both directions so that a terser
correct answer still scores. Several adaptations are disclosed rather than silently applied: text
after an explicit final-answer marker is scored when present; standard refusal phrasings count as
correct on null questions; unicode dashes are normalised; and common entity suffixes are tolerated.
The study's containment is stricter than the benchmark's own released scorer, which accepts any word
overlap, so reported accuracy errs conservative.

Secondary measures are **exact match** — normalised string equality, strict, and expected to be near
zero for verbose instruction-tuned generators; reported for completeness and as part of the metric
audit — and **token-level F1** in the SQuAD style, comparable to the answer-F1 reported by the
decomposition literature. Retrieval quality is reported with the benchmark's own measures — hit rate,
mean reciprocal rank and mean average precision at standard cutoffs — together with precision and
recall at five. An LLM-as-judge protocol was implemented but excluded from the final evaluation by
design: the primary and secondary metrics are deterministic, and removing the judge removes a
non-deterministic, costed dependency. Faithfulness measurement was descoped early in the project and
is discussed as future work; no faithfulness numbers are reported.

Every run also records billed cost, token counts, latency, and the number of retrievals performed.
Two aggregates carry the economic analysis: total cost, and **cost per correct answer** — total cost
divided by the number of correct answers — for which no precedent was found in the literature
surveyed for this study (Chapter 2; the survey is tabulated in Appendix `[X]`). A crashed run counts
as a wrong answer: failures remain in the accuracy denominator and their rate is reported rather than
hidden. In the final matrix, no run failed.

## 3.8 Reproducibility, provenance and ethics

*Figure 3.7 — the reproducible, idempotent pipeline: ingest, index, run the two studies at a frozen
commit, compute metrics, analyse. [Source: `thesis/figures/figure_3_7.mmd`.]*

The full stack — document store, search index, tracing service and application — is containerised,
and every stage of the pipeline (ingestion, indexing, experiment execution, metric computation) is
idempotent and resumable: re-running a command continues from where it stopped, enforced by a
uniqueness constraint on the experiment–system–query triple, so an interrupted evaluation never
re-bills completed work. Each experiment persists a provenance fingerprint: the git commit hash, the
full configuration (retriever settings, budgets, model identifiers, and the pricing-library version),
a corpus fingerprint, and the exact query identifiers of the sample. The final matrix was run at a
single frozen commit, on one container image, against one index build. Hardware is recorded because
the embedder runs locally and shapes latency `[CONFIRM: CPU/RAM]`. Every model call is traced
end-to-end via OpenTelemetry for post-hoc inspection, and the repository, configurations and
run-level data are released for reproduction (Appendix `[X]`).

The study carries low ethical risk: it uses two public research datasets, contains no personal or
sensitive data, and makes only metered, authorised calls to commercial model APIs. The principal
threats to validity are stated where they arise: the cost-efficient panel bounds the generality of
cross-model claims (§3.5); cross-provider latency comparisons carry a serving-infrastructure confound
and are restricted to within-model contrasts (§3.7); and the MuSiQue retrieval findings must be
interpreted in light of that benchmark's BM25-mined distractor construction (§3.6), a point taken up
with the results in Chapter 4.
