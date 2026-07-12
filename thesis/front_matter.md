# Front matter — abstract, declaration, preliminaries

> **Status: drafts + checklist for your review — not a hand-in.** The abstract is a content-complete
> draft to rewrite in your voice. The declaration section contains a DRAFT AI-usage disclosure that
> you must adapt to the university's current declaration form wording — check the exact form your
> faculty issues before submission. Everything under "Preliminaries checklist" is assembled at Word
> conversion, not written here.

---

## Abstract (draft, ~290 words)

Retrieval-augmented generation (RAG) systems answer multi-hop questions by wrapping control logic —
an *orchestration strategy* — around retrieve-and-generate primitives, but published comparisons of
these strategies confound orchestration with retriever, prompt, and generator choices, almost never
report cost, and concentrate on frontier models. This dissertation evaluates four orchestration
strategies — single-pass, iterative retrieval, parallel decomposition, and sequential decomposition —
under matched conditions: a frozen hybrid retrieval substrate, a uniform twenty-passage answer
budget, identical prompts, and deterministic decoding. Each strategy is paired with a dense-only
retrieval ablation, forming a 4×2 factorial evaluated under three cost-efficient language models
(Amazon Nova Lite, Qwen3-32B, DeepSeek-V3) on two multi-hop benchmarks with deliberately different
retrieval characteristics (MultiHop-RAG and MuSiQue) — 8,400 runs in total, with the billed cost of
every model call recorded.

The central finding is that which orchestration wins is a property of the dataset, not of the
method: iterative retrieval ranks first under every model on MuSiQue's sequentially dependent hops,
where parallel decomposition fails to beat single-pass retrieval, while on MultiHop-RAG the ranking
inverts and parallel decomposition leads at a third of iteration's cost. Orchestration effects are
directionally unanimous across models but do not survive multiplicity correction; the retrieval
pipeline's advantage over its dense-only ablation does, in every model on news and pooled on the
adversarial benchmark. The accuracy–cost frontier is owned entirely by the two cheaper models. The
iterative agent's self-termination is shown to be a zero-cost confidence signal, residual errors are
predominantly generation-side with gold evidence already in context, and a fifty-question pilot
conclusion that reversed at full sample size documents why small-sample RAG evaluation is unsafe.

**Keywords:** retrieval-augmented generation; multi-hop question answering; query decomposition;
agentic RAG; cost-effectiveness; evaluation methodology.

---

## Declaration — AI tools usage disclosure (DRAFT — adapt to the faculty form)

> **Why this section exists:** the university's declaration form requires disclosure of AI tool
> usage. Given how this project was built, the disclosure below is the honest and defensible
> statement. Pair it with the standing rule: every sentence in the submitted thesis is written by
> you; the AI-generated chapter scaffolds in this repository are source material, not submission
> text.

Draft statement:

*In the course of this project, AI assistants (Anthropic Claude, accessed via Claude Code) were used
under my direction as development and analysis tools: for assistance in implementing and debugging
the benchmark codebase; for executing and monitoring experiment runs; for statistical computation
and verification of results (all of which are independently reproducible from the released code and
database); for literature-search assistance, with every cited claim subsequently verified by me
against the primary source; and for producing working notes and draft outlines that I used as source
material. The text of this dissertation was written by me. All experimental results reported are the
output of the released, deterministic evaluation pipeline and are reproducible from the archived
repository. Responsibility for the content, analysis, and conclusions is entirely my own.*

`[ACTION: check the exact AI-usage wording/checkbox on the current UoM declaration form and adapt.
If the form asks for named tools and purposes, the paragraph above enumerates them.]`

---

## Preliminaries checklist (assemble at Word conversion, in institutional order)

| Item | Source / action |
|---|---|
| Title page | Faculty template — title, name, student ID, degree, department, supervisor, date |
| Declaration form | Faculty form + the AI disclosure above |
| Abstract | This file, §Abstract — rewritten in your voice |
| Acknowledgements | You (supervisor, family — keep short) |
| Table of contents | Auto-generate in Word from heading styles |
| List of figures | Auto-generate (Figures 3.1–3.7, 4.1–4.4) |
| List of tables | Auto-generate (Tables 3.1, 4.1–4.4 + appendix tables) |
| List of abbreviations | RAG, RRF, BM25, kNN, HNSW, CI, EM, F1, MRR, MAP, LLM, API, SRW |

## Main-text assembly order

chapter1_introduction.md → chapter2_literature.md → chapter3_methodology.md →
chapter4_results.md (retitled "Results and Discussion") → chapter5_conclusions.md →
References (from references_draft.md, ⚠/[verify] all resolved, tags stripped) →
Appendices A–F (thesis/appendices/).

**Strip before assembly:** every "Status:" banner block, the Ch2 verification end-note, and all
draft-only bracketed notes. Formatting at conversion: 1.5 line spacing, ≥10pt Times New Roman or
Arial, numbered headings, captions below figures / above tables.
