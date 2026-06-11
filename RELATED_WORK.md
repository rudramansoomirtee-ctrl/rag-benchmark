# Related Work — verified comparison against this study's design

Compiled 2026-06-11 via 5-angle web research (MultiHop-RAG papers · iterative/agentic
methods · decomposition + index-structure methods · trained/RL methods · metric
conventions). Every number below carries the URL it was verified at in the source
reports; **⚠ = not fully verified, re-check the PDF before it goes in print**.
Read together with `DISSERTATION_AUDIT.md` (action register) — this file is the
related-work evidence base. Do not re-run this search; update in place.

## 1. The frame

This study holds retriever (hybrid BM25+dense+RRF+rerank), generator (budget LLM,
temperature 0), corpus, and prompts constant, varying **only orchestration**:
A (single-shot) · B (iterative reformulation agent) · F (parallel decomposition+RRF)
· F-tuned (F + few-shot decomposer + weighted RRF + source-metadata fan-out + CoT).
The literature divides cleanly by *what each method changes* — that classification
decides what is and is not citable as a fair comparator.

## 2. MultiHop-RAG (the benchmark) — verified ground truth

**Tang & Yang 2024, COLM** (arXiv:2401.15391, github.com/yixuantt/MultiHop-RAG):
2,556 queries (Inference 816 / Comparison 856 / Temporal 583 / Null 301), news
corpus, 256-token chunks, evidence in 2–4 docs.

- **Metric set: {Hits@4, Hits@10, MRR@10, MAP@10}** for retrieval; **accuracy** for
  generation. Our `retrieval-eval` already emits this exact set.
- **Official answer scorer (verified in `qa_evaluate.py`): word-set intersection**
  (`len(set(gold.split()) & set(pred.split())) > 0`) — *looser* than our
  `contains_match`. Our primary metric is therefore **stricter than the official
  benchmark scoring**, not a deviation. Cite this when defending containment.
- Best original retrieval: voyage-02 + bge-reranker-large → **Hits@10 0.7467,
  Hits@4 0.6625, MRR@10 0.586**.
- Generation: **GPT-4 accuracy 0.56 with best retrieved chunks vs 0.89 with gold
  evidence** ⚠ (multi-source incl. secondary; re-check Table 6) — the
  benchmark-level retrieval ceiling; cite alongside our N3 per-system ceiling.
- **No leaderboard exists** (PapersWithCode page has no results table); published
  results use non-comparable chunkers/embedders/LLMs → direct support for Gap 1.

### Papers reporting on MultiHop-RAG

| Paper | Changes | Headline (verified) | LLM |
|---|---|---|---|
| **Ammann, Golde & Akbik 2025** (ACL SRW, arXiv:2507.00355) | orchestration (decompose → per-sub-q dense retrieve → merged-pool rerank) | **MRR@10 +36.7%, answer F1 +11.6%**; QD+RR → **Hits@10 87.2%, MRR@10 0.635**; QD alone +4.4pp Hits@4, RR alone +7.6pp | Qwen2.5-32B-Instruct, **temp 0.8/top-p 0.8** ⚠(model/temp from one agent's read); bge-large-en-v1.5 + FAISS (dense-only) + bge-reranker-large |
| **Multi-Meta-RAG** (Poliakov & Shvai 2024, arXiv:2406.13213, ICTERI/Springer) | retrieval-side **metadata filtering** (LLM-extracted source/date → DB filter) | MRR@10 0.6016 → **0.6748**; Hits@4 **+17.2% rel** (Springer abstract says 18% ⚠ version drift); GPT-4 acc +7.89% rel, PaLM +25.6% rel ⚠ absolutes unverified | GPT-4, PaLM |
| **RAG vs GraphRAG** (Han et al. 2025, arXiv:2502.11371, third-party) | evaluation study | vanilla RAG **80.07 on Null** vs Community-GraphRAG(Local) **50.50** — graph/iterative retrieval over-answers nulls; HippoRAG-2 cells ⚠ backbone-dependent | GPT-4o-mini/Llama ⚠ |
| SPA knowledge-injection (arXiv:2603.22213, 2026) ⚠ | **training** | Qwen2.5-7B 60.91→86.64 acc ⚠ unverified | trained 7–8B |

Ammann's stated limitation — pipeline is **single-shot; sub-queries never updated
on retrieved evidence; limits adaptive multi-step reasoning** — is a **verified
paraphrase, not a verbatim quote** (PDF bot-blocked). Keep the CLAUDE.md caution:
check the PDF before quoting. System B is precisely the iterative comparator
this limitation calls for.

## 3. Comparability matrix

**Comparable under the orchestration-only constraint** (frozen retriever+LLM, prompting only):

| Method | Family | Metrics they report | Benchmarks | LLM |
|---|---|---|---|---|
| IRCoT (Trivedi+ ACL'23, 2212.10509) | iterative (CoT-interleaved) | para-recall@budget, **answer F1** (EM appendix) | HotpotQA/2Wiki/MuSiQue/IIRC | GPT-3 code-davinci-002, **Flan-T5 base→XXL** |
| Self-Ask (Press+ 2023, 2210.03350) | sequential decomposition | accuracy/EM, **Cover-EM** (containment, appendix), F1 | 2Wiki/MuSiQue/Bamboogle/CC | davinci-002 (+Google Search) |
| ReAct (Yao+ ICLR'23) | agent loop (tool calls) | answer EM | HotpotQA (27.4→35.1 w/ CoT-SC combo) | PaLM-540B |
| FLARE (Jiang+ EMNLP'23, 2305.06983) | active retrieval (confidence-triggered) | EM/F1, retrieval-trigger rate | 2Wiki EM **51.0** vs 39.4 single-shot | text-davinci-003 |
| Iter-RetGen (Shao+ EMNLP'23-F, 2305.15294) | iterative (answer-conditioned re-retrieve) | **EM, LLM-judged Acc†, answer-recall, gains-vs-T** | HotpotQA 31.6→**45.2** (T=3) | text-davinci-003 + Contriever |
| Adaptive-RAG (Jeong+ NAACL'24, 2403.14403) | router (trained T5 classifier ⚠ borderline) | EM, F1, **containment Acc, Step, relative Time** | MuSiQue/HotpotQA/2Wiki | GPT-3.5, Flan-T5-XL/XXL |
| ReSP (Jiang+ 2024, 2407.13101) | iterative + summarizer memory | EM/F1 | HotpotQA +4.1 F1 over iterative SOTA | Llama3-8B |
| BeamAggR (Chu+ ACL'24, 2406.19820) | tree decomposition + answer aggregation | token F1 (**+8.5% avg over SOTA**) | HotpotQA/2Wiki/MuSiQue/Bamboogle | GPT-3.5 |
| **FlashRAG** (Jin+ 2024, 2405.13576, WWW'25) | **toolkit: 23 methods, one frozen retriever+LLM** | **EM/F1 only — no cost, no latency** | NQ/HotpotQA/2Wiki… (not MultiHop-RAG) | Llama3-8B + e5 |

**Not comparable — they change the index/retrieval structure** (cite as context, not as competitors):
RAPTOR (ICLR'24, summary tree; QuALITY/QASPER/NarrativeQA — not multi-hop benchmarks),
GraphRAG (2404.16130, entity KG + community summaries; **LLM-judged win rates, no EM/F1**),
HippoRAG (NeurIPS'24, KG+PPR; R@2/R@5, EM/F1; **the $ precedent**: $0.10 vs IRCoT $1–3
per 1k queries, 10–30× cheaper / 6–13× faster vs IRCoT), HippoRAG 2 (2502.14802),
LongRAG (2406.15319, 4K-token retrieval units; AR@k, EM), Multi-Meta-RAG (metadata
filter — but the closest published cousin of F-tuned's source fan-out).

**Not comparable — they train weights** (the SOTA frontier; classify honestly):
Self-RAG (ICLR'24, reflection tokens, 7B/13B), RQ-RAG (COLM'24, trained query
refinement), EfficientRAG (EMNLP'24, trained DeBERTa labeler/filter — iterates
*without* per-hop LLM calls; ~3× faster), Beam Retrieval (NAACL'24, trained
end-to-end retriever; retrieval EM 97.5 HotpotQA), CoRAG (NeurIPS'25, trained
retrieval chains; reports **token-consumption scaling**), Search-R1 (COLM'25, RL;
EM-only; ⚠ headline % drifted across arXiv versions — cite camera-ready),
R1-Searcher (2503.05592, RL; **Cover-EM + LLM-judge** — same primary+secondary
pattern as ours), InstructRAG (ICLR'25), RankRAG (NeurIPS'24).
**None of these evaluates on MultiHop-RAG.**

## 4. Closest anchors per system (state these mappings in the dissertation)

- **A** ↔ the "standard RAG" baseline row every paper above compares against.
- **B** ↔ IRCoT / Iter-RetGen / ReAct family. Differences to state: B reformulates a
  *search query* (not a CoT sentence like IRCoT, not a full answer like Iter-RetGen);
  typed route step; hard budget ≤5; budget LLM not GPT-3.5/PaLM.
- **F** ↔ Ammann et al. (the recipe F mirrors) — but **F is not a replication**;
  exact deltas: our hybrid+RRF retriever (theirs dense-only FAISS), per-sub-question
  rerank then RRF fusion (theirs merged-pool single rerank), temperature 0 (theirs
  0.8), budget LLM (theirs Qwen-32B), bge-reranker-v2-m3 (theirs bge-reranker-large).
  Also kin: Self-Ask (sequential, ours parallel), BeamAggR (tree, answer-level
  aggregation vs our retrieval-level fusion).
- **F-tuned** ↔ Multi-Meta-RAG's metadata filtering + Ammann decomposition + CoT
  prompting, stacked. Present as engineering ceiling, not a clean arm (audit W7).
- **The study design itself** ↔ FlashRAG is the nearest published controlled
  comparison. Carve-out vs FlashRAG: (1) cost axis ($ + cost-per-correct — FlashRAG
  has none), (2) cross-model rank stability (FlashRAG: one LLM), (3) MultiHop-RAG
  + per-question-type breakdown, (4) HHEM faithfulness on every run.

## 5. Metrics catalogue — field standard ↔ this repo

| Metric | One-line definition | Canonical users | In repo |
|---|---|---|---|
| Hits@4 / Hits@10 | ≥1 gold chunk in top-k | **MultiHop-RAG**, Ammann, Multi-Meta-RAG | ✅ `hit_at_k` (retrieval-eval) |
| MRR@10 | mean 1/rank of first gold | MultiHop-RAG, Ammann | ✅ `reciprocal_rank_at_k` |
| MAP@10 | mean avg-precision at gold ranks | MultiHop-RAG | ✅ `average_precision_at_k` |
| Recall@k | frac. gold in top-k | HippoRAG (R@2/R@5), ours R@5 | ✅ `recall_at_k` |
| Precision@k | frac. top-k that is gold | less standard | ✅ `precision_at_k` |
| Retrieval EM/F1 (set match) | predicted passage set == gold set | Beam Retrieval | ❌ N/A (URL-keyed gold; low value) |
| Answer EM (SQuAD norm) | normalized string equality | HotpotQA/MuSiQue/IRCoT | ✅ `exact_match` — **deviation: ours does NOT strip articles a/an/the (SQuAD does)** → slightly stricter; disclose or align |
| Token F1 (SQuAD) | token-bag P/R harmonic mean | IRCoT, Adaptive-RAG, Ammann, BeamAggR | ✅ `token_f1` (C1; article-strip matches SQuAD) |
| Containment accuracy (Cover-EM / Acc) | gold appears in response | **MultiHop-RAG official (word-intersection — looser)**, Adaptive-RAG, Self-RAG, Self-Ask app., R1-Searcher ACC_R | ✅ `contains_match` (primary) — stricter than official scorer |
| LLM-judge | CRAG rubric perfect/acceptable/missing/incorrect | CRAG (KDD'24): human **1/0.5/0/−1**; auto-eval merges to 3-way 1/0/−1; R1-Searcher ACC_L | ✅ `judge.py` — **we use the human-rubric weights (4-way, 0.5)**; disclose variant |
| Supporting-fact EM/F1, joint EM/F1 | sentence-level evidence match | HotpotQA | ❌ N/A (no sentence gold; our R@5 plays this role) |
| Faithfulness | grounded-in-context score | RAGAS (LLM-based; excluded by design), **HHEM** (classifier; beats GPT-3.5/4 zero-shot on RAGTruth-style benchmarks), RAGTruth response-level P/R/F1 | ✅ HHEM mean + %flagged, RAGTruth-calibrated threshold |
| Steps / #retrievals | avg retrieve-generate rounds | **Adaptive-RAG "Step"**, Iter-RetGen T, Auto-RAG | ✅ `n_steps` / `avg_trajectory_length` |
| Time / latency | per-query time (Adaptive-RAG: *relative*) | Adaptive-RAG | ✅ `latency_ms` absolute + N2 p50/p95 (stronger) |
| Token consumption | tokens per query | CoRAG; third-party (π-CoT, A2RAG) | ✅ `tokens_in/out` |
| **Dollar cost** | $ per query/run | **HippoRAG appendix only** ($0.10 vs IRCoT $1–3 / 1k queries) | ✅ `cost_usd` |
| **Cost-per-correct** | $ ÷ #correct | **no precedent found — our contribution** | ✅ `cost_per_correct` |
| Failure rate | runs that crashed | not reported in the field | ✅ `pct_failed` |
| Uncertainty | CIs/seeds | **field norm: single-run, no CIs** (IRCoT/Adaptive-RAG/HippoRAG/Self-RAG spot-checked); GraphRAG win-rate p-values the exception | ✅ seeded sample + bootstrap CI (N2) — exceeds convention |

## 6. What this verifies about the slides

- **Gap 1 (confounded evaluations): supported.** No MultiHop-RAG leaderboard;
  published numbers mix chunkers/embedders/LLMs. FlashRAG is the partial
  counterexample (frozen stack) — cite it and state the carve-out (§4).
- **Gap 2 (cost overlooked): supported, with named exceptions.** Across 19 papers:
  zero report cost-per-query in the orchestration family; the only first-party $
  is HippoRAG's appendix; EfficientRAG (latency/LLM-calls) and CoRAG (tokens) treat
  efficiency seriously; Adaptive-RAG reports steps + *relative* time. Phrase as
  "dollar cost is almost never reported; no paper reports cost-per-correct."
- **Gap 3 (frontier-only): needs the refinement already flagged (W6).** IRCoT also
  ran Flan-T5-base→XXL and FlashRAG runs Llama3-8B — so say: *controlled
  orchestration comparisons on budget commercial API models with cost accounting
  are absent*, not "evidence is frontier-only".

### Citation cautions (feeds audit W8)
1. Ammann limitation quote = verified paraphrase only — check PDF before verbatim.
2. GPT-4 0.56/0.89 — re-check Tang & Yang Table 6 before printing.
3. Multi-Meta-RAG +17.2% vs Springer's "18%" — arXiv/Springer version drift; pick one and cite that version.
4. Search-R1 headline % drifted across arXiv versions — cite the COLM 2025 camera-ready.
5. Ammann model/temperature details (Qwen2.5-32B, temp 0.8) verified by one agent only — confirm in PDF.

## 7. Verification ledger

- **Code-verified (strongest):** MultiHop-RAG official scorer (`qa_evaluate.py` word-intersection).
- **Two-independent-agent convergence:** Ammann headline numbers; HippoRAG cost claim; dollar-cost absence across families.
- **Single-source / flagged ⚠:** GPT-4 0.56/0.89 split; Multi-Meta-RAG absolutes; Ammann stack details; RAG-vs-GraphRAG cell values; SPA 2026 numbers.
- **Environment caveat:** arXiv/ACL PDFs were bot-blocked during the search; verification ran through official GitHub repos, anthology landing pages, and search-indexed paper text.
