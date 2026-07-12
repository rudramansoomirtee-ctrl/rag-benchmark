# Appendix A — Comparator Survey: Cost Reporting and Leaderboard Absence

> Appendix draft — content copied from `RELATED_WORK.md`; renumber/reformat at Word conversion.

This appendix backs Chapter 1 §1.2 and Chapter 2 §2.4's claims about the nineteen comparator papers
surveyed for this study and the absence of a public MultiHop-RAG leaderboard (Gap 1) and of
cost-per-query / cost-per-correct reporting (Gap 2). All tables and text below are copied verbatim
from `RELATED_WORK.md` (compiled 2026-06-11 via 5-angle web research; ⚠ = not fully verified by the
source document, re-check the PDF before print).

---

## A.1 Leaderboard-absence evidence (Gap 1)

From `RELATED_WORK.md` §2 ("MultiHop-RAG — verified ground truth"):

> **No leaderboard exists** (PapersWithCode page has no results table); published results use
> non-comparable chunkers/embedders/LLMs → direct support for Gap 1.

From `RELATED_WORK.md` §6 ("What this verifies about the slides"):

> **Gap 1 (confounded evaluations): supported.** No MultiHop-RAG leaderboard; published numbers mix
> chunkers/embedders/LLMs. FlashRAG is the partial counterexample (frozen stack) — cite it and state
> the carve-out (§4 of `RELATED_WORK.md`).

## A.2 Papers reporting on MultiHop-RAG

Copied from `RELATED_WORK.md` §2:

| Paper | Changes | Headline (verified) | LLM |
|---|---|---|---|
| **Ammann, Golde & Akbik 2025** (ACL SRW, arXiv:2507.00355) | orchestration (decompose → per-sub-q dense retrieve → merged-pool rerank) | **MRR@10 +36.7%, answer F1 +11.6%**; QD+RR → **Hits@10 87.2%, MRR@10 0.635**; QD alone +4.4pp Hits@4, RR alone +7.6pp | Qwen2.5-32B-Instruct, **temp 0.8/top-p 0.8** ⚠(model/temp from one agent's read); bge-large-en-v1.5 + FAISS (dense-only) + bge-reranker-large |
| **Multi-Meta-RAG** (Poliakov & Shvai 2024, arXiv:2406.13213, ICTERI/Springer) | retrieval-side **metadata filtering** (LLM-extracted source/date → DB filter) | MRR@10 0.6016 → **0.6748**; Hits@4 **+17.2% rel** (Springer abstract says 18% ⚠ version drift); GPT-4 acc +7.89% rel, PaLM +25.6% rel ⚠ absolutes unverified | GPT-4, PaLM |
| **RAG vs GraphRAG** (Han et al. 2025, arXiv:2502.11371, third-party) | evaluation study | vanilla RAG **80.07 on Null** vs Community-GraphRAG(Local) **50.50** — graph/iterative retrieval over-answers nulls; HippoRAG-2 cells ⚠ backbone-dependent | GPT-4o-mini/Llama ⚠ |
| SPA knowledge-injection (arXiv:2603.22213, 2026) ⚠ | **training** | Qwen2.5-7B 60.91→86.64 acc ⚠ unverified | trained 7–8B |

## A.3 Comparability matrix

**Comparable under the orchestration-only constraint** (frozen retriever+LLM, prompting only) — copied from `RELATED_WORK.md` §3:

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

**Not comparable — they change the index/retrieval structure** (cite as context, not as competitors) — copied verbatim:

> RAPTOR (ICLR'24, summary tree; QuALITY/QASPER/NarrativeQA — not multi-hop benchmarks),
> GraphRAG (2404.16130, entity KG + community summaries; **LLM-judged win rates, no EM/F1**),
> HippoRAG (NeurIPS'24, KG+PPR; R@2/R@5, EM/F1; **the $ precedent**: $0.10 vs IRCoT $1–3
> per 1k queries, 10–30× cheaper / 6–13× faster vs IRCoT), HippoRAG 2 (2502.14802),
> LongRAG (2406.15319, 4K-token retrieval units; AR@k, EM), Multi-Meta-RAG (metadata
> filter — but the closest published cousin of F-tuned's source fan-out).

**Not comparable — they train weights** (the SOTA frontier; classify honestly) — copied verbatim:

> Self-RAG (ICLR'24, reflection tokens, 7B/13B), RQ-RAG (COLM'24, trained query
> refinement), EfficientRAG (EMNLP'24, trained DeBERTa labeler/filter — iterates
> *without* per-hop LLM calls; ~3× faster), Beam Retrieval (NAACL'24, trained
> end-to-end retriever; retrieval EM 97.5 HotpotQA), CoRAG (NeurIPS'25, trained
> retrieval chains; reports **token-consumption scaling**), Search-R1 (COLM'25, RL;
> EM-only; ⚠ headline % drifted across arXiv versions — cite camera-ready),
> R1-Searcher (2503.05592, RL; **Cover-EM + LLM-judge** — same primary+secondary
> pattern as ours), InstructRAG (ICLR'25), RankRAG (NeurIPS'24).
> **None of these evaluates on MultiHop-RAG.**

## A.4 Metrics catalogue — field standard vs. this repo

Copied from `RELATED_WORK.md` §5:

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
| Faithfulness | grounded-in-context score | RAGAS (LLM-based; excluded by design), **HHEM** (classifier), RAGTruth response-level P/R/F1 | ⚠ **descoped** — HHEM subsystem removed; `avg_faithfulness`/`pct_flagged` columns retained but unpopulated. Future work |
| Steps / #retrievals | avg retrieve-generate rounds | **Adaptive-RAG "Step"**, Iter-RetGen T, Auto-RAG | ✅ `n_steps` / `avg_trajectory_length` |
| Time / latency | per-query time (Adaptive-RAG: *relative*) | Adaptive-RAG | ✅ `latency_ms` absolute + N2 p50/p95 (stronger) |
| Token consumption | tokens per query | CoRAG; third-party (π-CoT, A2RAG) | ✅ `tokens_in/out` |
| **Dollar cost** | $ per query/run | **HippoRAG appendix only** ($0.10 vs IRCoT $1–3 / 1k queries) | ✅ `cost_usd` |
| **Cost-per-correct** | $ ÷ #correct | **no precedent found — our contribution** | ✅ `cost_per_correct` |
| Failure rate | runs that crashed | not reported in the field | ✅ `pct_failed` |
| Uncertainty | CIs/seeds | **field norm: single-run, no CIs** (IRCoT/Adaptive-RAG/HippoRAG/Self-RAG spot-checked); GraphRAG win-rate p-values the exception | ✅ seeded sample + bootstrap CI (N2) — exceeds convention |

## A.5 Cost-reporting gap verdict (Gap 2)

Copied from `RELATED_WORK.md` §6:

> **Gap 2 (cost overlooked): supported, with named exceptions.** Across 19 papers:
> zero report cost-per-query in the orchestration family; the only first-party $
> is HippoRAG's appendix; EfficientRAG (latency/LLM-calls) and CoRAG (tokens) treat
> efficiency seriously; Adaptive-RAG reports steps + *relative* time. Phrase as
> "dollar cost is almost never reported; no paper reports cost-per-correct."

And, from `DISSERTATION_AUDIT.md` §4 (W8 action item):

> New citable support: Gap 1 — no MultiHop-RAG leaderboard exists; Gap 2 — 19 papers audited, only
> HippoRAG reports $ (appendix), none report cost-per-correct.

## A.6 Verification ledger

Copied from `RELATED_WORK.md` §7 — grades the confidence level of every claim above:

> - **Code-verified (strongest):** MultiHop-RAG official scorer (`qa_evaluate.py` word-intersection).
> - **Two-independent-agent convergence:** Ammann headline numbers; HippoRAG cost claim; dollar-cost absence across families.
> - **Single-source / flagged ⚠:** GPT-4 0.56/0.89 split; Multi-Meta-RAG absolutes; Ammann stack details; RAG-vs-GraphRAG cell values; SPA 2026 numbers.
> - **Environment caveat:** arXiv/ACL PDFs were bot-blocked during the search; verification ran through official GitHub repos, anthology landing pages, and search-indexed paper text.
> - **Primary-verified 2026-06-28 (retriever ablation, §8):** BEIR abstract (arXiv:2104.08663) and MuSiQue distractor construction (ar5iv.org/abs/2108.00573) — quotes confirmed from the papers, not secondary pages.

## A.7 Citation cautions (feeds audit item W8)

Copied from `RELATED_WORK.md` §3 citation cautions list:

1. Ammann limitation quote = verified paraphrase only — check PDF before verbatim.
2. GPT-4 0.56/0.89 — re-check Tang & Yang Table 6 before printing.
3. Multi-Meta-RAG +17.2% vs Springer's "18%" — arXiv/Springer version drift; pick one and cite that version.
4. Search-R1 headline % drifted across arXiv versions — cite the COLM 2025 camera-ready.
5. Ammann model/temperature details (Qwen2.5-32B, temp 0.8) verified by one agent only — confirm in PDF.

---

**Not found in sources:** a literal count/list confirming exactly "19" named papers (the survey text
refers to "19 papers audited" as a summary count without itemising all 19 by name in one place — the
tables above list every individually named paper that could be found across `RELATED_WORK.md` §§2–3).
No separate PapersWithCode screenshot or leaderboard URL snapshot exists in the source documents beyond
the textual claim quoted in A.1.
