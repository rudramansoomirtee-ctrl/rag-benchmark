# Dissertation Writing Guide — the playbook

*Purpose: one place that maps every existing resource to every chapter, fixes the non-negotiable rules,
and gives the writing order. Work through it top-to-bottom. Compiled 2026-07-06 from the three
independent chapter reviews + the verified analysis.*

---

## 0. Ground rules (non-negotiable)

1. **Every sentence in the submitted thesis is written by you.** The repo's scaffolds (Ch3/Ch4 drafts,
   analysis report) supply *facts, structure and numbers* — never sentences. Rewrite, don't edit.
2. **No citation goes in without primary-source verification** (arXiv/ACL/publisher page — check the
   author list, venue, year, and the specific claim). Two of your existing documents contain
   *fabricated* reference metadata (the "Schwenk & Schwartz" tail; invented venues). One fabricated
   reference found by an examiner poisons trust in all of them.
3. **Internal documents are not citable.** `RELATED_WORK.md`, `DISSERTATION_AUDIT.md`,
   `musique_matrix_analysis.md`, notebook cells (N1–N5), CLI commands, file paths — all must be
   replaced in the thesis by the underlying primary sources, or moved to a reproducibility appendix.
4. **Nothing "pending" in a submitted thesis.** Either E4–E6 (MultiHop) runs, or the scope is
   permanently narrowed to MuSiQue-only in Ch1/Ch3 — no `[INSERT]`, `[CONFIRM]`, `[pending]` markers.
5. **Supervisor alignment first.** The registered BG proposal describes a different project
   ("Imperium" case study). Confirm the pivot to the benchmark study is agreed and on record before
   investing weeks of writing.

## 1. Writing order (inside-out) and status

Write from what is most settled outward. Recommended order:

| # | Chapter | Status (2026-07-11) | Gate |
|---|---|---|---|
| 1 | **Ch3 Methodology** | ✅ full prose draft; both arms final; one placeholder left (hardware, §3.8) | your voice rewrite |
| 2 | **Ch4 Results** | ✅ full prose draft, BOTH arms populated (9,600 runs; §4.8 written; no pending markers) | your voice rewrite |
| 3 | **Ch2 Literature review** | ✅ full prose draft from the verified base | your voice rewrite + ⚠ ref re-checks |
| 4 | **Ch1 Introduction** | ✅ full prose draft, updated to the completed matrix | your voice rewrite + supervisor scope |
| 5 | **Ch5 Conclusions** | ❌ not started — **now unblocked** (every RQ has complete evidence; see analysis §10 for the per-RQ answers) | none — the next writing task |
| 6 | Abstract, front matter, reference list, appendices, figure exports | not started | Ch5 |

Why this order: Ch3 re-grounds you in your own design (best warm-up); Ch4 is transcription of verified
results; Ch2/Ch1 are framing that must match what Ch3/Ch4 actually say; Ch5 answers RQs that Ch4 must
first establish; the abstract is written last, always.

## 2. Resource map — what feeds each chapter

| Chapter | Draw facts/structure from | Do NOT use |
|---|---|---|
| **Ch1** | The as-built design (Ch3 scaffold §3.1–3.2), Gaps 1–3 with their evidence (RELATED_WORK.md §6), the four RQs (RQ1 accuracy-by-orchestration · RQ2 cost-per-correct · RQ3 cross-model ranking · RQ4 rank-stability/robustness) | Plan.docx content, BG.pdf content (tonal/structural template ONLY — its section order is fine) |
| **Ch2** | RELATED_WORK.md (verified numbers, comparability matrix §3, metrics catalogue §5, gap evidence §6, cautions), the verified citation base (§4 below), sources.docx entries *that survive re-verification* | chap 2.docx (wrong project; fabricated refs) |
| **Ch3** | chapter3_methodology.md (facts/structure), CLAUDE.md (system truth), budget/step ablation results (analysis §4.1 note) | its own scaffold *sentences*; internal-doc citations |
| **Ch4** | chapter4_results.md (verified tables), musique_matrix_analysis.md (stats, behavioural §7, cost §5) + E4–E6 when run | scaffold sentences; "Finding (verified):" headline style; notebook/CLI refs |
| **Ch5** | Ch4's per-RQ summary (§4.7), the limitations already catalogued (analysis §8 + ⚠ checklist), future work (oracle run, faithfulness, escalation-up, F-seq conversion gap) | new claims not established in Ch4 |
| **Appendices** | Reproducibility: repo, frozen SHA, experiment IDs, config snapshot, CLI pipeline, notebook; the 19-paper cost-survey table (evidence for the "no precedent" claim); pilot history (exp36–43) as the small-sample case study | — |

## 3. Per-chapter one-page briefs

### Ch1 — Introduction (~1,500–2,000 words)
Structure (borrow the BG.pdf skeleton): context → problem → gaps → RQs → contributions → scope → thesis map.
- Motivation = **Gaps 1–3**, concretely evidenced: (1) published RAG results are confounded (no
  MultiHop-RAG leaderboard; mixed chunkers/embedders/LLMs); (2) dollar cost almost never reported,
  cost-per-correct never; (3) no controlled orchestration comparison on cost-efficient models.
- State the four RQs *exactly* as Ch4 answers them. Contributions list = controlled 4×2 factorial;
  cost-per-correct metric; the orchestration findings; the behavioural/abstention findings; the
  small-sample reliability demonstration.
- Scope honestly: cost-efficient model tier (not frontier); pooled-distractor MuSiQue setting;
  faithfulness out of scope.

### Ch2 — Literature review (~3,000–3,500 words)
Structure around **the argument, not paper summaries**: (i) RAG and multi-hop QA foundations →
(ii) orchestration families (iterative: IRCoT/Iter-RetGen/FLARE/ReAct · decomposition:
Self-Ask/least-to-most/Ammann/BeamAggR · adaptive: Adaptive-RAG) → (iii) retrieval pipelines & hybrid
vs dense (BEIR; rerankers) → (iv) benchmarks (MultiHop-RAG, MuSiQue and its BM25-mined distractors) →
(v) evaluation practice: metrics conventions, **the cost gap**, single-run norm → each section ends by
sharpening one of Gaps 1–3 → close with the gap statement this study fills.
- Be critical: every cited paper gets at least one limitation/qualifier. (The old chap2's total absence
  of criticism was itself an AI-tell.)
- Every number from RELATED_WORK.md carries its ⚠ where flagged (Ammann quote = paraphrase; Search-R1
  version drift; Multi-Meta-RAG drift; GPT-4 0.56/0.89 needs Table-6 re-check).

### Ch3 — Methodology (~3,000 words)
The scaffold's content survives; the language doesn't. Rewrite section by section:
- Kill: "purest", "cleanest possible", "crucially", "load-bearing", "degrade gracefully",
  "not merely X — it is Y", bolded verdict sentences, self-praise of the design.
- Convert: 7 Mermaid diagrams → exported numbered figures; code paths → Appendix; internal-doc
  citations → primary sources; `[CONFIRM]`/`[PLACEHOLDER]` → real values (models: DeepSeek-V3,
  Qwen3-32B, Nova Lite on Bedrock; N=150 MuSiQue seed 42; N=200 MultiHop seed 42 when run; hardware).
- Add: the budget ablation (why 20, held constant) as a short justification paragraph; the
  crash-is-wrong policy; the pilot→final protocol as design evolution (referenced to an appendix).

### Ch4 — Results (~3,000–3,500 words + tables/figures)
- Transcribe the verified tables; write the analysis prose fresh (the scaffold's "Analysis (expand)"
  bullets list the points to make — make them in paragraphs, with hedging where the stats say so).
- Statistical phrasing rules: retriever effect = *directionally universal, significant pooled*
  (p=5.5×10⁻⁵), never per-cell; B>A pooled p=.019; F-seq>F directional-only; F≯A stated as a novel
  finding in tension with Ammann (2025), resolved (or not) by the MultiHop arm.
- The pilot correction becomes a measured paragraph in a "reliability of small-sample evaluation"
  subsection (feeds RQ4) — no ⚠ boxes, no changelog tone.
- Behavioural results (§7 of the analysis) get their own subsection: termination-as-confidence,
  abstention policy, position sensitivity — with the causal caveats stated.

### Ch5 — Conclusions (~1,500–2,000 words)
One subsection per RQ, answering it in 2–3 sentences each from Ch4 evidence, then: contributions
(restated against Gaps 1–3), limitations (n per hop bucket; cost-efficient tier only; pooled-distractor
setting; MuSiQue finding partly by-construction; correlational position analysis), future work
(faithfulness; escalation-up cascade; randomized-position oracle; scaling the sample; native-distractor
comparison run).

## 4. Verified citation base (formal entries to build the reference list from)

Status: ✅ = primary-verified this project · ⚠ = verify against the PDF before print.

| Paper | Citation core | Status |
|---|---|---|
| MultiHop-RAG | Tang, Y. & Yang, Y. (2024). MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. COLM. arXiv:2401.15391 | ✅ (Table-6 GPT-4 split ⚠) |
| MuSiQue | Trivedi, H., Balasubramanian, N., Khot, T. & Sabharwal, A. (2022). MuSiQue: Multihop Questions via Single-hop Question Composition. TACL. arXiv:2108.00573 | ✅ (distractor construction quote verified) |
| IRCoT | Trivedi, H. et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning… ACL. arXiv:2212.10509 | ✅ (Table 4 F1 verified) |
| Self-Ask | Press, O. et al. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. arXiv:2210.03350 | ✅ (2-hop subset numbers verified) |
| Least-to-most | Zhou, D. et al. (2023). Least-to-Most Prompting… ICLR. arXiv:2205.10625 | ⚠ (not quote-verified) |
| Ammann | Ammann, N., Golde, J. & Akbik, A. (2025). Question Decomposition for RAG. ACL SRW. arXiv:2507.00355 | ✅ numbers / ⚠ limitation quote is a paraphrase |
| Adaptive-RAG | Jeong, S. et al. (2024). Adaptive-RAG… NAACL. arXiv:2403.14403 | ✅ (Table 8 containment verified) |
| BEIR | Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. & Gurevych, I. (2021). BEIR… NeurIPS D&B. arXiv:2104.08663 | ✅ (abstract quote verified) |
| Lost in the Middle | Liu, N. F. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172 | ⚠ (cite as U-shaped/forced-position; ours is observational) |
| BeamAggR | Chu, Z. et al. (2024). BeamAggR… ACL. arXiv:2406.19820 | ✅ Table 1 / ⚠ "+8.5%" headline |
| RAG | Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. | ⚠ re-verify — the BG.pdf version of this entry was FABRICATED |
| DPR | Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. & Yih, W. (2020). Dense Passage Retrieval… EMNLP. | ⚠ same warning |
| Iter-RetGen | Shao, Z. et al. (2023). arXiv:2305.15294 | ⚠ |
| FLARE | Jiang, Z. et al. (2023). EMNLP. arXiv:2305.06983 | ⚠ |
| ReAct | Yao, S. et al. (2023). ICLR. | ⚠ |
| Search-R1 / R1-Searcher | arXiv:2503.09516 (cite v5) / arXiv:2503.05592 | ✅ with version-drift note |
| CRAG (if judge mentioned) | Yang, X. et al. (2024). KDD. | ⚠ |

**Never reuse** a reference entry from chap 2.docx or BG.pdf without rebuilding it from the primary
source — both documents contain confirmed fabricated metadata.

## 5. Voice checklist (from the reviews — check every section against this before calling it done)

Banned words/patterns: *delve · landscape · pivotal · multifaceted · underscores · leverages ·
crucially/critically (sentence-initial) · "plays a crucial role" · "load-bearing" · "degrade(s)
gracefully" · "purest/cleanest possible" · "not merely X — it is Y" · "This confirms/validates a
critical principle" · "the economic logic is unambiguous" · Moreover/Furthermore/Additionally chains ·
bolded verdict-first sentences · ⚠/emoji callouts · "Marking hook" or any exam-strategy meta-text.*

Positive habits: vary paragraph length (some 2-sentence, some 8); hedge where the statistics hedge;
criticise at least one aspect of every cited paper; prefer plain verbs (shows, finds, reports) over
boosters; read each section aloud — if it sounds like a product page, rewrite it.

## 6. Mechanical conversion checklist

- [ ] Export all Mermaid diagrams → numbered PNG/SVG figures with captions (mermaid.live or `mmdc`)
- [ ] Replace every `RELATED_WORK.md`/`DISSERTATION_AUDIT`/`analysis §` reference with a primary source or appendix pointer
- [ ] Move notebook-cell (N1–N5), CLI, file-path, experiment-id references to the reproducibility appendix
- [ ] Resolve every `[CONFIRM]`/`[PLACEHOLDER]`/`[INSERT]`
- [ ] Table footnotes in academic style (the † Nova note)
- [ ] One citation style throughout (confirm the university's required style first)
- [ ] Reference list: every entry re-verified; every in-text citation in the list and vice versa
- [ ] Word-count check against the institutional requirement

## 7. Suggested schedule (adjust to your deadline)

| Session | Deliverable |
|---|---|
| 1 (today) | Ch3 §3.1–3.3 rewritten in your voice; supervisor email sent re scope |
| 2 | Ch3 §3.4–3.8; launch E4–E6 in parallel |
| 3 | Ch4 MuSiQue half rewritten; figures exported |
| 4 | Ch2 part 1 (foundations + orchestration families) from the citation base |
| 5 | Ch2 part 2 (benchmarks + evaluation gap); reference list built & verified |
| 6 | Ch4 MultiHop half (once E4–E6 done); Ch1 |
| 7 | Ch5; abstract; appendices; full voice-checklist pass |
| 8 | Buffer: citation verification sweep, formatting, proofread |
