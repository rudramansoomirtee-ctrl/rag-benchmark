# Appendix C — Per-Hop and Per-Question-Type Accuracy Breakdowns

> Appendix draft — content copied from `thesis/musique_matrix_analysis.md`; renumber/reformat at Word
> conversion.

## C.1 MuSiQue — headline containment accuracy (n=150 per cell)

Copied from `thesis/musique_matrix_analysis.md` §2.1:

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A (single-pass) | 0.487 | 0.473 | 0.273 |
| A-minus (dense-only) | 0.420 | 0.420 | 0.260 |
| **B (iterative)** | **0.513** | **0.527** | **0.347** |
| B-minus | 0.447 | 0.467 | 0.307 |
| F (parallel decomp.) | 0.453 | 0.440 | 0.300 |
| F-minus | 0.427 | 0.420 | 0.260 |
| F-seq (sequential decomp.) | 0.480 | 0.480 | 0.320 |
| F-seq-minus | 0.433 | 0.467 | 0.260 |

> Token-F1 (DeepSeek): A 0.444 · B 0.477 · F 0.416 · F-seq 0.463 (secondaries track the primary
> ordering). Nova F-family cells are **degraded** (decompose parse-fail → ≈ naive retrieval); report
> flagged †, not as decomposition.

## C.2 MuSiQue — accuracy by hop count (n=78 2-hop / 45 3-hop / 27 4-hop per system per model)

Copied from `thesis/musique_matrix_analysis.md` §2.2:

| System | DS 2h | DS 3h | DS 4h | Qw 2h | Qw 3h | Qw 4h | Nv 2h | Nv 3h | Nv 4h |
|---|---|---|---|---|---|---|---|---|---|
| A | .538 | .444 | .407 | .487 | .511 | .370 | .346 | .222 | .148 |
| A-minus | .462 | .489 | .185 | .410 | .422 | .444 | .308 | .244 | .148 |
| B | .577 | .533 | .296 | .564 | .489 | .481 | .410 | .289 | .259 |
| B-minus | .513 | .467 | .222 | .564 | .378 | .333 | .359 | .289 | .185 |
| F | .500 | .444 | .333 | .487 | .400 | .370 | .372 | .267 | .148 |
| F-minus | .474 | .489 | .185 | .436 | .400 | .407 | .282 | .267 | .185 |
| F-seq | .577 | .422 | .296 | .577 | .333 | .444 | .385 | .311 | .148 |
| F-seq-minus | .551 | .333 | .259 | .615 | .378 | .185 | .295 | .244 | .185 |

> Accuracy declines with hop depth everywhere; 4-hop cells (n=27) are noisy — do not over-interpret
> single 4-hop cells.

### C.2.1 Multi-hop orchestration by hop, exp38 pilot detail (n=50, hybrid, DeepSeek — superseded by C.2 above but retained for the F-seq mechanism narrative)

Copied from `DISSERTATION_AUDIT.md` §5c:

> F-seq is the strongest decomposition system on deep hops — 3-hop **0.600**, 4-hop **0.444** (vs F
> 0.467/0.111; up to 4× on 4-hop). B leads 2-hop (0.615). F-seq ties B overall (0.540) at ~55% of B's
> cost. (F-seq-vs-F = parallel-vs-sequential; F-seq-vs-B = pre-decomposed self-ask vs free-form
> iteration.)

This n=50 4-hop multiplier ("up to 4×") did **not replicate** at n=150 — see
`thesis/musique_matrix_analysis.md` §4.1 and Appendix E: "F-seq ≫ F on deep hops (4-hop 0.444 vs
0.111, 4×)" is marked **Not replicated. F-seq>F is directional-only; its edge concentrates at 2-hop;
3/4-hop are statistical ties.**

## C.3 MuSiQue — position sensitivity ("lost-in-the-context"), by deepest gold rank

Copied from `thesis/musique_matrix_analysis.md` §7.9 (pooled across all 8 systems × 3 models, complete-evidence runs only, n=1,791):

| Deepest gold rank | n | Accuracy |
|---|---|---|
| 1–5 | 726 | **0.727** |
| 6–10 | 534 | 0.582 |
| 11–15 | 307 | 0.472 |
| 16–20 | 224 | 0.469 |

> Even with **every** required gold chunk in the context, accuracy falls **26 points** as the deepest
> needed chunk moves from the top-5 into the bottom half of the 20-chunk window. Controlled for hop
> depth (2-hop only): 0.742 → 0.562 → 0.424 (16–20 bucket bounces to 0.563 at n=80 — noisy tail).
> Correct runs hold their deepest gold at mean rank 7.05 vs 9.49 for incorrect. First-gold rank shows
> the same pattern (0.436 top-5 vs ~0.21–0.26 below).

## C.4 MultiHop-RAG — headline containment accuracy (n=200 per cell)

Copied from `thesis/musique_matrix_analysis.md` §9.1:

| System | DeepSeek-V3 | Qwen3-32B | Nova Lite |
|---|---|---|---|
| A | 0.830 | 0.820 | 0.785 |
| A-minus | 0.670 | 0.685 | 0.675 |
| B | 0.825 | 0.845 | 0.755 |
| B-minus | 0.760 | 0.775 | 0.760 |
| **F** | **0.855** | **0.875** | 0.770 † |
| F-minus | 0.785 | 0.790 | 0.695 † |
| F-seq | 0.845 | 0.845 | 0.775 † |
| F-seq-minus | 0.715 | 0.725 | 0.710 † |

> † Nova F-family degraded again (decompose parse-fail; n_steps 1.45–1.57 vs 3.4–3.7 on
> DeepSeek/Qwen) — same robustness finding as MuSiQue, replicated on the second dataset. Token-F1
> tracks containment throughout (e.g. Qwen-F 0.870); Nova's token-F1 is depressed by verbosity
> (0.42–0.52) while containment holds.

## C.5 MultiHop-RAG — retriever effect localised by question type

Copied from `thesis/musique_matrix_analysis.md` §9.3:

> **Mechanism, localised by question type:** the hybrid advantage concentrates overwhelmingly in
> **comparison questions** — DeepSeek A 0.791 vs A-minus **0.373** (+0.42; Qwen +0.36, Nova +0.28) —
> which name publishers ("the TechCrunch article…"), giving BM25 exact lexical anchors dense
> embeddings blur. Inference-type shows near-zero retriever effect (0.938 vs 0.938 on DeepSeek). This
> is the cleanest mechanistic evidence in the study that the pipeline's value tracks the *lexical
> anchorability* of the query distribution.
>
> **Null questions: no over-answering found.** All systems score 0.833–0.958 on nulls; B is *better*
> than A on nulls (0.958 vs 0.917 on DeepSeek) — the over-answering concern from the GraphRAG
> literature does not materialise for these orchestrations under refusal-equivalence scoring.

The MultiHop-RAG sample is stratified 64 inference / 67 comparison / 45 temporal / 24 null (seed 42,
n=200 per model; see `thesis/musique_matrix_analysis.md` header for Part II).

## C.6 Cross-dataset synthesis table (ties the per-hop and per-type breakdowns together)

Copied from `thesis/musique_matrix_analysis.md` §10:

| | MuSiQue (adversarial, sequential hops) | MultiHop-RAG (news, lexically anchored) |
|---|---|---|
| Best orchestration | **B** (rank-1 all models; B>A pooled p=.019) | **F** (best on both capable models; F>A p=.011, F>B p=.049) |
| F vs A | F ≤ A (novel negative) | **F > A, significant** (replicates Ammann's direction) |
| B vs A | +2.6 to +7.4 pts | ≈ 0 (p=.80) |
| F-seq vs F | F-seq ahead (directional) | F ahead (directional) |
| Hybrid vs dense | small (+3–7 pts/cell), pooled-only p=5.5×10⁻⁵ | large (up to +16 pts), **per-model significant**, overall p=3×10⁻²⁴ |
| Retriever mechanism | reranker rescues BM25-mined distractor noise | BM25 exploits lexical anchors (comparison-type +42 pts) |
| Nova decomposition collapse | 85–87% parse-fail | replicated (n_steps ≈ 1.5) |
| Pareto | Qwen-B tops frontier | **Qwen-F dominates everything** |
| Absolute difficulty | hard (best 0.527) | easier (best 0.875) |

## C.7 Cost-effectiveness — the full 24-cell grid (MuSiQue) and the MultiHop frontier

Copied from `thesis/musique_matrix_analysis.md` §5 (MuSiQue, all 8 systems × 3 models, accuracy ·
$/correct, sorted by cost; ⁻ = dense-only twin):

| Cell | Acc | $/correct | | Cell | Acc | $/correct |
|---|---|---|---|---|---|---|
| Nova-A | .273 | $0.00083 | | Qwen-F-seq⁻ | .467 | $0.00228 |
| Nova-A⁻ | .260 | $0.00084 | | Qwen-F-seq | .480 | $0.00230 |
| Nova-F-seq | .320 | $0.00119 | | Nova-B | .347 | $0.00276 |
| Nova-F | .300 | $0.00119 | | Nova-B⁻ | .307 | $0.00303 |
| Qwen-A | .473 | $0.00128 | | Qwen-B | **.527** | $0.00416 |
| Nova-F⁻ | .260 | $0.00133 | | DS-A | .487 | $0.00425 |
| Qwen-A⁻ | .420 | $0.00140 | | DS-A⁻ | .420 | $0.00473 |
| Nova-F-seq⁻ | .260 | $0.00143 | | Qwen-B⁻ | .467 | $0.00483 |
| Qwen-F | .440 | $0.00170 | | DS-F | .453 | $0.00574 |
| Qwen-F⁻ | .420 | $0.00171 | | DS-F⁻ | .427 | $0.00588 |
| | | | | DS-F-seq | .480 | $0.00814 |
| | | | | DS-F-seq⁻ | .433 | $0.00872 |
| | | | | DS-B | .513 | $0.01710 |
| | | | | DS-B⁻ | .447 | $0.01927 |

> Pareto frontier (accuracy ↑ vs $/correct ↓): Nova-A → Nova-F-seq → Qwen-A → Qwen-F-seq → Qwen-B.
> Every DeepSeek cell is dominated. B costs ~4× A per correct answer within every model. Caveat:
> $/correct covers LLM generation only; the Cohere reranker is a separately-metered per-retrieval
> charge borne identically by all hybrid systems.

MultiHop-RAG (from §9.4 — only the frontier-relevant cells carry printed $/correct values): the
frontier collapses to **Nova-A (0.785, $0.00064/correct) → Qwen-F (0.875, $0.00147/correct)**;
Qwen-F dominates every other configuration, including every DeepSeek cell (DeepSeek-F: 0.855 at
$0.00557, 3.8× the cost for less accuracy) and Qwen-B (0.845 at $0.00384, 2.6× the cost for less
accuracy).

## C.8 System B termination behaviour (MuSiQue)

Copied from `thesis/musique_matrix_analysis.md` §7.1:

| Exp | System | Early-stop rate (n<5) | Early-stop acc | Forced-stop rate (n=5) | Forced-stop acc |
|---|---|---|---|---|---|
| DeepSeek | B | 20.7% | **0.839** | 79.3% | 0.429 |
| DeepSeek | B-minus | 16.0% | **0.958** | 84.0% | 0.349 |
| Qwen3 | B | 56.7% | **0.706** | 43.3% | 0.292 |
| Qwen3 | B-minus | 43.3% | **0.646** | 56.7% | 0.329 |
| Nova | B | 34.7% | **0.577** | 65.3% | 0.224 |
| Nova | B-minus | 29.3% | **0.545** | 70.7% | 0.208 |

> Early self-termination beats budget-forced termination by 20–61 accuracy points in all six cells.

## C.9 Post-hoc abstention policy (MuSiQue; zero new LLM calls)

Copied from `thesis/musique_matrix_analysis.md` §7.8. Policy: answer only when B self-terminates
(n_steps < 5); abstain when forced to the budget. Re-scored from stored runs:

| Exp | System | Coverage (answers) | Selective acc | Baseline acc | Correct kept | Wrong answers avoided |
|---|---|---|---|---|---|---|
| DeepSeek | B | 31/150 (21%) | **0.839** | 0.513 | 26/77 | 68 (45% of all queries) |
| DeepSeek | B-minus | 24/150 (16%) | **0.958** | 0.447 | 23/67 | 82 |
| **Qwen3** | **B** | **85/150 (57%)** | **0.706** | 0.527 | **60/79 (76%)** | **46** |
| Qwen3 | B-minus | 65/150 (43%) | 0.646 | 0.467 | 42/70 | 57 |
| Nova | B | 52/150 (35%) | 0.577 | 0.347 | 30/52 | 76 |
| Nova | B-minus | 44/150 (29%) | 0.545 | 0.307 | 24/46 | 84 |

> Selective accuracy rises +17 to +51 points over baseline in every cell. The escalation-to-A
> alternative (answer with single-pass A whenever B is forced) is *worse than plain B everywhere*
> (DeepSeek 0.487 vs 0.513; Qwen 0.520 vs 0.527; Nova 0.320 vs 0.347), because A performs even worse
> than forced-B on that subset (0.395 / 0.277 / 0.184 vs B's 0.429 / 0.292 / 0.224) — budget
> exhaustion flags queries that are intrinsically hard for every orchestration. The gold-in-context /
> generation-side failure-attribution table supporting §4.7's remaining claim is in Appendix D (§D.4).

## C.10 Secondary-metric agreement

Copied from `thesis/musique_matrix_analysis.md` (lines as noted): MuSiQue token-F1 under DeepSeek-V3
— A 0.444 · B 0.477 · F 0.416 · F-seq 0.463 — tracks the containment ordering (§2). On MultiHop-RAG,
token-F1 tracks containment throughout (e.g. Qwen-F 0.870); Nova Lite's token-F1 is depressed by
verbosity (0.42–0.52) while its containment holds (§9.1). Exact match is zero across all systems on
MuSiQue. No per-cell token-F1 grid beyond these values is recorded in the source documents.

---

**Not found in sources:** a full MultiHop-RAG per-question-type × per-system numeric grid (i.e. all 8
systems × 3 models × 4 question types, analogous to the MuSiQue per-hop table in C.2) was not located
in `thesis/musique_matrix_analysis.md` — only the comparison-type and inference-type retriever-effect
figures (C.5) and the null-type summary range (0.833–0.958) are present. The full by-type breakdown
may exist in the live `/api/experiments/{id}/by-type` endpoint or notebook output but is not recorded
as a static table in the reviewed source documents. Likewise no full per-cell token-F1 grid exists
(C.10 carries what is recorded).
