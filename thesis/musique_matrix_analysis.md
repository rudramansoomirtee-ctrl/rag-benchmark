# Final Matrix — Comprehensive Analysis Report

**Part I (§1–§8): MuSiQue arm, E1–E3 (ids 50, 51, 53) · 3 models × 8 systems × 150 queries = 3,600 runs · 0 failures.**
**Part II (§9–§10): MultiHop-RAG arm, E4–E6 (ids 54, 56, 57) · 3 models × 8 systems × 200 queries = 4,800 runs ·
1 failure (0.02%) — completed 2026-07-11 at SHA `d03dd3b`. Matrix total: 9,600 runs, $24.59 LLM spend.**
**Frozen config:** git SHA `12f2a49` (E3: `ec457dc`, verified inert — resume-logic commit only), seed 42, `top_k=20`,
`fused_answer_top_k=20`, `retrieval_pool=40`, `max_agent_steps=5`, Cohere Rerank 3.5 (Bedrock, eu-central-1),
`BAAI/llm-embedder`, LiteLLM 1.83.0, temperature 0. LLM spend: **$7.05** (DeepSeek $5.14 / Qwen $1.39 / Nova $0.52);
Cohere rerank metered separately (not in `cost_usd`).
**Verification:** three independent audits (data-integrity; statistics, itself independently cross-checked; literature),
compiled 2026-07-05. Metrics table verified to match recomputation from raw runs exactly in all 24 cells.

**Systems.** Orchestration axis: **A** = Single-pass RAG · **B** = Iterative RAG (free-text route, ≤5 steps, evidence
accumulates via RRF) · **F** = Parallel decomposition (decompose → retrieve sub-questions concurrently → RRF-fuse) ·
**F-seq** = Sequential decomposition / Self-Ask (resolve each hop, substitute the bridge answer into the next hop's
retrieval). Retriever axis: hybrid BM25+dense+RRF+rerank (default) vs dense-kNN-only (the `-minus` twins) — a full
4×2 factorial. Dataset: MuSiQue answerable, pooled-distractor setting (each question's ~20 candidate paragraphs,
incl. BM25-mined hard distractors, pooled into one 3,000-chunk index). Sample: 150 seeded stratified queries
(78 2-hop / 45 3-hop / 27 4-hop), identical across every cell. Primary metric: alias-aware containment accuracy
(`answer_match`); secondary: SQuAD token-F1.

---

## Executive summary

**Verdict: the MuSiQue arm is clean, verified, and internally consistent — findings are correct with the phrasing
qualified below.** Data integrity passes all 7 checks; every reported number traces to the raw runs.

**What holds (confirmed):** (1) iterative retrieval **B is the best orchestration** — rank-1 in all 3 models, B>A
pooled p=.019; (2) **hybrid > dense-only** — directional in all 24 cells, pooled p=5.5×10⁻⁵ (small per-cell effect,
so report pooled, not per-cell); (5) **generation, not retrieval, is the bottleneck** — gold is in context 93–99% of
the time, 87–99% of errors occur *with* the answer present, zero lucky guesses; (7) **rankings are stable** across the
capability gradient (Kendall τ 0.64–0.74); (8) **the expensive model is never rational** — every DeepSeek cell is
Pareto-dominated, Qwen-B beats DeepSeek-B on accuracy at 24% of cost; (10) **the agent knows when it's done** —
self-terminated answers are 20–61 pts more accurate than budget-forced ones, which (§7.8) yields a free
abstention policy (+17–51 pts selective accuracy; Qwen the deployable sweet spot); (12) **iteration is depth-adaptive**.

**What is weaker than the pilots suggested:** (3) **parallel decomposition F does NOT beat single-pass A** — refuted as
an improvement; a *novel* result, in tension with Ammann 2025 (different dataset — MultiHop is the reconciling test);
(4) **F-seq > F is directional-only** (p=.203), though its bridge mechanism is real (complete-gold assembly +15 pts on
deep hops) — the generator squanders it (§7.9 shows why: accuracy falls 26 pts as needed evidence sits deeper in
context). (11) compensatory search is directional (significant only on Qwen).

**The correction that must propagate:** the n=50 pilot's "dense wins MuSiQue / B-minus champion" headline is **refuted**
at n=150 — reframe it as a *methods finding* about small-sample RAG unreliability (§4.1; feeds RQ4). Stale claims still
live in DISSERTATION_AUDIT §5c, Chapter 3/4 pilot boxes, RELATED_WORK §8.

**Status:** COMPLETE — both arms. MuSiQue (E1–E3, Part I §1–§8) and MultiHop-RAG (E4–E6, Part II §9–§10). The
Study-2 dataset contrast is resolved, and it is the study's central result: **both the orchestration ranking and
the retriever effect flip by dataset, consistently across models** — F (parallel decomposition) is the best system
on MultiHop and significantly beats both A and B pooled over the capable models, while it fails against A on
MuSiQue; the hybrid retriever's advantage is per-cell significant on MultiHop (p ≤ 5×10⁻⁵ in every model) but
pooled-only on MuSiQue. The Ammann tension is reconciled as dataset-dependence (§9.2). Numbers are externally
plausible (above the open-domain GPT-3.5 band for explainable reasons; Nova recovers the published weak-model band
— an internal control). See §8 for the write-up mapping and ⚠ cite-checks; §10 for the cross-dataset synthesis.

---

## 1. Data integrity audit — PASS on all seven checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Sample identity | **PASS** | Identical 150 `query_ids` byte-for-byte across E1/E2/E3 and across all 8 systems within each (all pairwise set-differences = 0) |
| 2 | Provenance constancy | **PASS** | All config identical except `model`. E3's SHA drift (`12f2a49`→`ec457dc`) diff-verified: only the `--resume-id` CLI flag + a pre-billing skip; `_run_one` (execution/scoring) moved verbatim, unchanged |
| 3 | Completeness | **PASS** | 1,200 rows each; 150/system; 0 NULL answers; 0 duplicate (system, query) pairs |
| 4 | Stratification | **PASS** | Exactly 78 / 45 / 27 by hop count |
| 5 | Scoring sanity | **PASS** | Spot-checks: alias matching and date-range normalisation working; incorrect rows are genuine misses (wrong entity, hallucination, refusal); no false positives/negatives found |
| 6 | Nova degradation evidence | **PASS** | F-family `n_steps` = 1.35–1.41 on Nova vs 3.54–3.68 on DeepSeek/Qwen; B ≈ 4 everywhere |
| 7 | Contamination | **PASS** | Stale exp 44 (`final-musique-nova-full`, old `top_k=10` config) identified — **must be excluded from all analysis**. Exp 54 = clean resumable MultiHop partial (515 rows). Nothing else conflicts |

---

## 2. Headline results

### 2.1 Containment accuracy (n=150 per cell)

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

Token-F1 (DeepSeek): A 0.444 · B 0.477 · F 0.416 · F-seq 0.463 (secondaries track the primary ordering).
Nova F-family cells are **degraded** (decompose parse-fail → ≈ naive retrieval); report flagged †, not as decomposition.

### 2.2 Accuracy by hop count

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

Accuracy declines with hop depth everywhere; 4-hop cells (n=27) are noisy — do not over-interpret single 4-hop cells.

---

## 3. Statistical verification (paired exact sign / McNemar tests; identical query sets ⇒ all contrasts paired)

### 3.1 Retriever effect — hybrid vs dense-only

Per-pair discordant counts (b = hybrid-only-right, c = dense-only-right):

| Model | A vs A⁻ | B vs B⁻ | F vs F⁻ | F-seq vs F-seq⁻ |
|---|---|---|---|---|
| DeepSeek | 19:9 (p=.087) | 23:13 (p=.133) | 15:11 (p=.557) | 19:12 (p=.281) |
| Qwen3 | 18:10 (p=.185) | 24:15 (p=.200) | 14:11 (p=.690) | 17:15 (p=.860) |
| Nova | 13:11 (p=.839) | 16:10 (p=.327) | 16:10 (p=.327) | 18:9 (p=.122) |

- **12/12 pairs favour hybrid; zero reversals; 0/12 individually significant.**
- Pooled per model: DeepSeek b=76,c=45 (**p=.0062**) · Nova b=63,c=40 (**p=.0297**) · Qwen b=73,c=51 (p=.0589, marginal).
- **Pooled overall: b=212, c=136, p = 5.45×10⁻⁵ — highly significant.**
- Required phrasing: *directionally universal, significant pooled overall and in 2 of 3 models; the per-cell effect
  (~3–7 pts) is real but small relative to per-cell power — never imply per-cell significance.*

### 3.2 Orchestration contrasts

| Contrast | DeepSeek | Qwen3 | Nova | Pooled |
|---|---|---|---|---|
| B vs A | 18:14 (p=.60) | 22:14 (p=.24) | 16:5 (**p=.027**) | 56:33 (**p=.019**) |
| F vs A | 7:12 (A ahead) | 12:17 (A ahead) | 4:0 | 23:29 (p=.49; A ahead) |
| F-seq vs F | 21:17 | 26:20 | 4:1 | 51:38 (p=.203) |
| F-seq vs B | 14:19 (B ahead) | 16:23 (B ahead) | 10:14 (B ahead) | 40:56 (p=.125; B ahead) |

### 3.3 Rank stability (RQ3/RQ4)

| System | DS rank | Qwen rank | Nova rank |
|---|---|---|---|
| B | 1 | 1 | 1 |
| A | 2 | 3 | 5 |
| F-seq | 3 | 2 | 2 |
| F | 4 | 6 | 4 |
| B-minus | 5 | 4= | 3 |
| F-seq-minus | 6 | 4= | 6= |
| F-minus | 7 | 7= | 6= |
| A-minus | 8 | 7= | 6= |

Kendall τ-b: DS↔Qwen **0.741** (p=.012) · DS↔Nova **0.643** (p=.030) · Qwen↔Nova **0.706** (p=.020) — all significant.
B is rank-1 in all three models; F-seq is the best decomposition variant in all three; `-minus` twins cluster at the bottom.

### 3.4 Failure attribution (E1/DeepSeek; coverage = ≥1 gold chunk in the top-20 answering context)

| System | Coverage | ALL gold present | Gen-failures (gold present, wrong) | Ret-failures (no gold) | Lucky guesses |
|---|---|---|---|---|---|
| A | 96.7% | 73/150 | 72 | 5 | **0** |
| B | 98.7% | 88 | 71 | 2 | **0** |
| F | 97.3% | 75 | 78 | 4 | **0** |
| F-seq | 99.3% | **100** | 77 | 1 | **0** |
| (minus twins) | 93–99% | 63–74 | 73–85 | 1–10 | **0** |

- **Errors are 87–99% generation-side**; retrieval-failure explains only 1–10 of ~75 errors per system.
- **Zero correct answers without gold in context** across all 1,200 runs — no parametric leakage; correctness genuinely
  required retrieval.
- **F-seq assembles complete gold evidence for 100/150 queries vs F's 75** — the sequential-bridge mechanism
  demonstrably improves full-evidence assembly even where the accuracy gain is directional-only (§4, Finding 4).
- Note: `recall_at_5` (0.37–0.59) uses a top-5 window; coverage uses the full top-20 context — different denominators,
  both correct.

### 3.5 Anomaly scan

- 0 NULL answers (3,600/3,600); 0 cost outliers (>5× cell median); latency p95 ≤ 11.8 s (structural: B/F-seq multi-call).
- `max_tokens=800` cap held for every single-call system (A max 582). **3–4 Nova rows exceeded the cap on single calls**
  (max 2,340 tokens; qids 17845/17853/17868/17873) — LiteLLM likely dropped the param for Nova's API shape
  (`drop_params=True`). Benign (Nova near-free, correctness unaffected); footnote in the write-up.

---

## 4. Findings (graded)

| # | Finding | Verdict | Key evidence |
|---|---|---|---|
| 1 | **Iterative retrieval (B) is the best orchestration** | ✅ Confirmed | Rank-1 all 3 models; B>A pooled p=.019 (Nova individually p=.027); literature-consistent |
| 2 | **Hybrid > dense-only** | ✅ Confirmed (pooled) | 24/24 directional, pooled p=5.5×10⁻⁵; small per-cell effect; Qwen marginal (p=.059) |
| 3 | **Parallel decomposition (F) does not beat single-pass (A)** | ❌ F-as-improvement **refuted** → **novel finding** | Pooled slightly favours A; no published precedent either way; explicit tension with Ammann 2025 (MultiHop-RAG) — E4–E6 are the reconciling test |
| 4 | **Sequential > parallel decomposition (F-seq > F)** | ⚠ Directional-only (p=.203) | b>c in all 3 models; mechanism proven independently: complete-evidence assembly 100 vs 75/150 — the generator squanders the assembled evidence |
| 5 | **Generation, not retrieval, is the bottleneck** | ✅ Confirmed | Coverage 93–99%; 87–99% of errors have gold in context; 0 lucky guesses |
| 6 | **Small models cannot execute decomposition** | ✅ Confirmed (measured) | Nova F-family n_steps 1.35–1.41 vs 3.5–3.7; iteration (B) still works on Nova — the model-robust strategy |
| 7 | **System ranking is stable across the capability gradient** | ✅ Confirmed | τ-b 0.64–0.74, all pairs significant |
| 8 | **The most expensive model is never the rational choice** | ✅ Confirmed | Every DeepSeek cell Pareto-dominated; Qwen-B > DeepSeek-B on accuracy at 24% of cost |
| 9 | **Absolute numbers are externally plausible** | ✅ Pass | §6 — above the open-domain GPT-3.5 band for explainable reasons; Nova recovers the weak-model band (internal control) |
| 10 | **The agent knows when it's done** | ✅ Confirmed (behavioural) | §7 — early-stop accuracy exceeds forced-stop accuracy in all 6 B cells by 20–61 pts; hitting the step budget is a hard-query signal |
| 11 | **Compensatory search under a weaker retriever** | ⚠ Directional (sig. on Qwen) | §7.2 — B-minus takes strictly more steps than B ~2–2.5× as often as fewer (Qwen p=.0013; DS/Nova p≈.06, tie-limited) |
| 12 | **Depth-adaptive iteration** | ✅ Confirmed | §7.3 — B's mean steps rise monotonically 2-hop→4-hop in all 3 models (+0.83 to +1.53) |

*Nuance to Finding 4:* on 3/4-hop queries specifically (n=72, DeepSeek), F-seq's complete-gold advantage holds
(52.8% vs F's 37.5%) but head-to-head answer wins do **not** (F-seq 9 vs F 11) — F-seq's overall directional edge
comes mostly from 2-hop; sequential bridge errors (UNKNOWN substitutions) partly offset the better evidence pool.

### 4.1 Correction of the n=50 pilot claims (must propagate to all documents)

| Pilot claim (n=50, DeepSeek, pre-final config) | n=150 verdict |
|---|---|
| "Dense-only wins on MuSiQue; B-minus (0.640) is the champion; the retriever effect is dataset-dependent and reverses on MuSiQue" | **REFUTED.** Hybrid wins all 24 cells, pooled p=5.5×10⁻⁵. The pilot reversal was small-sample noise compounded by config drift (`top_k` 10→20, prompt softening, `max_tokens` cap; the reranker did NOT change — pilots also used Cohere). |
| "F-seq ≫ F on deep hops (4-hop 0.444 vs 0.111, 4×)" | **Not replicated.** F-seq>F is directional-only; its edge concentrates at 2-hop; 3/4-hop are statistical ties. |

**Reframe for the thesis:** the pilot-vs-final reversal is itself a *finding* — a documented, quantified case study in
the unreliability of small-sample RAG evaluation (ties directly into RQ4 and the field-wide single-run critique).
Affected artefacts still carrying the stale claim: `DISSERTATION_AUDIT.md §5c`, Chapter 3/4 pilot boxes,
`RELATED_WORK.md §8` framing, session memory. **None of the refuted claims may survive as live claims.**

---

## 5. Cost-effectiveness (RQ2)

Full grid (accuracy · $/correct):

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

**Pareto frontier (accuracy ↑ vs $/correct ↓): Nova-A → Nova-F-seq → Qwen-A → Qwen-F-seq → Qwen-B.**
Every DeepSeek cell is dominated. B costs ~4× A per correct answer within every model. F-seq appears twice on the
frontier — the genuine cost/accuracy middle ground. Caveat: $/correct covers LLM generation only; Cohere rerank is a
separately-metered per-retrieval charge borne by hybrid systems only (disclosed in Ch3 §3.5).

---

## 6. Paper-vs-paper analysis

### 6.1 Comparison table (primary-source verified unless ⚠)

| Paper (arXiv) | Setting | Metric | Their number | Our closest | Comparable? |
|---|---|---|---|---|---|
| **IRCoT** — Trivedi et al., ACL'23 (2212.10509, T4) | Open-domain BM25, 139k paras, GPT-3 code-davinci-002 | answer F1 | NoR 25.2 → OneR 29.4 → **IRCoT 36.5** (Flan-T5-XXL: 13.7→25.8→30.8) | A 0.444 → B 0.477 (token-F1, DS) | **Partially** — same dataset+metric; our corpus ~46× smaller, reranked, stronger generators. Direction anchor (iterative>single), not level |
| **Adaptive-RAG** — Jeong et al., NAACL'24 (2403.14403, T8) | Open-domain BM25, same 139k corpus, GPT-3.5 | **containment Acc** (= our metric family) | No-retr 24.4 · single 23.6 · **multi 31.6** · adaptive 29.6 | A 0.487 → B 0.513 (DS) | **Partially — the best anchor**: same dataset, same metric family, same single-vs-multi contrast. Nova (0.273→0.347) lands inside their band |
| **Self-Ask** — Press et al. '23 (2210.03350, T1/T14) | Closed-book davinci-002, **2-hop subset only** | EM / Cover-EM | Direct 5.6 → CoT 12.6 → Self-Ask 13.8 EM (cEM 16.2) | F-seq 0.480 | **No** (subset, no retrieval) — cite as containment-metric precedent + sequential-decomposition lineage |
| **BeamAggR** — Chu et al., ACL'24 (2406.19820, T1) | Open-domain wiki + **web search**, GPT-3.5 | token F1 | Self-Ask 16.2 · IRCoT 24.9 · FLARE 31.9 · **BeamAggR 36.9** ⚠ (+8.5% headline unreconciled with T1) | F 0.416 / F-seq 0.463 | **Partially** — same dataset+metric; web augmentation + answer-level aggregation ≠ our retrieval-level fusion |
| **Search-R1** (2503.09516 **v5**, T2) | Open-domain E5/2018-wiki, RL-trained Qwen-7B | EM | 0.196 ⚠ (version drift v1→v5 confirmed: 0.142→0.196) | — | **No** — trained weights; context row for the SOTA frontier |
| **R1-Searcher** (2503.05592, T2) | Open-domain KILT-wiki 29M, RL 7B/8B | **Cover-EM** | 0.282 (baselines: naive GPT-4o-mini 0.134, IRCoT 0.192) | B 0.513 (DS) | **Metric-convention citation only** — radically harder corpus, trained weights |
| **Ammann et al.** '25 (2507.00355) | **MultiHop-RAG** (news), dense-only FAISS, Qwen2.5-32B, temp 0.8 | answer-F1 delta | decomposition **+11.6%** | our F−A ≤ 0 on MuSiQue | **The tension row** — different benchmark; our E4–E6 MultiHop cells are the reconciling evidence |
| **BEIR** — Thakur et al., NeurIPS'21 (2104.08663) | 18-dataset zero-shot IR | nDCG@10 | "BM25 is a robust baseline… dense … often underperform" (primary-verified) | hybrid>dense 24/24 | **Supports** retriever dataset-dependence; grounds Finding 2 |
| **MuSiQue** — Trivedi et al., TACL'22 (2108.00573) | dataset construction | — | distractors BM25-mined with intermediate answers masked (primary-verified) | — | Grounds the adversarial-to-lexical premise; NB we find hybrid wins **despite** this |

### 6.2 Plausibility assessment — PASS

- Our containment 0.45–0.53 / token-F1 0.42–0.48 sit **above every open-domain published number** (Adaptive-RAG
  containment 0.236–0.316; IRCoT F1 29–37; BeamAggR 36.9) and **below** trained-reader native-distractor regimes ⚠ —
  exactly where a pooled-distractor setting (3,000-chunk pool ≈ 46× smaller than IRCoT's corpus), a containment metric,
  and 2024/25-class generators should land. Being only ~1.6× above Adaptive-RAG with those advantages is conservative.
- **Nova Lite (0.273–0.347) recovers the published GPT-3.5-class band — a built-in external sanity anchor.**
- RL-trained 7B open-domain Cover-EM tops out ~0.28 (R1-Searcher); we exceed it in an easier setting — consistent.
- **Framing obligation:** these are plausibility-band comparisons, **never leaderboard claims** — our setting differs on
  corpus size (↓), metric leniency (↑), and generator strength (↑) simultaneously, all inflating absolutes.

### 6.3 Directional findings vs the literature

| Our finding | Literature status |
|---|---|
| Iterative > single-pass | **Strongly supported on MuSiQue specifically** (IRCoT +7.1 F1; Adaptive-RAG +8.0 containment; R1-Searcher baseline table). Our smaller margins (+2.6 to +7.4 pts) are expected: our single-pass baseline is far stronger (hybrid+rerank, 20-chunk context, small pool). Adaptive-RAG even records single-step retrieval *hurting* vs no-retrieval on GPT-3.5 — retrieval-noise sensitivity is on the record |
| Parallel decomposition ≯ single-pass | **No direct precedent found in either direction → novel.** Tension with Ammann 2025 is real but cross-dataset. BeamAggR's decomposition win uses answer-level aggregation, not retrieval-level fusion — does not contradict us. Mechanism (unresolved bridge entities make parallel sub-questions unretrievable) is precisely Self-Ask's design rationale |
| Hybrid > dense-only | **Consistent with BEIR.** The added nuance — hybrid wins *despite* MuSiQue's BM25-mined distractors — implies the cross-encoder reranker filters the lexical noise; ⚠ **no direct published evaluation-time evidence for this mechanism exists** (only indirect training-time hard-negative literature, e.g. 2206.08063). Present as this study's interpretation |

---

## 7. Agent behaviour analysis (the qualitative/behavioural arm)

*How the agentic systems behave, from the persisted trajectories (n_steps, evidence sets, answers) — the analysis
promised for the "study its behaviour" objective. Per-step route texts live in Phoenix traces; everything below is
recomputed from Postgres.*

### 7.1 System B termination behaviour — the agent knows when it's done

| Exp | System | Early-stop rate (n<5) | Early-stop acc | Forced-stop rate (n=5) | Forced-stop acc |
|---|---|---|---|---|---|
| DeepSeek | B | 20.7% | **0.839** | 79.3% | 0.429 |
| DeepSeek | B-minus | 16.0% | **0.958** | 84.0% | 0.349 |
| Qwen3 | B | 56.7% | **0.706** | 43.3% | 0.292 |
| Qwen3 | B-minus | 43.3% | **0.646** | 56.7% | 0.329 |
| Nova | B | 34.7% | **0.577** | 65.3% | 0.224 |
| Nova | B-minus | 29.3% | **0.545** | 70.7% | 0.208 |

Early self-termination (the route call choosing ANSWER) beats budget-forced termination by **20–61 accuracy points in
all six cells** — the ANSWER decision is a genuine confidence signal, and n_steps=5 is a hard-query flag, not a
slow-success flag. Forced-stop dominates on DeepSeek/Nova (65–84% of runs); only Qwen3 self-terminates the majority
of the time — its route model is the most decisive. Deployable heuristic: *budget exhaustion should trigger
abstention/escalation, not a confident answer.*

### 7.2 Compensatory search — the agent searches longer over a weaker retriever

Per-query paired comparison of steps (B-minus vs B): strictly-more 14:5 (DeepSeek, p=.064), **41:16 (Qwen, p=.0013)**,
24:12 (Nova, p=.065). Direction consistent everywhere (~2–2.5:1); most pairs tie at the 5-step ceiling, limiting power
outside Qwen. The same agent logic, given a weaker retriever, behaviourally compensates by iterating more.

### 7.3 Depth-adaptive iteration

Mean B steps by hop count: DeepSeek 4.06→4.56→4.89; Qwen 2.69→3.87→4.22; Nova 3.44→4.36→4.70 —
**monotonic in all three models**. The agent takes more retrieve→route cycles as question depth grows.

### 7.4 Evidence accumulation — a symptom of difficulty, not a cure

B roughly doubles its evidence pool through iteration (all_retrieved ≈ 35–40 unique chunks vs the final 20-slot
context); F/F-seq accumulate ~41–50 on DeepSeek/Qwen but collapse to ~23–24 on Nova (the decompose-failure footprint).
Critically, B's **incorrect** runs accumulate *more* evidence than its correct ones (42.9 vs 35.7, DeepSeek) —
accumulation and failure share the same cause (hard queries force more iterations); more searching marks difficulty
rather than producing success.

### 7.5 Decomposition behaviour — failure quantified per query

Empty/failed decomposition (n_steps=1): DeepSeek 3.3–6.0%, Qwen 0.0–0.7%, **Nova 84.7–87.3%** — the per-query
quantification of the Nova collapse (Finding 6). The decomposer never emits exactly one sub-question (0 or 2–4).
F-seq accuracy peaks at 2 sub-questions and declines at 3–4 — like B's forced-stop, deeper decomposition marks harder
queries more than it fixes them.

### 7.6 Case studies (DeepSeek, hybrid)

Iteration wins (B✓ A✗): **18**; reverse (A✓ B✗): **14** → net +4 for B (matches the paired test, §3.2).
- **id 17748** — *"What university did the author of 1967: The Last Good Year attend?"* A identified Pierre Berton but
  lacked the university chunk → refused. B (5 steps) retrieved the missing bridge fact → "University of British
  Columbia" ✓.
- **id 17777** — *"In which county is Mark Dismore's birthplace?"* B solved it in **2 steps** (early-stop win):
  retrieved Greenfield → re-queried → Hancock County ✓ while A refused. A clean recognize-gap→re-query→stop trajectory.
- **id 17747** (budget-exhausted failure) — *"Who is the spouse of the actor of Ethan in A Dog's Purpose?"* B locked
  onto the wrong Ethan-actor (KJ Apa, not Dennis Quaid) at step 1 and five iterations never corrected the initial
  entity-resolution error → refused; gold Meg Ryan. Iteration cannot fix a wrong first anchor.
- Honesty: B also *introduces* 14 new failures (query drift / needless reformulation of directly-answerable questions).

### 7.7 F-seq mechanism on deep hops (3/4-hop, n=72, DeepSeek)

Complete-gold assembly: **F-seq 52.8% vs F 37.5%** (+15 pts) — sequential bridge-resolution demonstrably improves
full-evidence retrieval on deep questions. But head-to-head answers: F-seq 9 wins vs F 11 — the retrieval-mechanism
advantage does not convert on deep hops (early bridge errors / UNKNOWN substitutions narrow the final prompt; F's
brute-force breadth sometimes wins anyway). F-seq's overall directional edge over F is driven by 2-hop questions.

### 7.8 Post-hoc abstention policy — exploiting the self-termination signal (zero new LLM calls)

Policy: *answer only when B self-terminates (n_steps<5); abstain when forced to the budget.* Re-scored from stored runs.

| Exp | System | Coverage (answers) | Selective acc | Baseline acc | Correct kept | Wrong answers avoided |
|---|---|---|---|---|---|---|
| DeepSeek | B | 31/150 (21%) | **0.839** | 0.513 | 26/77 | 68 (45% of all queries) |
| DeepSeek | B-minus | 24/150 (16%) | **0.958** | 0.447 | 23/67 | 82 |
| **Qwen3** | **B** | **85/150 (57%)** | **0.706** | 0.527 | **60/79 (76%)** | **46** |
| Qwen3 | B-minus | 65/150 (43%) | 0.646 | 0.467 | 42/70 | 57 |
| Nova | B | 52/150 (35%) | 0.577 | 0.347 | 30/52 | 76 |
| Nova | B-minus | 44/150 (29%) | 0.545 | 0.307 | 24/46 | 84 |

Selective accuracy rises **+17 to +51 points** over baseline in every cell. The trade is coverage: DeepSeek's
indecisive router abstains on 79% of queries (precision 0.84 but keeps only a third of its correct answers), while
**Qwen3 is the sweet spot** — it answers 57% of queries at 0.706 precision, keeping 76% of its correct answers while
filtering out 65% of its wrong ones (errors among given answers drop 71→25). For precision-critical deployments,
*iterative agent + abstain-on-budget-exhaustion* is a genuinely usable selective-QA policy, obtained for free from a
signal the agent already emits.

**Escalation-to-A refuted (honest negative).** The alternative policy — answer with the cheap single-pass A whenever B
is forced — is *worse than plain B everywhere* (DeepSeek 0.487 vs 0.513; Qwen 0.520 vs 0.527; Nova 0.320 vs 0.347),
because A performs even worse than forced-B on that subset (0.395 / 0.277 / 0.184 vs B's 0.429 / 0.292 / 0.224).
This confirms the interpretation of §7.1: **budget exhaustion flags queries that are intrinsically hard** — hard for
every orchestration — not B-specific failures; there is nothing cheaper to escalate *down* to. The right response to
the signal is abstention (or escalation *up* to a stronger model — untested, future work).

### 7.9 Position sensitivity — where the generation bottleneck bites ("lost-in-the-context")

*Pure SQL over stored contexts (JSONB arrays preserve rank order); pooled across all 8 systems × 3 models.*

**Accuracy by rank of the DEEPEST needed gold chunk, on complete-evidence runs only** (all gold present in the
top-20 context — so retrieval is fully controlled; n=1,791):

| Deepest gold rank | n | Accuracy |
|---|---|---|
| 1–5 | 726 | **0.727** |
| 6–10 | 534 | 0.582 |
| 11–15 | 307 | 0.472 |
| 16–20 | 224 | 0.469 |

Even with **every** required gold chunk in the context, accuracy falls **26 points** as the deepest needed chunk moves
from the top-5 into the bottom half of the 20-chunk window. Controlled for hop depth (2-hop only): 0.742 → 0.562 →
0.424 (16–20 bucket bounces to 0.563 at n=80 — noisy tail). Correct runs hold their deepest gold at mean rank 7.05 vs
9.49 for incorrect. First-gold rank shows the same pattern (0.436 top-5 vs ~0.21–0.26 below).

**Interpretation — the mechanism behind Finding 5:** generation failures concentrate when needed evidence sits deep in
the context, consistent with the *primacy* component of position-sensitivity findings (Liu et al. 2023,
"Lost in the Middle", arXiv:2307.03172 ⚠ cite-check: their canonical result is U-shaped over *forced* positions; ours
is an observational, monotone-declining variant over *ranker-assigned* positions). **Causal caveat:** rank is assigned
by the reranker/RRF, so a deep-ranked gold chunk also marks a harder query–evidence match — position and difficulty are
partially confounded. The randomized-position oracle run (gold chunks injected at controlled positions) is the causal
follow-up. Even as a correlation, the practical lever is real: anything that lifts gold higher in the final context
(better fusion weighting, smaller-but-cleaner contexts, position-aware ordering) attacks the dominant failure mode
directly.

## 8. Implications for the write-up

1. **Chapter 4 headline set** = Findings 1–12 (§4) plus the two behavioural sub-studies (§7.8 abstention, §7.9 position),
   with the statistical phrasing rules baked in (pooled tests for the retriever effect; "directional-only" for F-seq>F
   and compensatory search; F≯A framed as novel-with-mechanistic-support).
2. **The pilot reversal becomes a methods contribution** (§4.1): quantified evidence that n=50 RAG comparisons mislead
   (feeds RQ4 + the single-run field critique in RELATED_WORK §5). Purge the refuted claim from DISSERTATION_AUDIT §5c,
   the Ch3/Ch4 pilot boxes, and RELATED_WORK §8 before submission.
3. **The generation-bottleneck result** (Finding 5) reframes the whole benchmark: at top-20 context with a reranked
   pool, retrieval is nearly solved on MuSiQue-pooled; the residual problem is reading/synthesis. This unifies why
   orchestration gains are modest, why F-seq's evidence-assembly win doesn't convert, and — via §7.9 — *where* it fails
   (deep-in-context evidence). Present §5, §7.9, and Finding 4's non-conversion as one chained argument.
4. **The agentic-behaviour arc** (§7 + §7.8): the agent's free self-termination signal is a valid confidence estimator,
   exploitable as a zero-cost selective-QA policy — a self-contained, deployable, novel contribution. Pair it with the
   honest escalate-to-A negative (budget exhaustion = intrinsic query difficulty).
5. **RQ2 story**: Qwen3-32B + iteration is the rational configuration; frontier-cost models are dominated (§5).
6. **E4–E6 (MultiHop) remain decisive** for: the Ammann tension (does decomposition pay on news?), the dataset-contrast
   arm of Study 2, and cross-dataset rank stability. Everything above is analysis of data already in hand; MultiHop is
   the one outstanding *run*.
7. **Optional causal backfill** (~$0.50, held in reserve): the randomized-position oracle run would convert §5/§7.9 from
   correlational to causal and give a Tang & Yang-comparable gold-evidence ceiling — only if an examiner presses.

### Pre-print verification checklist (⚠ items)
- IRCoT full-MuSiQue EM (not reported; only 2-hop subset) — do not cite a full-EM figure.
- Flan-T5 base/large/XL MuSiQue values exist only as a plot (Fig. 9).
- BeamAggR "+8.5%" headline vs Table 1 discrepancy — cite as their quoted claim only.
- Search-R1: cite the v5/camera-ready numbers; version drift confirmed.
- Zhou et al. (least-to-most) sequential-beats-parallel specifics — not quote-verified.
- Native-distractor MuSiQue ceilings — not re-verified.
- Reranker-robustness-to-BM25-distractors — no direct evidence exists; frame as interpretation.
- Nova single-call `max_tokens` overshoot (3–4 rows) — one-line footnote.

---

# Part II — MultiHop-RAG arm (E4–E6) and cross-dataset synthesis

**Completed 2026-07-11 at SHA `d03dd3b` (trace-feature WIP stashed during the run to preserve the frozen-SHA
guarantee).** Experiments: 54 (DeepSeek-V3), 56 (Qwen3-32B), 57 (Nova Lite); 200 seeded stratified queries
(64 inference / 67 comparison / 45 temporal / 24 null; seed 42), identical across every cell; 4,800 runs;
**1 failure** (exp 57, System A, qid 175 — clean NULL stub, scored wrong per the crash-is-wrong policy; 0.02%).
Costs: DeepSeek $12.79 / Qwen $3.16 / Nova $1.55. E4 resumed from a 515-row partial via `--resume-id` with zero
re-billing (the resumability fix working as designed).

## 9. MultiHop-RAG results

### 9.1 Accuracy grid (containment, n=200 per cell)

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

† Nova F-family degraded again (decompose parse-fail; n_steps 1.45–1.57 vs 3.4–3.5 on DeepSeek/Qwen) — same
robustness finding as MuSiQue, replicated on the second dataset. Token-F1 tracks containment throughout
(e.g. Qwen-F 0.870); Nova's token-F1 is depressed by verbosity (0.42–0.52) while containment holds.

### 9.2 Orchestration on MultiHop — the ranking INVERTS relative to MuSiQue

Paired discordant counts (b:c) and exact two-sided binomial p:

| Contrast | DeepSeek | Qwen3 | Nova | Pooled |
|---|---|---|---|---|
| F vs A | 11:6 (p=.33) | 15:4 (**p=.019**) | 0:4 † | DS+Qwen 26:10 (**p=.011**) |
| B vs A | 8:9 (p=1.0) | 14:9 (p=.41) | 6:13 (p=.17, A ahead) | 28:31 (p=.80) — **B does not beat A** |
| F vs B | 11:5 | 11:5 | 11:8 | 33:18 (**p=.049**) — **F beats B** |
| F-seq vs F | 11:13 | 10:16 | 3:2 | 24:31 (p=.42, F ahead) |

- **Parallel decomposition (F) is the best system on MultiHop** — significantly better than A pooled over the
  two capable models (p=.011; individually significant on Qwen) and significantly better than B pooled (p=.049) —
  at roughly a third of B's cost. This **replicates Ammann et al. (2025)'s direction on their benchmark** and,
  combined with Part I (F ≤ A on MuSiQue), **resolves the tension as dataset-dependence**: parallel
  decomposition's blind-bridge weakness is fatal on MuSiQue's sequentially-dependent hops but irrelevant on
  MultiHop's more independent evidence sets, where the extra sub-question retrievals are pure coverage gain.
- **Iteration (B) does not pay on MultiHop** (pooled p=.80; directionally *behind* A on DeepSeek and Nova) —
  the mirror image of MuSiQue where B was rank-1 everywhere. Iteration earns its cost only where hops are
  genuinely sequential.
- **F-seq ≤ F on MultiHop** (direction reverses from MuSiQue): sequential resolution adds latency/cost and a
  bridge-error surface without benefit when hops don't depend on each other.

### 9.3 Retriever effect on MultiHop — larger, and per-cell significant

Hybrid − dense deltas: A +0.160/+0.135/+0.110 · B +0.065/+0.070/−0.005 · F +0.070/+0.085/+0.075 ·
F-seq +0.130/+0.120/+0.065 (DS/Qwen/Nova). 11 of 12 cells favour hybrid (single tiny exception: Nova B −0.005).

Pooled sign tests per model: DeepSeek 115:30 (**p=6×10⁻¹³**) · Qwen 129:47 (**p=5×10⁻¹⁰**) · Nova 99:49
(**p=5×10⁻⁵**) · overall 343:126 (**p=3×10⁻²⁴**). Where MuSiQue needed pooling across the whole matrix to
secure a small effect, MultiHop's effect is **individually significant within every model** — the
dataset-dependence of the retrieval pipeline, now measured on both sides.

**Mechanism, localised by question type:** the hybrid advantage concentrates overwhelmingly in **comparison
questions** — DeepSeek A 0.791 vs A-minus **0.373** (+0.42; Qwen +0.36, Nova +0.28) — which name publishers
("the TechCrunch article…"), giving BM25 exact lexical anchors dense embeddings blur. Inference-type shows
near-zero retriever effect (0.938 vs 0.938 on DeepSeek). This is the cleanest mechanistic evidence in the study
that the pipeline's value tracks the *lexical anchorability* of the query distribution.

**Null questions: no over-answering found.** All systems score 0.833–0.958 on nulls; B is *better* than A on
nulls (0.958 vs 0.917 on DeepSeek) — the over-answering concern from the GraphRAG literature does not
materialise for these orchestrations under refusal-equivalence scoring.

### 9.4 Cost on MultiHop

Pareto frontier collapses to two points: **Nova-A (0.785, $0.00064/correct) → Qwen-F (0.875, $0.00147/correct)**.
Qwen-F dominates *everything* else — including every DeepSeek cell (DeepSeek-F: 0.855 at $0.00557, 3.8× the
cost for less accuracy) and Qwen-B (0.845 at $0.00384, 2.6× the cost for less accuracy). The RQ2 conclusion
sharpens: on MultiHop, the rational configuration is the mid-tier model with parallel decomposition — the
cheapest multi-query orchestration — and neither iteration nor the strongest model earns its premium.

## 10. Cross-dataset synthesis — the study's central result

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

**Within-dataset, across models: stable** (MuSiQue τ-b .64–.74 all significant; MultiHop preserves F/F-seq/A/B
top-4 on both capable models). **Across datasets: the rankings invert.** The refined RQ3/RQ4 answer: *an
orchestration choice transfers across cost-efficient models but NOT across datasets* — benchmark choice, not
model choice, is the decision-relevant variable. Combined deployable guidance: match orchestration to hop
structure (iterative for sequentially-dependent questions; parallel decomposition for independent multi-evidence
questions) and match the retriever to the corpus's lexical anchorability — then choose the mid-tier model,
which owned every Pareto frontier in the study.

**Write-up impact (supersedes the corresponding §8 lines):** Ch4 §4.8 is now writable from §9; Ch4 §4.2/§4.3's
`[MultiHop]` markers resolve to: F≯A is *MuSiQue-specific* (not a general negative), the Ammann tension is
resolved as dataset-dependence, and the retriever finding upgrades from "pooled-only" to "per-model significant
on news." Ch5 per-RQ answers: RQ1 dataset-conditional; RQ2 mid-tier dominates both frontiers; RQ3 stable across
models within dataset; RQ4 rankings not portable across datasets + Nova robustness replicated. One integrity
audit item for the appendix: E4–E6 ran at SHA `d03dd3b` (Part I at `12f2a49`/`ec457dc`); the intervening commits
touched thesis prose and the resume/billing fix only — no retrieval, scoring, or generation semantics — and the
budget/prompt/config snapshot is field-identical across all six experiments (verified in `config_json`).

---

*Compiled from: integrity audit (27 checks), statistical verification (independently cross-checked; exact binomial
sign tests, Kendall τ-b, failure attribution over 3,600 runs), and primary-source literature verification (arXiv
PDF/HTML extraction). All numbers trace to Postgres experiments 50/51/53 at the frozen SHAs above.*
