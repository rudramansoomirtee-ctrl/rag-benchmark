# Dissertation Audit — Aims/Objectives vs Implementation

Audit of the dissertation slides (Gaps, RQ1–4, Aims A1–A4, Objectives O1–O6)
against this codebase, plus the action register that falls out of it.

**How to use this file:** read it INSTEAD of re-auditing the codebase. Update
statuses in place when an action is done (`[ ]` → `[x]`, verdicts ✅/⚠️/❌).
Evidence is given as `file:symbol` references. Audited at commit `59ccec3`
(2026-06-11).

**Companion:** `RELATED_WORK.md` — verified literature comparison (MultiHop-RAG
papers, orchestration/index/trained method classification, metrics catalogue
mapped to this repo, citation cautions). Read it INSTEAD of re-searching the
literature.

Legend: ✅ on point · ⚠️ partial (wording fix or pending run) · ❌ missing/wrong.

> **⚠️ Design has evolved since the original audit (§§1–4, 2026-06-11).** Those sections audit the
> *original* slide design (a 4×3 A/B/F/F-tuned matrix, MultiHop-only, HHEM faithfulness). **§5c is
> authoritative for the current design:** a 4×2 retrieval×orchestration factorial (A/B/F/F-seq × hybrid/
> dense-only = 8 systems) over MuSiQue + MultiHop on Qwen3-32B / DeepSeek-V3 / Nova Lite. Two scope
> changes not yet reflected in §§1–4: **F-tuned → F-seq** (+ the A/B/F/F-seq "-minus" twins), and
> **faithfulness/HHEM is descoped** (subsystem removed; columns unpopulated; future work). Read §§1–4 as
> historical; trust §5c + the thesis chapters for the as-built design.

---

## 1. Slide ↔ repo mapping

| Slide term | Repo reality |
|---|---|
| S1 | System A — naive (`systems/system_a.py`) |
| S2 | System B — LangGraph iterative agent (`systems/system_b.py`) |
| S3 | System F — query decomposition (`systems/system_f.py`) |
| S4 | ~~System F-tuned~~ **removed → replaced by F-seq** (sequential self-ask, `systems/system_fseq.py`). See §5c. F-tuned historical runs (exp18-26) remain in DB |
| (new) | **A-minus / B-minus** — A and B over a semantic-kNN-only retriever (`system_a_minus.py`, `SystemB(semantic_only=True)`); the retrieval-pipeline ablation axis. See §5c |
| "3 LLMs / model tiers" | Per repo evidence: Haiku 4.5, Nova Lite, Qwen3, **DeepSeek-V3** (§5c program) — **CONFIRM final matrix models** |
| "1 fixed retriever" | `retrieval/retrieve.py:retrieve` — hybrid BM25+dense+RRF + cross-encoder rerank, shared by A/B/F/F-seq; A-minus/B-minus deliberately use dense-kNN only (`semantic_only=True`) for the retrieval ablation |

---

## 2. Aims checklist

| Aim | Claim | Verdict | Evidence / gap | Actions |
|---|---|---|---|---|
| A1 | Controlled platform: 4 strategies on one shared retrieval substrate | ✅ | Shared `retrieve()`; A/B/F share `ANSWER_SYSTEM_PROMPT`; `SYSTEM_REGISTRY` = exactly A,B,F,F-tuned. Caveats: F-tuned also varies prompt + context budget; environment provenance not recorded per experiment | W7, C2 |
| A2 | 12 runs (4×3), accuracy + Token F1, stratified incl. null | ⚠️ | Seeded stratified `--sample` records `query_ids` (`runner.py`) ✅; accuracy ✅; **Token F1 not implemented anywhere** ❌; matrix not yet run on a frozen pipeline | C1, P1, P2, W5 |
| A3 | Cost:quality — per-query cost, cost-per-correct, Pareto frontier | ✅* | `cost_usd` per call + `cost_per_token` fallback; `metrics.cost_per_correct`; notebook Pareto scatter. *Slide says "LiteLLM **proxy**" — repo deliberately uses the **SDK** (CLAUDE.md decision list). Frontier-per-accuracy-level derivation pending | W2, N5, C7, C6 |
| A4 | Metric divergence audit + retrieval ceiling | ❌ | **Only aim with zero implementation.** `compute-metrics` prints contains/exact/CRAG side by side but no agreement/correlation; no ceiling / failure attribution; Token F1 missing; System B drops `all_retrieved_ids` before persisting | C1, C3, N3, N4 |

## 3. Objectives checklist

| Obj | Claim | Verdict | Notes | Actions |
|---|---|---|---|---|
| O1 | Sophisticated retrieval across all four pipelines | ✅ | Hybrid BM25+dense+RRF+rerank, single entry point. Reword "a sophisticated retrieval" → "a shared hybrid retrieval substrate" | — |
| O2 | S1–S4 "in a single LangGraph framework" + full cost tracing | ⚠️ | Cost tracing ✅ (tokens + `cost_usd` per run, Phoenix spans). **LangGraph claim wrong: only System B uses LangGraph**; A/F/F-tuned are plain Python on the `System` protocol | W1 |
| O3 | Evaluate all 12 system–model combos on stratified sample | ⚠️ | Capability ✅; not yet executed as a frozen matrix. Slide omits **null** question type (dataset + stratification + refusal-equivalence scoring all include it). Grammar: "a … sample sets", duplicated "stratified" | P1, P2, W4 |
| O4 | Quantify quality via contains-match + Token F1 across types | ⚠️ | contains ✅ but **adapted** (post-marker, refusal-equivalence, entity-suffix layers in `evaluation/metrics.py:contains_match` — must be disclosed); by-type ✅ (`metrics-by-type`, `/by-type`); **Token F1 ❌** | C1, W3 |
| O5 | Cost-per-correct per system–model pair + Pareto frontier | ✅* | Per-experiment ✅; cross-model frontier needs the 12 runs + frontier derivation | N5, P1 |
| O6 | Metric divergence + retrieval-ceiling attribution | ❌ | Nothing computes either; data is sufficient (`runs.retrieved_chunk_ids` × `queries.relevant_chunk_ids` × `is_correct`) | N3, N4, C3 |

## 4. RQ coverage

| RQ | Status | Notes |
|---|---|---|
| RQ1 accuracy by orchestration + type | ✅ now | `contains_match` + by-type breakdown |
| RQ2 cost per correct + latency | ✅ cost / ⚠️ latency | Latency captured per-run (`runs.latency_ms`) but **never aggregated** — no metrics column, no notebook cell. Cross-provider latency = serving-infra confound, scope the claim |
| RQ3 ranking across models | ⚠️ | Needs frozen 12-run matrix (same SHA/sample/index) + rank-stability stat. "Model strength" wording overreaches if all 3 models are budget-class |
| RQ4 predictability (rank stability + variance) | ❌ until N1/N2 + P1 | No rank-stability or variance computation exists yet |

Gap slide citations: Trivedi 2023 (IRCoT) + Ammann 2025 fit "cost as side issue".
Shi et al. 2024 ambiguous (REPLUG vs the 2023 distraction paper) — verify bib.
Asai (Self-RAG) is arXiv 2023 / ICLR 2024 — check year style. Gao 2024 + Barnett
2024 support "frontier-only evidence" only **indirectly** — soften or re-cite.
Verify the Ammann quote against the PDF before verbatim use (CLAUDE.md already warns).

---

## 5. Action register

### C — Code changes

**Before the final runs:**
- [x] **C2** *(done, commit pending)* Provenance now recorded in
  `experiments.config_json` (`evaluation/runner.py`): `git_sha` (via `_git_sha()`
  — reads `GIT_SHA` env, falls back to `git rev-parse`), `reranker_model`,
  `rerank_provider`, `retrieval_pool`, `hhem_threshold`, and a per-dataset
  `corpus` fingerprint (`_corpus_fingerprint()` → n_chunks, n_passage_chunks,
  granularity). **Action for final runs:** export `GIT_SHA=$(git rev-parse HEAD)`
  when invoking `run-experiment` inside the container (no `.git` is mounted there).
- [x] **C3** *(done — user: "C3 yes")* `RunResult.all_retrieved_chunk_ids` + the
  `runs.all_retrieved_chunk_ids` column (migration `0004`) now persist evidence
  ever seen: System B sets the deduped union across iterations, F-tuned the union
  of all per-query/source pools; A/F fall back to the final context in the runner
  (identical for them). N3 coverage uses it — verified it flips B's
  found-early-then-reformulated-away case from a false retrieval-failure to a
  correctly-attributed agent failure.
- [x] **C6** *(done — user: "(a)")* Policy is **a crash is a wrong answer**: failed
  (`answer IS NULL`) rows stay in the accuracy denominator, and the rate is now
  **visible** via `metrics.pct_failed` (migration `0004`) + a `Fail%` column in
  `compute-metrics` (no longer hidden). State the policy in the methodology.
- [x] **C7** *(resolved differently — user: "idk" the version)* Instead of a hard
  requirements pin (can't guess the installed version), each experiment now records
  `config_json.litellm_version` at runtime, so every cost number is attributable.
  **For the final runs:** build the image once and run all 12 on it (don't rebuild
  mid-matrix) → costs are internally consistent regardless. A hard pin is optional;
  read the version off any experiment if you ever want one.
- [ ] **C8** Verify index granularity purity: no mixed `<url>` and `<url>#p<i>`
  chunk IDs for multihop (ingest is additive; `index_corpus` indexes ALL chunks).
  If mixed → wipe Postgres chunks + OpenSearch index, re-ingest, re-index.
  *Now auto-flagged*: every new experiment's `config_json.corpus[ds].granularity`
  reads `"mixed"` when both granularities coexist (C2 fingerprint) — but acting on
  the flag (the wipe/re-ingest) is still manual.

**Post-hoc safe (answers stored; recomputable over historical runs):**
- [x] **C1** *(done, commit pending)* `evaluation/metrics.py:token_f1` — SQuAD-style
  token-overlap F1 over post-marker text, with refusal-equivalence so null-type is
  meaningful (entity-suffix stripping intentionally omitted to keep it a clean
  lexical metric). Persisted as `metrics.avg_token_f1` (migration `0003_token_f1`),
  surfaced in `compute-metrics`, `metrics-by-type`, and the `/metrics` + `/by-type`
  API routes. Recomputable over historical experiments via `compute-metrics`.
  **Action for the user:** run `alembic upgrade head` before the next `compute-metrics`.
  Unblocks slide wording W3.
- [x] **C4** *(done in notebook)* Latency avg/p50/p95 per system in N2
  (`variance_tbl`). A persisted `metrics` column was deemed unnecessary (notebook
  covers it; avoids schema bloat).
- [x] **C5** *(skipped — user: "fresh runs only")* No `rescore` command. Headline
  numbers will come from the fresh 4×3 matrix, so the mid-project metric drift on
  historical `is_correct` is moot. (N4's agreement analysis recomputes metrics from
  stored answers anyway, so it's unaffected.)

### N — Analysis (in `notebooks/analysis.py`)
All five added as marimo cells after the existing per-type cell, plus a shared
helper cell (imports the canonical `metrics.py` scoring so notebook numbers match
`compute-metrics`; defines `kendall_tau_b`, `bootstrap_ci`, `pareto_frontier`,
`covered`, `agreement`). Math + exact cell logic verified standalone on synthetic
data; the live versions need a populated Postgres. *(All done, commit pending.)*
- [x] **N1** Rank stability: `kendall_tau_b` (tie-aware, hand-rolled — no scipy dep)
  over per-system accuracy across experiments, labelled by model. Degrades to a
  "need ≥2 experiments" message until the matrix is run. (Found+fixed a diagonal
  duplicate-column bug here via the cell-logic test.)
- [x] **N2** Per-system accuracy bootstrap CI + latency p50/p95/mean/std + cost
  mean/std/total/per-correct (`variance_tbl`). Works on a single experiment.
- [x] **N3** Retrieval ceiling + failure attribution (`ceiling` + stacked-bar):
  coverage = fraction with ≥1 gold article retrieved (the ceiling); every error
  partitioned into `err_retrieval` (no evidence) vs `err_generation` (had evidence,
  still wrong) — verified `err_retrieval + err_generation == 1 − accuracy`.
  Null-type queries (no gold) excluded. NOTE: uses each run's persisted
  `retrieved_chunk_ids`, whose semantics differ per system (A top-5; B final-step
  top-5; F full fused; F-tuned 10) — so "coverage" is *evidence in the answering
  context*, not *ever seen* (that needs C3 for System B).
- [x] **N4** Metric agreement matrix + heatmap (`agree_mat`): pairwise agreement
  rate over contains / exact / tokenF1≥.5 / CRAG-good, recomputed under current
  scoring code (sidesteps stale stored `is_correct`, C5). CRAG cells use judged
  runs only.
- [x] **N5** Pareto frontier overlay (`fig_pareto`): marks non-dominated
  (accuracy, cost-per-correct) points over the scatter.

### P — Final-run protocol (procedure, no code)
- [ ] **P1** One wipe + re-ingest + re-index; freeze git SHA; run all 12
  (4 systems × 3 models) back-to-back on the same machine against the same index build.
- [ ] **P2** ONE fixed seeded stratified sample reused across all 12 runs
  (identical `query_ids` — verify via `config_json.selection` across experiments).
- [ ] **P3** Record hardware (CPU, RAM) — embedder/reranker/HHEM are CPU-bound and
  shape latency numbers.
- [ ] **P4** Judge pass post-hoc with a fixed `JUDGE_MODEL` independent of generation,
  same judge for all 12 experiments.

### W — Slides / dissertation wording (no code)
- [ ] **W1** "implement S1–S4 in a single LangGraph framework" → "in a single
  benchmark harness with a shared system interface" (only S2/B uses LangGraph).
- [ ] **W2** "via LiteLLM proxy" → "via the LiteLLM SDK" (no proxy container, by
  documented decision).
- [x] **W3** *(unblocked)* Token F1 now exists (C1), so the slides may keep it —
  but only report numbers once `alembic upgrade head` + `compute-metrics` have
  populated `avg_token_f1`. Until then the column is NULL.
- [ ] **W4** Objective 3: add the **null** question type; fix "stratified …
  stratified" and "a … sample sets".
- [ ] **W5** "4 × 3 experimental sample sets" → "12 runs over one fixed stratified
  sample" (plural implies different samples per cell → breaks comparability).
- [ ] **W6** RQ3 "model strength" → "across heterogeneous budget-class models", or
  add one stronger anchor model run to claim a strength axis.
- [ ] **W7** Present A/B/F as the controlled single-variable comparison; F-tuned as
  a stacked engineering system (it also changes prompt + context budget), so it
  doesn't undermine the Gap-1 confound claim.
- [ ] **W8** Citations *(partially resolved — see `RELATED_WORK.md` §6 for the
  verified evidence base)*: still open — pin down Shi et al. 2024 identity; check
  Asai year; soften/re-cite Gap 3 (now with the verified refinement: IRCoT ran
  Flan-T5-base→XXL and FlashRAG runs Llama3-8B, so phrase as "controlled
  orchestration comparisons on budget commercial API models with cost accounting
  are absent"). Verified by search: Ammann limitation quote is a **paraphrase**
  (check PDF before verbatim); GPT-4 0.56/0.89 needs Table 6 re-check;
  Multi-Meta-RAG 17.2% vs 18% version drift; Search-R1 % drifted across arXiv
  versions (cite COLM camera-ready). New citable support: Gap 1 — no MultiHop-RAG
  leaderboard exists; Gap 2 — 19 papers audited, only HippoRAG reports $ (appendix),
  none report cost-per-correct.
- [ ] **W10** Metric-definition disclosures in the methodology chapter (from
  `RELATED_WORK.md` §5): (a) `exact_match` does NOT strip articles a/an/the —
  slightly stricter than SQuAD EM; disclose or align with `token_f1`'s
  normalization; (b) the CRAG judge uses the **human-rubric** 4-way weights
  (1/0.5/0/−1), not the auto-eval 3-way merge — name the variant; (c)
  `contains_match` is **stricter** than the official MultiHop-RAG scorer
  (word-set intersection, verified in `qa_evaluate.py`) — a *defence*, cite it.
- [ ] **W9** Scope cross-provider latency claims (serving infrastructure confound;
  systems run sequentially so time-of-day load differs per system).

### D — Docs housekeeping
- [x] **D1** *(done, commit pending)* CLAUDE.md corrected: shared-calls table now
  shows the hybrid+rerank `retrieve()` pipeline (not plain `knn_search`) and that
  only the embedder *model* is lru_cached; System B section rewritten to the
  two-call ROUTE/execute machine (`RouteDecision`, not the unused `AgentDecision`);
  B1/B3/B5 noted as removed (incl. the dead `ksweep` CLI example); config block
  updated (`litellm_model`=Nova Lite, `hhem_threshold`=0.10, added
  `retrieval_pool`/`reranker_model`/`rerank_provider`); gap #5 reflects
  passage-by-default ingestion.
- [x] **D2** *(done, commit pending)* README corrected: the dead B1/B3/B5 sweep
  ablation replaced with the A/B/F/F-tuned lineup; "systems A–D" → A/B/F/F-tuned;
  implementation-status rows fixed (`hybrid_search` is implemented not a stub;
  System B two-call note; `metrics.py` row reflects contains-match-as-primary +
  token_f1/CRAG secondaries).
  *Not touched (out of D-scope, code docstrings):* `main.py` still says "Systems
  A/B/C/D" in the FastAPI `description=` — cosmetic, no doc impact.

---

## 5b. System changes after the audit (2026-06-11) — accuracy quick wins

Three approved improvements landed **before** the final 4×3 matrix (so the matrix
runs the improved systems; historical runs in the DB predate them):

1. **All systems answer over top-10** (`settings.top_k = 10`, `FUSED_ANSWER_TOP_K = 10`):
   A retrieves 10 in one shot; B fuses its iteration lists to top-10; F fuses its
   sub-question lists to top-10. F-tuned was already at 10 — unchanged.
   Single-hop F / B(1 step) still degenerate to A's exact 10-chunk context.
2. **F uses the few-shot decomposer** (`DECOMPOSE_FEWSHOT_PROMPT` now lives in
   `system_f.py`; F-tuned imports it). F also inherits F-tuned's graceful
   decompose-failure fallback (parse failure → no sub-questions, not a stub row).
3. **B accumulates evidence across iterations** (IRCoT-style): route/reformulate/
   answer all see `rrf_fuse(iteration_hits)[:10]` — the same fusion + budget as F —
   instead of only the latest batch. B(1 step) still equals A's context.

**Disclosure implications for the write-up:**
- The controlled comparison line becomes: *retriever, generator, and answer context
  budget (10 chunks) held constant across A/B/F. A selects its 10 in one retrieval
  pass; B and F select theirs by fusing multiple targeted retrievals.* B and F
  differ **only** in how the extra queries are produced (sequential conditioned
  reformulation vs parallel upfront decomposition) — the purest possible comparison.
- Metric semantics shifted for B and F (fresh runs only, consistent with the C5
  skip-historical decision): B's `retrieved_chunk_ids` (⇒ P@10/R@10, HHEM premise)
  is now the fused answering context, not the last iteration's batch; F's HHEM
  premise now equals its actual top-10 context instead of the whole fused list.
  Do **not** compare these columns against pre-change experiment rows.
- F-tuned's residual deltas vs F: per-query top-10 pools (same as F now) + weighted
  RRF + source fan-out + reserved slots + CoT prompt (decomposer no longer a delta).
- W7 framing unchanged: F-tuned stays the stacked engineering ceiling.

## 5c. Retrieval × orchestration program (2026-06-28) — F-seq, A-minus/B-minus, ablations

Run on **DeepSeek-V3** (`bedrock/deepseek.v3-v1:0`) over **MuSiQue** (pooled-distractor,
index `rag-chunks-musique`) and **MultiHop-RAG** (`rag-chunks`), on the standardized
improved pipeline. Comparisons use the SAME query set per dataset (MuSiQue: the 50
ids in exp36 `config_json.selection`, reused via `--query-ids`; MultiHop:
`--sample 50 --seed 42`).

### Lineup change
- **F-tuned removed → F-seq** (`systems/system_fseq.py`): SEQUENTIAL self-ask
  decomposition — resolves each hop, substitutes the resolved bridge answer into the
  next hop's retrieval query (fixes F's dead-bridge "that director" problem), answers
  over the RRF-fused union of all hops. Shares F's `DECOMPOSE_FEWSHOT_PROMPT`/`_decompose`.
- **Added a dense-kNN-only twin for every orchestration** — A-minus, B-minus, F-minus, F-seq-minus
  (via a per-call `retrieve(semantic_only=True)`) — a full **4×2 retrieval×orchestration factorial**.
  Each `X↔X-minus` delta isolates the retrieval-pipeline effect for that orchestration; together they
  test whether the dataset-dependent retriever finding holds across *all* orchestrations.
- `SYSTEM_REGISTRY` = A, A-minus, B, B-minus, F, F-minus, F-seq, F-seq-minus (8). All three models run
  all 8 systems. On Nova Lite the F-family decompose parse-fails and degrades to naive retrieval
  (measured n_steps ≈ 1 vs 3.5–3.7 on Qwen3/DeepSeek) — these cells are reported as the
  orchestration-robustness result, not as genuine decomposition measurements.

### Tier-1 answer-quality changes (lift every system sharing `ANSWER_SYSTEM_PROMPT`)
1. **Softened answer prompt** — licenses synthesis-derived answers; refuses only when a
   required fact is genuinely absent (cut F's refusal rate 50%→38% on MuSiQue). Keeps
   the `Final answer:` scoring contract intact.
2. **`fused_answer_top_k` 10→20**, held CONSTANT across B/F/F-seq (a 10-slot fused
   context was evicting retrieved gold — the dominant 4-hop failure, confirmed at
   chunk level in exp37: gold retrieved-but-not-in-answer-context).
3. **Unicode-dash normalization** in `metrics._normalize` (`1943-1992` now matches gold
   `1943–1992`); verified refusals still score wrong.

### Budget-sensitivity ablation (MuSiQue, n=50) — exp38/39/40
| System | @10 | @20 | optimum |
|---|---|---|---|
| B (iterative) | 0.600 | 0.540 | **10** |
| F (parallel decomp) | 0.340 | 0.400 | **20** |
| F-seq (sequential decomp) | 0.380 | 0.540 | **20** |

Per-strategy optimum differs (iterative accumulation dilutes with a wide budget;
fan-out needs it). **Decision: budget held CONSTANT at 20** for a controlled comparison
(trades ~0.06 of B's tuned accuracy for comparability). Lives in `config.fused_answer_top_k`.

### Headline result — retrieval × orchestration is dataset-dependent (DeepSeek-V3)
**MuSiQue 2×2 (n=50; exp38 hybrid, exp41/43 semantic-only):**
| | hybrid+rerank | semantic-only | semantic − hybrid |
|---|---|---|---|
| naive (A / A-minus) | 0.380 | 0.420 | +0.04 |
| iterative (B / B-minus) | 0.540 | **0.640** | +0.10 |

**MultiHop-RAG (n=50; exp42):** A (hybrid+rerank) **0.800** vs A-minus (semantic-only)
0.600 → **+0.20** for the pipeline (gold recall@10 0.665 vs 0.456).

**Finding (⚠ REFUTED at n=150 — see §5d; retained verbatim as the pilot record only):** the value of
the hybrid+rerank pipeline is dataset-dependent. On news (MultiHop) it is worth +0.20; on MuSiQue it
is flat-to-negative, and dense-only + iteration is best — B-minus 0.640, the top MuSiQue score, above
B and F-seq (0.540). *Mechanism:* MuSiQue's hard distractors are BM25-mined (Trivedi 2022 —
RELATED_WORK §8), so the lexical component is adversarial by construction; iteration compounds the
lexical noise under hybrid but not under semantic-only (semantic−hybrid widens +0.04→+0.10).
**n=150 verdict: the MuSiQue reversal did NOT replicate — hybrid beats dense in all 24 cells (pooled
p=5.5e-05). The dataset-dependence survived in revised form (effect size, not direction, flips).**

### Multi-hop orchestration (MuSiQue, hybrid, exp38) — by hop
F-seq is the strongest decomposition system on deep hops — 3-hop **0.600**, 4-hop
**0.444** (vs F 0.467/0.111; up to 4× on 4-hop). B leads 2-hop (0.615). F-seq ties B
overall (0.540) at ~55% of B's cost. (F-seq-vs-F = parallel-vs-sequential; F-seq-vs-B =
pre-decomposed self-ask vs free-form iteration.)

### Data integrity (MuSiQue) — verified clean
3,000 chunks = 150 questions × 20 paragraphs (exact); all 399 gold supporting
paragraphs present as chunks (0 missing); OpenSearch `rag-chunks-musique` = 3,000 =
Postgres. Retrieval searches the full pooled corpus (each query's 20 + cross-question
distractors) — the intended anti-shortcut setting.

### Caveats (must accompany any claim)
- **n=50** per dataset; hop buckets as small as 9 (4-hop); several deltas are 2–5
  queries. **Confirm the headline retriever×orchestration result at `--sample 500`.**
- **DeepSeek-V3-specific.** Re-run on Haiku/another model before generalizing.
- The MuSiQue "semantic > hybrid" effect is **partly by construction** (BM25-mined
  distractors). State as "on benchmarks whose distractors are BM25-mined," not "all
  adversarial multi-hop."
- Cross-dataset comparison is of the A-vs-A-minus *deltas* (different query sets per
  dataset) — valid; absolute MuSiQue vs MultiHop numbers are not directly comparable.

### Experiment ID map (DeepSeek-V3)
| exp | systems | dataset | note |
|---|---|---|---|
| 36 | A,B,F | MuSiQue | pre-Tier-1 baseline (B@10, old prompt) |
| 37 | B (8 steps) | MuSiQue | step ablation — worse (0.52); archived |
| 38 | A,B,F,F-seq | MuSiQue | Tier-1, budget 20 |
| 39 | B (@10) | MuSiQue | budget ablation |
| 40 | F,F-seq (@10) | MuSiQue | budget ablation |
| 41 | A-minus | MuSiQue | semantic-only naive |
| 42 | A,A-minus | MultiHop | retrieval-pipeline effect on news |
| 43 | B-minus | MuSiQue | semantic-only iterative (best MuSiQue, 0.640) |

## 5d. FINAL MATRIX — COMPLETE (2026-07-11)

The frozen final matrix is done: **9,600 runs, 1 failure (0.01%), $24.59 LLM spend.** Experiment ids:
**50/51/53** (MuSiQue × DeepSeek/Qwen/Nova, SHA `12f2a49`/`ec457dc`) and **54/56/57** (MultiHop ×
DeepSeek/Qwen/Nova, SHA `d03dd3b`; intervening commits verified inert — thesis prose + resume/billing
fix only). Full analysis: `thesis/musique_matrix_analysis.md` (Part I MuSiQue §1–8, Part II MultiHop +
cross-dataset synthesis §9–10). **Supersedes §5c's n=50 findings entirely.**

Headline (the study's central result): **orchestration and retriever effects both flip by dataset,
consistently across models.** MuSiQue: B rank-1 all models (B>A pooled p=.019); F ≤ A; hybrid>dense
small, pooled-only (p=5.5e-05). MultiHop: **F best on both capable models** (F>A pooled p=.011, F>B
p=.049) at ~⅓ of B's cost; B does NOT beat A (p=.80); hybrid>dense large and **per-model significant**
(p ≤ 5e-05 each; comparison-type +42 pts — lexical-anchor mechanism); Ammann tension resolved as
dataset-dependence. Nova decomposition collapse replicated on both datasets. Pareto: Qwen-B (MuSiQue)
and **Qwen-F dominates everything (MultiHop)** — the mid-tier model owns both frontiers. No null
over-answering found. Rankings stable across models within a dataset (τ-b .64–.74), NOT across
datasets. Thesis chapters 1/3/4 updated to the completed matrix (Ch4 §4.8 written; all [MultiHop]
markers resolved; remaining placeholder: hardware CPU/RAM in Ch3 §3.8).

## 6. Bottom line

**The code side of the audit is complete.** Done: **C1** (token F1), **C2**
(provenance + corpus fingerprint), **C3** (evidence-ever-seen for the ceiling),
**C4** (latency dispersion, in N2), **C6** (crash=wrong policy + visible
`pct_failed`), **C7** (runtime `litellm_version` capture), **D1/D2** (doc drift),
and the full **N-series** (N1–N5). **Skipped by decision:** C5 (fresh runs only).
Migrations `0003` (token F1) and `0004` (ceiling + failure) are ready to apply.

Everything still open is **yours**, not buildable here:
- **Run** (your env): `alembic upgrade head` → `compute-metrics` (token F1,
  pct_failed) + `judge` (CRAG) → the frozen 4×3 matrix, each run prefixed with
  `GIT_SHA=$(git rev-parse HEAD)`, on a single image build. Then every N-cell
  populates and A2/O3, A4/O6, RQ3/RQ4 become claimable.
- **Slides** (your deck): W1 "single LangGraph framework" → "benchmark harness";
  W2 "LiteLLM proxy" → "LiteLLM SDK"; W3 keep "Token F1" (now real); plus the
  W4–W9 wording/citation items.

Claim confidently now: A1, A3, O1, O5, RQ1, RQ2-cost.

Verification status: all added Python compiles; `token_f1`, the C3 union/fallback,
C6 `pct_failed`, C7 capture, and every N-cell's math + pandas logic were unit-tested
on synthetic data (which caught two real bugs pre-commit: the N1 diagonal crash and
the B found-then-dropped misattribution C3 fixes). Nothing has run against the live
Postgres/OpenSearch/Bedrock stack — that's your environment.
