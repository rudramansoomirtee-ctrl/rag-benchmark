# Appendix D — Full Statistical Tables

> Appendix draft — content copied from `thesis/musique_matrix_analysis.md` §3, §3.6 and §9;
> renumber/reformat at Word conversion.

All tests below are paired (identical query sets across systems within a model, so every
system-vs-system contrast is a paired exact sign test / binomial test on discordant pairs; b =
count favouring the first-named system, c = count favouring the second).

## D.1 MuSiQue — retriever effect: hybrid vs dense-only (paired sign tests)

Copied from `thesis/musique_matrix_analysis.md` §3.1. Per-pair discordant counts (b = hybrid-only-right, c = dense-only-right):

| Model | A vs A⁻ | B vs B⁻ | F vs F⁻ | F-seq vs F-seq⁻ |
|---|---|---|---|---|
| DeepSeek | 19:9 (p=.087) | 23:13 (p=.133) | 15:11 (p=.557) | 19:12 (p=.281) |
| Qwen3 | 18:10 (p=.185) | 24:15 (p=.200) | 14:11 (p=.690) | 17:15 (p=.860) |
| Nova | 13:11 (p=.839) | 16:10 (p=.327) | 16:10 (p=.327) | 18:9 (p=.122) |

> - **12/12 pairs favour hybrid; zero reversals; 0/12 individually significant.**
> - Pooled per model: DeepSeek b=76,c=45 (**p=.0062**) · Nova b=63,c=40 (**p=.0297**) · Qwen b=73,c=51
>   (p=.0589, marginal).
> - **Pooled overall: b=212, c=136, p = 5.45×10⁻⁵ — highly significant.**
> - Required phrasing: *directionally universal, significant pooled overall and in 2 of 3 models; the
>   per-cell effect (~3–7 pts) is real but small relative to per-cell power — never imply per-cell
>   significance.*

## D.2 MuSiQue — orchestration contrasts

Copied from `thesis/musique_matrix_analysis.md` §3.2:

| Contrast | DeepSeek | Qwen3 | Nova | Pooled |
|---|---|---|---|---|
| B vs A | 18:14 (p=.60) | 22:14 (p=.24) | 16:5 (**p=.027**) | 56:33 (**p=.019**) |
| F vs A | 7:12 (A ahead) | 12:17 (A ahead) | 4:0 | 23:29 (p=.49; A ahead) |
| F-seq vs F | 21:17 | 26:20 | 4:1 | 51:38 (p=.203) |
| F-seq vs B | 14:19 (B ahead) | 16:23 (B ahead) | 10:14 (B ahead) | 40:56 (p=.125; B ahead) |

## D.3 MuSiQue — rank stability (Kendall τ-b)

Copied from `thesis/musique_matrix_analysis.md` §3.3:

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

> Kendall τ-b: DS↔Qwen **0.741** (p=.012) · DS↔Nova **0.643** (p=.037) · Qwen↔Nova **0.706** (p=.018)
> — all significant (exact permutation p-values). B is rank-1 in all three models; F-seq is the best
> decomposition variant in all three; `-minus` twins cluster at the bottom.

## D.4 MuSiQue — failure attribution (E1/DeepSeek; coverage = ≥1 gold chunk in the top-20 answering context)

Copied from `thesis/musique_matrix_analysis.md` §3.4:

| System | Coverage | ALL gold present | Gen-failures (gold present, wrong) | Ret-failures (no gold) | Lucky guesses |
|---|---|---|---|---|---|
| A | 96.7% | 73/150 | 72 | 5 | **0** |
| B | 98.7% | 88 | 71 | 2 | **0** |
| F | 97.3% | 75 | 78 | 4 | **0** |
| F-seq | 99.3% | **100** | 77 | 1 | **0** |
| (minus twins) | 93–99% | 65–74 | 73–85 | 1–10 | **0** |

> - **Errors are 87–99% generation-side**; retrieval-failure explains only 1–10 of ~75 errors per
>   system.
> - **Zero correct answers without gold in context** across all 1,200 runs — no parametric leakage;
>   correctness genuinely required retrieval.
> - **F-seq assembles complete gold evidence for 100/150 queries vs F's 75** — the sequential-bridge
>   mechanism demonstrably improves full-evidence assembly even where the accuracy gain is
>   directional-only.
> - Note: `recall_at_5` (0.37–0.59) uses a top-5 window; coverage uses the full top-20 context —
>   different denominators, both correct.

## D.5 Anomaly scan

Copied from `thesis/musique_matrix_analysis.md` §3.5:

> - 0 NULL answers (3,600/3,600); 0 cost outliers (>5× cell median); latency p95 ≤ 11.8 s (structural:
>   B/F-seq multi-call).
> - `max_tokens=800` cap held for every single-call system (A max 582). **10 Nova rows across 6 query
>   ids exceeded the cap on single calls** (max 2,340 tokens; qids 17845/17853/17868/17873) — LiteLLM
>   likely dropped the param for Nova's API shape (`drop_params=True`). Benign (Nova near-free,
>   correctness unaffected); footnote in the write-up.

## D.6 Bootstrap confidence intervals and multiplicity correction

Copied verbatim from `thesis/musique_matrix_analysis.md` §3.6 (added 2026-07-11; covers both arms):

> **Bootstrap 95% CIs** (seeded, 10,000 resamples, percentile) computed for all 48 accuracy cells and
> embedded in Ch4 Tables 4.1/4.4. Interval half-widths ≈ ±.07–.08 (n=150) and ±.05–.06 (n=200). Per-cell
> intervals overlap heavily across orchestrations within a model — comparative claims correctly rest on
> the *paired* tests, not marginal intervals. Notable exception: **A vs A-minus intervals are disjoint
> under every model on MultiHop** — the retriever effect there is visible even marginally.
>
> **Holm–Bonferroni sensitivity (α=.05).** Family 1 — the six pooled orchestration contrasts
> (B-vs-A, F-vs-A, F-seq-vs-F × 2 datasets): the smallest p (MultiHop F>A, p=.0113) misses its
> corrected threshold (.0083), so **no orchestration contrast survives correction**; all orchestration
> claims are *nominally significant / directionally unanimous*, and are phrased accordingly in Ch4.
> Family 2 — the four pooled retriever contrasts (MuSiQue pooled 5.45e-5; MultiHop per-model 6.2e-13,
> 5.0e-10, 4.8e-5): **all four survive** with large margins. The asymmetry is now stated explicitly in
> Ch4 §4.1/§4.9: retriever conclusions are multiplicity-robust; orchestration conclusions rest on
> cross-model consistency and the cross-dataset pattern. (F>B on MultiHop, p=.049, is reported as
> boundary-level only.)

The 48 bootstrap cells referenced above (24 MuSiQue + 24 MultiHop accuracy cells: 8 systems × 3
models × 2 datasets) are stated to be embedded in Chapter 4 Tables 4.1/4.4 rather than tabulated
separately in `thesis/musique_matrix_analysis.md`; see the "Not found in sources" note below.

## D.7 MultiHop-RAG — orchestration contrasts (ranking inverts vs MuSiQue)

Copied from `thesis/musique_matrix_analysis.md` §9.2:

| Contrast | DeepSeek | Qwen3 | Nova | Pooled |
|---|---|---|---|---|
| F vs A | 11:6 (p=.33) | 15:4 (**p=.019**) | 0:4 † | DS+Qwen 26:10 (**p=.011**) |
| B vs A | 8:9 (p=1.0) | 14:9 (p=.41) | 6:13 (p=.17, A ahead) | 28:31 (p=.80) — **B does not beat A** |
| F vs B | 11:5 | 11:5 | 11:8 | 33:18 (**p=.049**) — **F beats B** |
| F-seq vs F | 11:13 | 10:16 | 3:2 | 24:31 (p=.42, F ahead) |

## D.8 MultiHop-RAG — retriever effect (per-model significant)

Copied from `thesis/musique_matrix_analysis.md` §9.3:

> Hybrid − dense deltas: A +0.160/+0.135/+0.110 · B +0.065/+0.070/−0.005 · F +0.070/+0.085/+0.075 ·
> F-seq +0.130/+0.120/+0.065 (DS/Qwen/Nova). 11 of 12 cells favour hybrid (single tiny exception: Nova
> B −0.005).
>
> Pooled sign tests per model: DeepSeek 115:30 (**p=6×10⁻¹³**) · Qwen 129:47 (**p=5×10⁻¹⁰**) · Nova
> 99:49 (**p=5×10⁻⁵**) · overall 343:126 (**p=3×10⁻²⁴**). Where MuSiQue needed pooling across the whole
> matrix to secure a small effect, MultiHop's effect is **individually significant within every
> model** — the dataset-dependence of the retrieval pipeline, now measured on both sides.

## D.9 Kendall τ-b summary (both arms)

MuSiQue (§D.3 above): DS↔Qwen 0.741 (p=.012); DS↔Nova 0.643 (p=.037); Qwen↔Nova 0.706 (p=.018).

MultiHop-RAG: `thesis/musique_matrix_analysis.md` §10 states rankings "invert" across datasets and
that within-dataset rank stability "preserves F/F-seq/A/B top-4 on both capable models," but does not
give a separate numeric τ-b table for the MultiHop arm distinct from the cross-dataset synthesis
narrative — see the "Not found" note below.

---

**Not found in sources:** the task brief for this appendix asks for "all 48 bootstrap CIs" and "all
paired sign tests with discordant-pair counts and p-values" as full itemised tables. The source
document (`thesis/musique_matrix_analysis.md` §3.6) states the 48 bootstrap CIs are computed and
embedded in Chapter 4 Tables 4.1/4.4, but does not itemise the 48 individual interval bounds
(lower/upper) anywhere in the three source files reviewed — only the half-width ranges (±.07–.08 for
n=150, ±.05–.06 for n=200) and the one qualitative exception (A vs A-minus on MultiHop) are given. If
per-cell CI bounds are required for the appendix, they must be pulled from Chapter 4 Tables 4.1/4.4
directly (not reviewed as part of this task) or recomputed from the notebook (`notebooks/analysis.py`,
cell `bootstrap_ci`) against the live Postgres database — flag this as a follow-up rather than
fabricate numbers. Similarly, no standalone numeric MultiHop-RAG Kendall τ-b table (paralleling D.3)
was found in the reviewed sources.
