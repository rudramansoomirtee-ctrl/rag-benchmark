# Appendix B — Answer-Context Budget Ablation and Step-Budget Material

> Appendix draft — content copied from `CLAUDE.md` and `thesis/musique_matrix_analysis.md` /
> `DISSERTATION_AUDIT.md`; renumber/reformat at Word conversion.

## B.1 Why the budget was ablated (`CLAUDE.md`)

Copied from `CLAUDE.md`, "Shared (every system uses the same underlying calls)" table, `Multi-list fusion` row:

> `retrieval/retrieve.py:rrf_fuse` — client-side RRF over per-query ranked lists; B fuses its
> iteration lists, F its sub-question lists. Answer context = fused top `FUSED_ANSWER_TOP_K = 20`,
> held **constant** across all fusing systems (B/F/F-seq) so the comparison isolates orchestration
> strategy, not the budget knob (raised 10→20 after exp36/37 showed retrieved gold evicted from a
> 10-slot context). The budget ablation (exp38/39/40) found the optimum is per-strategy — B@10=0.600
> > B@20=0.540, but F-seq@20=0.540 ≫ F-seq@10=0.380 — so a fixed value was adopted for a controlled
> comparison. Budget is now **uniform at 20 across ALL systems**: `top_k=20` (so A/A-minus also answer
> over 20) = `FUSED_ANSWER_TOP_K=20`, removing the earlier A=10 vs fusing=20 asymmetry.

Copied from `CLAUDE.md` config block:

```python
top_k                   = 20                     # uniform answer budget; A/A-minus answer over their top-20, = fused_answer_top_k (removes the old A=10 asymmetry)
fused_answer_top_k      = 20                     # answer-context budget for B/F/F-seq (fused top-N); = top_k so the budget is uniform across all 8 systems
retrieval_pool          = 40                     # hybrid first-stage pool before rerank (~2× top_k so the reranker selects, not just reorders)
```

## B.2 Budget-sensitivity ablation results (MuSiQue, n=50) — exp38/39/40

Copied from `DISSERTATION_AUDIT.md` §5c:

| System | @10 | @20 | optimum |
|---|---|---|---|
| B (iterative) | 0.600 | 0.540 | **10** |
| F (parallel decomp) | 0.340 | 0.400 | **20** |
| F-seq (sequential decomp) | 0.380 | 0.540 | **20** |

> Per-strategy optimum differs (iterative accumulation dilutes with a wide budget; fan-out needs it).
> **Decision: budget held CONSTANT at 20** for a controlled comparison (trades ~0.06 of B's tuned
> accuracy for comparability). Lives in `config.fused_answer_top_k`.

Cross-referenced in `thesis/musique_matrix_analysis.md` §5 caveat:

> Caveat: $/correct covers LLM generation only; Cohere rerank is a separately-metered per-retrieval
> charge borne by hybrid systems only (disclosed in Ch3 §3.5).

## B.3 Experiment ID map for the ablation (DeepSeek-V3)

Copied from `DISSERTATION_AUDIT.md` §5c:

| exp | systems | dataset | note |
|---|---|---|---|
| 36 | A,B,F | MuSiQue | pre-Tier-1 baseline (B@10, old prompt) |
| 37 | B (8 steps) | MuSiQue | step ablation — worse (0.52); archived |
| 38 | A,B,F,F-seq | MuSiQue | Tier-1, budget 20 |
| 39 | B (@10) | MuSiQue | budget ablation |
| 40 | F,F-seq (@10) | MuSiQue | budget ablation |
| 41 | A-minus | MuSiQue | semantic-only naive |
| 42 | A,A-minus | MultiHop | retrieval-pipeline effect on news |
| 43 | B-minus | MuSiQue | semantic-only iterative (best MuSiQue at n=50, 0.640 — later refuted, see Appendix E / §B.5 below) |

## B.4 Step-budget material

Copied from `CLAUDE.md`, System B spec:

> Max steps | n/a | per-instance; default `settings.max_agent_steps = 5` (B1/B3/B5 sweep removed)

> **Iteration budget** is per-instance. The B1/B3/B5 sweep (budgets 1/3/5 as separate registry
> entries) was **removed** — `SYSTEM_REGISTRY` is now the 4×2 retrieval×orchestration factorial `A,
> A-minus, B, B-minus, F, F-minus, F-seq, F-seq-minus` (the `-minus` twins force
> `retrieve(semantic_only=True)`); historical sweep runs remain in the DB.

Copied from `DISSERTATION_AUDIT.md` §5c experiment table (exp37): a step-budget ablation of B at 8
steps was also run and found worse than the 5-step default (0.52 accuracy vs the 5-step baseline);
noted as "archived."

The frozen final-matrix step budget is `max_agent_steps = 5` for System B in all final experiments
(50/51/53/54/56/57) — see `CLAUDE.md` config block and Appendix F.

## B.5 Note on the n=50 pilot ablation vs the final n=150/n=200 matrix

The budget ablation itself (exp38/39/40, §B.2 above) is **not** among the claims refuted by the final
matrix — it concerns the answer-context budget knob and was used to justify freezing
`fused_answer_top_k=20`, a decision that stands. What *was* refuted at n=150 is a separate, adjacent
n=50 pilot claim ("dense-only wins on MuSiQue; B-minus (0.640) is the champion") recorded in the same
§5c section and in exp43's note above — see Appendix E (integrity audit) and
`thesis/musique_matrix_analysis.md` §4.1 for the correction. The two must not be conflated: the budget
ablation's per-strategy-optimum finding is retained; the retriever-reversal finding built on the same
n=50 sample is refuted.

---

**Not found in sources:** no ablation of `retrieval_pool` (40) or of the reranker choice
(`bge-reranker-v2-m3` vs Cohere Rerank 3.5) was located in the source documents — only the frozen
values are recorded (Appendix F). No numeric ablation table for `max_agent_steps` values other than 5
and 8 (exp37) was found.
