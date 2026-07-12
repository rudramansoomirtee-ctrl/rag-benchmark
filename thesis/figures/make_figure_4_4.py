"""
Figure 4.4 - MultiHop-RAG retriever effect (hybrid A vs dense-only A-minus) by question type,
DeepSeek-V3.

Data source: thesis/musique_matrix_analysis.md Section 9.3 / chapter4_results.md Section 4.8.

IMPORTANT DATA GAP: only two of the four MultiHop-RAG question types (comparison, inference)
have a printed A vs A-minus pair under DeepSeek-V3:
  - comparison: A = 0.791, A-minus = 0.373  (musique_matrix_analysis.md S9.3 / chapter4 S4.8)
  - inference:  A = 0.938, A-minus = 0.938  (chapter4_results.md S4.8)
Temporal has no printed A/A-minus by-type numbers in either source file. Null has only A = 0.917
and B = 0.958 printed (musique_matrix_analysis.md S9.3) -- no A-minus value for null -- so it
cannot be included in this hybrid-vs-dense paired figure. Per instructions, these two types are
OMITTED rather than invented; see the final report for this note.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_HYBRID = "#0072B2"   # DeepSeek-V3 blue, reused as "hybrid A" series
COLOR_DENSE = "#E69F00"    # orange, "dense-only A-minus" series

question_types = ["Inference", "Comparison"]
hybrid_acc = [0.938, 0.791]
dense_acc = [0.938, 0.373]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")

x = np.arange(len(question_types))
bar_width = 0.32

bars_h = ax.bar(x - bar_width / 2, hybrid_acc, width=bar_width,
                 color=COLOR_HYBRID, edgecolor="black", linewidth=0.5,
                 label="A (hybrid retrieval)")
bars_d = ax.bar(x + bar_width / 2, dense_acc, width=bar_width,
                 color=COLOR_DENSE, edgecolor="black", linewidth=0.5,
                 label="A-minus (dense-only retrieval)")

for bars in (bars_h, bars_d):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", (bar.get_x() + bar.get_width() / 2, height),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)

ax.set_xlim(-0.55, 1.55)

# Annotate the comparison-type gap explicitly, as requested.
ax.annotate(
    "", xy=(1 + bar_width / 2, 0.373), xytext=(1 + bar_width / 2, 0.791),
    arrowprops=dict(arrowstyle="<->", color="#444444", linewidth=1.0),
)
ax.text(1 + bar_width / 2 + 0.04, 0.58, "gap = 0.418\n(0.791 − 0.373)",
        fontsize=8, color="#444444", va="center")

ax.set_xticks(x)
ax.set_xticklabels(question_types, fontsize=10)
ax.set_ylabel("Containment accuracy (DeepSeek-V3)", fontsize=10)
ax.set_ylim(0, 1.12)
ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0))

fig.text(
    0.01, 0.01,
    "Temporal and null types omitted: no hybrid vs dense-only by-type pair is reported for them.",
    fontsize=6.5, color="#444444", ha="left",
)

fig.tight_layout(rect=[0, 0.06, 1, 1])
out = "thesis/figures/figure_4_4_retriever_by_type.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
