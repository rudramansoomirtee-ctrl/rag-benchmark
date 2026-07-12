"""
Figure 4.1 - Accuracy vs cost-per-correct-answer Pareto scatter, MuSiQue (left) vs MultiHop-RAG (right).

Data source: thesis/chapter4_results.md (Table 4.3, Section 4.8) and
thesis/musique_matrix_analysis.md (Section 5 full cost grid; Section 9.4 MultiHop cost points).
Every (accuracy, cost) pair below is copied verbatim from those files - no interpolation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Okabe-Ito colorblind-safe palette, consistent model -> color mapping across all figures.
COLOR = {
    "DeepSeek-V3": "#0072B2",
    "Qwen3-32B": "#E69F00",
    "Nova Lite": "#009E73",
}
MARKER = {
    "DeepSeek-V3": "o",
    "Qwen3-32B": "s",
    "Nova Lite": "^",
}

# ---------------------------------------------------------------------------
# MuSiQue: hybrid-system cells (A, B, F, F-seq) x 3 models.
# accuracy, cost-per-correct-answer  -- thesis/musique_matrix_analysis.md Section 5
# (matches the subset in chapter4_results.md Table 4.3 to rounding)
musique_points = [
    # (model, system, accuracy, cost_per_correct)
    ("Nova Lite", "A", 0.273, 0.00083),
    ("Nova Lite", "F", 0.300, 0.00119),
    ("Nova Lite", "F-seq", 0.320, 0.00119),
    ("Nova Lite", "B", 0.347, 0.00276),
    ("Qwen3-32B", "A", 0.473, 0.00128),
    ("Qwen3-32B", "F", 0.440, 0.00170),
    ("Qwen3-32B", "F-seq", 0.480, 0.00230),
    ("Qwen3-32B", "B", 0.527, 0.00416),
    ("DeepSeek-V3", "A", 0.487, 0.00425),
    ("DeepSeek-V3", "F", 0.453, 0.00574),
    ("DeepSeek-V3", "F-seq", 0.480, 0.00814),
    ("DeepSeek-V3", "B", 0.513, 0.01710),
]

# Pareto frontier stated explicitly in musique_matrix_analysis.md Section 5:
# "Nova-A -> Nova-F-seq -> Qwen-A -> Qwen-F-seq -> Qwen-B"
musique_frontier = [
    ("Nova Lite", "A"),
    ("Nova Lite", "F-seq"),
    ("Qwen3-32B", "A"),
    ("Qwen3-32B", "F-seq"),
    ("Qwen3-32B", "B"),
]

# ---------------------------------------------------------------------------
# MultiHop-RAG: only 4 cells have a printed cost-per-correct value, in
# chapter4_results.md Section 4.8 and musique_matrix_analysis.md Section 9.4.
# No full 24-cell cost grid is printed for MultiHop-RAG (unlike MuSiQue Section 5),
# so only these four points are plotted -- see figure caption / report for the omission.
multihop_points = [
    ("Nova Lite", "A", 0.785, 0.00064),
    ("Qwen3-32B", "F", 0.875, 0.00147),
    ("Qwen3-32B", "B", 0.845, 0.00384),
    ("DeepSeek-V3", "F", 0.855, 0.00557),
]

# Pareto frontier stated explicitly: "collapses to two points: Nova-A ... -> Qwen-F"
multihop_frontier = [
    ("Nova Lite", "A"),
    ("Qwen3-32B", "F"),
]


def plot_panel(ax, points, frontier, title, xticks):
    lookup = {(m, s): (acc, cost) for m, s, acc, cost in points}

    for model, system, acc, cost in points:
        ax.scatter(
            cost, acc,
            marker=MARKER[model], color=COLOR[model],
            s=90, edgecolor="black", linewidth=0.6, zorder=3,
        )
        ax.annotate(
            system, (cost, acc),
            textcoords="offset points", xytext=(7, 4),
            fontsize=9, color="black",
        )

    frontier_xy = [lookup[key] for key in frontier]
    fx = [c for _, c in frontier_xy]
    fy = [a for a, _ in frontier_xy]
    ax.plot(fx, fy, color="#555555", linewidth=1.4, linestyle="--",
             zorder=2, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per correct answer ($, log scale)", fontsize=10)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.4g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")

plot_panel(ax1, musique_points, musique_frontier, "MuSiQue",
           xticks=[0.001, 0.002, 0.005, 0.01, 0.02])
ax1.set_ylabel("Containment accuracy", fontsize=10)
ax1.set_ylim(0.20, 0.58)

plot_panel(ax2, multihop_points, multihop_frontier, "MultiHop-RAG",
           xticks=[0.0006, 0.001, 0.002, 0.005])
ax2.set_ylim(0.60, 0.92)

legend_handles = [
    plt.Line2D([0], [0], marker=MARKER[m], color="w", markerfacecolor=COLOR[m],
               markeredgecolor="black", markersize=8, label=m)
    for m in ["DeepSeek-V3", "Qwen3-32B", "Nova Lite"]
]
legend_handles.append(
    plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.4, label="Pareto frontier")
)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.06, 1, 1])
out = "thesis/figures/figure_4_1_pareto.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
