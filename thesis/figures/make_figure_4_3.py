"""
Figure 4.3 - Grouped bar chart of containment accuracy by system and model, MultiHop-RAG.

Data source: thesis/chapter4_results.md, Table 4.4 (accuracy + bootstrap 95% CI bounds).
All values copied verbatim; systems in table order.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = {
    "DeepSeek-V3": "#0072B2",
    "Qwen3-32B": "#E69F00",
    "Nova Lite": "#009E73",
}

systems = ["A", "A-minus", "B", "B-minus", "F", "F-minus", "F-seq", "F-seq-minus"]
models = ["DeepSeek-V3", "Qwen3-32B", "Nova Lite"]

# accuracy, ci_low, ci_high -- Table 4.4
data = {
    "DeepSeek-V3": {
        "A": (0.830, 0.78, 0.88),
        "A-minus": (0.670, 0.61, 0.74),
        "B": (0.825, 0.77, 0.88),
        "B-minus": (0.760, 0.70, 0.82),
        "F": (0.855, 0.81, 0.90),
        "F-minus": (0.785, 0.73, 0.84),
        "F-seq": (0.845, 0.79, 0.90),
        "F-seq-minus": (0.715, 0.65, 0.78),
    },
    "Qwen3-32B": {
        "A": (0.820, 0.77, 0.87),
        "A-minus": (0.685, 0.62, 0.75),
        "B": (0.845, 0.80, 0.90),
        "B-minus": (0.775, 0.72, 0.83),
        "F": (0.875, 0.83, 0.92),
        "F-minus": (0.790, 0.74, 0.85),
        "F-seq": (0.845, 0.80, 0.90),
        "F-seq-minus": (0.725, 0.67, 0.79),
    },
    "Nova Lite": {
        "A": (0.785, 0.73, 0.84),
        "A-minus": (0.675, 0.61, 0.74),
        "B": (0.755, 0.70, 0.81),
        "B-minus": (0.760, 0.70, 0.82),
        "F": (0.770, 0.71, 0.83),
        "F-minus": (0.695, 0.63, 0.76),
        "F-seq": (0.775, 0.72, 0.83),
        "F-seq-minus": (0.710, 0.65, 0.77),
    },
}

# Nova's decomposition systems (F, F-minus, F-seq, F-seq-minus) are flagged degraded (dagger) in Table 4.4.
nova_degraded = {"F", "F-minus", "F-seq", "F-seq-minus"}

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")

n_groups = len(systems)
n_models = len(models)
bar_width = 0.25
x = np.arange(n_groups)

for i, model in enumerate(models):
    accs = [data[model][s][0] for s in systems]
    lo = [data[model][s][0] - data[model][s][1] for s in systems]
    hi = [data[model][s][2] - data[model][s][0] for s in systems]
    offset = (i - (n_models - 1) / 2) * bar_width
    hatches = ["///" if (model == "Nova Lite" and s in nova_degraded) else None for s in systems]
    bars = ax.bar(
        x + offset, accs, width=bar_width,
        yerr=[lo, hi], capsize=2.5,
        color=COLOR[model], edgecolor="black", linewidth=0.5,
        label=model, error_kw={"linewidth": 0.8, "ecolor": "black"},
    )
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)

ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=9)
ax.set_ylabel("Containment accuracy", fontsize=10)
ax.set_ylim(0, 1.12)
ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

handles, labels = ax.get_legend_handles_labels()
hatch_patch = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                             hatch="///", label="Nova Lite, degraded decomposition (†)")
handles.append(hatch_patch)
labels.append("Nova Lite, degraded decomposition (†)")
ax.legend(handles, labels, loc="upper center", fontsize=8, frameon=False, ncol=2)

fig.text(
    0.01, 0.01,
    "† decomposition failed to parse (mean retrieval counts 1.45-1.57 vs 3.4-3.7) and "
    "collapsed to near-single-retrieval behaviour (see Table 4.4).",
    fontsize=6.5, color="#444444", ha="left",
)

fig.tight_layout(rect=[0, 0.035, 1, 1])
out = "thesis/figures/figure_4_3_multihop_accuracy.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
