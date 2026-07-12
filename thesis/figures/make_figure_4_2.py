"""
Figure 4.2 - Grouped bar chart of containment accuracy by system and model, MuSiQue.

Data source: thesis/chapter4_results.md, Table 4.1 (accuracy + bootstrap 95% CI bounds).
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

# accuracy, ci_low, ci_high -- Table 4.1
data = {
    "DeepSeek-V3": {
        "A": (0.487, 0.41, 0.57),
        "A-minus": (0.420, 0.35, 0.50),
        "B": (0.513, 0.43, 0.59),
        "B-minus": (0.447, 0.37, 0.53),
        "F": (0.453, 0.37, 0.53),
        "F-minus": (0.427, 0.35, 0.51),
        "F-seq": (0.480, 0.40, 0.56),
        "F-seq-minus": (0.433, 0.35, 0.51),
    },
    "Qwen3-32B": {
        "A": (0.473, 0.39, 0.55),
        "A-minus": (0.420, 0.34, 0.50),
        "B": (0.527, 0.45, 0.61),
        "B-minus": (0.467, 0.39, 0.55),
        "F": (0.440, 0.36, 0.52),
        "F-minus": (0.420, 0.34, 0.50),
        "F-seq": (0.480, 0.40, 0.56),
        "F-seq-minus": (0.467, 0.39, 0.55),
    },
    "Nova Lite": {
        "A": (0.273, 0.20, 0.35),
        "A-minus": (0.260, 0.19, 0.33),
        "B": (0.347, 0.27, 0.43),
        "B-minus": (0.307, 0.23, 0.38),
        "F": (0.300, 0.23, 0.37),
        "F-minus": (0.260, 0.19, 0.33),
        "F-seq": (0.320, 0.25, 0.39),
        "F-seq-minus": (0.260, 0.19, 0.33),
    },
}

# Nova's decomposition systems (F, F-minus, F-seq, F-seq-minus) are flagged degraded (dagger) in Table 4.1.
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
ax.set_ylim(0, 0.68)
ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

handles, labels = ax.get_legend_handles_labels()
hatch_patch = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                             hatch="///", label="Nova Lite, degraded decomposition (†)")
handles.append(hatch_patch)
labels.append("Nova Lite, degraded decomposition (†)")
ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)

fig.text(
    0.01, 0.01,
    "† decomposition failed to parse on ~85% of questions and collapsed to "
    "single-retrieval behaviour (see Table 4.1).",
    fontsize=6.5, color="#444444", ha="left",
)

fig.tight_layout(rect=[0, 0.035, 1, 1])
out = "thesis/figures/figure_4_2_musique_accuracy.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
