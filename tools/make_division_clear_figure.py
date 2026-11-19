"""Draw the tree/sequence division of labour with a consistent restrained style."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_style import apply_style, clean_axis, panel_label, NAVY, TEAL, BLUE

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "reviewer_response" / "results" / "v3_hybrid"
OUT = ROOT / "paper" / "current" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
v2 = json.loads((SRC / "cv_v2_report.json").read_text(encoding="utf-8"))
base = json.loads((SRC / "nn_base_c_report.json").read_text(encoding="utf-8"))
curves = json.loads((SRC / "nn_gate_b_report.json").read_text(encoding="utf-8"))
targets = ["POR", "PERM", "SW"]
models = [("Tree (v2)", v2, NAVY), ("Network + features", base, TEAL), ("Network, curves only", curves, BLUE)]
value = {label: [(base if label == "Tree (v2)" else report)[t]["nondefault_only"]["v2" if label == "Tree (v2)" else "nn"] for t in targets] for label, report, _ in models}
gate = {label: [(base if label == "Tree (v2)" else report)[t]["gate_acc"]["v2" if label == "Tree (v2)" else "nn@0.5"] for t in targets] for label, report, _ in models}

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.78, wspace=.25)
x = np.arange(len(targets))
width = .23
handles = []
for j, (label, _, color) in enumerate(models):
    for ax, data in zip(axes, [value, gate]):
        bars = ax.bar(x + (j - 1) * width, data[label], width, label=label, color=color, zorder=3)
        for bar, val in zip(bars, data[label]):
            ax.text(bar.get_x() + bar.get_width()/2, val + .006, f"{val:.2f}", ha="center", va="bottom", fontsize=7.8)
        if ax is axes[0]:
            handles.append(bars)
axes[0].set_ylabel("Official-kernel score")
axes[0].set_ylim(0, .91)
axes[1].set_ylabel("Default-state accuracy")
axes[1].set_ylim(.84, .955)
for ax in axes:
    ax.set_xticks(x, targets)
    clean_axis(ax)
panel_label(axes[0], "a", "Value estimation on non-default samples")
panel_label(axes[1], "b", "Default-state segmentation")
fig.legend(handles=[h[0] for h in handles], labels=[m[0] for m in models], loc="upper center",
           bbox_to_anchor=(.5, .995), ncol=3, frameon=False, fontsize=8.5, handlelength=1.4, columnspacing=1.6)
fig.savefig(OUT / "fig_division_clear.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_division_clear.png", dpi=300, bbox_inches="tight")
plt.close(fig)
metadata = {
    "sources": ["cv_v2_report.json", "nn_base_c_report.json", "nn_gate_b_report.json"],
    "value_nondefault_only": value, "default_gate_accuracy": gate,
    "layout": "Two-panel comparison; shared model colours; labels use two decimals.",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / "fig_division_clear.pdf", OUT / "fig_division_clear.png"]},
}
(OUT / "fig_division_clear_data.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
