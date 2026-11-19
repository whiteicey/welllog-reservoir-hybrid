"""Regenerate the paper's public-test figure from verified metrics, without retraining."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "reproducibility" / "public_volve"
sys.path.insert(0, str(ROOT.parent / "tools"))
from paper_style import apply_style, clean_axis, panel_label, NAVY, BLUE, TEAL, PURPLE, GREY, DARK

R = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
M = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
assert hashlib.sha256((ROOT / "metrics.json").read_bytes()).hexdigest() == M["outputs"]["metrics.json"]["sha256"]

names = ["rf_five_log", "tree_regression", "distribution_mean", "distribution_tolerance_bayes", "tree_regression_joint_sw_gate"]
labels = ["RF five logs", "Tree regression", "Distribution mean", "Distribution tolerance", "Tree + SW gate"]
colors = [GREY, NAVY, BLUE, BLUE, TEAL]
per_well = R["per_well"]
wells = list(per_well)
apply_style()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.12, 1]})
fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.88, wspace=.25)
vals = [R["variants"][n]["rmse_all3"] for n in names]
x = np.arange(len(names))
axes[0].bar(x, vals, .62, color=colors, zorder=3)
axes[0].set_xticks(x, labels, rotation=22, ha="right")
axes[0].set_ylabel("Pooled RMSE")
axes[0].set_ylim(0, 0.142)
for i, v in enumerate(vals):
    axes[0].text(i, v + .0025, f"{v:.4f}", ha="center", fontsize=8.0, color=DARK)
panel_label(axes[0], "a", "Model and decoder")

x = np.arange(len(wells))
# Distinguish overlapping lines with dashed style and white marker fill for Tree
for name, label, color, marker, ls, mfc in [
    ("rf_five_log", "RF", GREY, "s", "-", GREY),
    ("tree_regression", "Tree", NAVY, "o", "--", "white"),
    ("tree_regression_joint_sw_gate", "Tree + SW gate", TEAL, "D", "-", TEAL),
]:
    axes[1].plot(x, [per_well[w][name]["rmse_all3"] for w in wells], marker=marker,
                 markersize=5.2, markeredgewidth=1.3, markerfacecolor=mfc,
                 linestyle=ls, linewidth=1.35, label=label, color=color)
axes[1].set_xticks(x, wells)
axes[1].set_xlabel("Released test well")
axes[1].set_ylabel("RMSE")
axes[1].set_ylim(0.035, 0.165)
axes[1].legend(frameon=False, loc="upper right", ncol=1, fontsize=8.5)
panel_label(axes[1], "b", "Well-to-well variation")
for ax in axes:
    clean_axis(ax)
fig.savefig(ROOT / "fig_public_volve_paper.pdf", bbox_inches="tight")
fig.savefig(ROOT / "fig_public_volve_paper.png", dpi=240, bbox_inches="tight")
plt.close(fig)
meta = {
    "source": "metrics.json; SHA checked against experiment_manifest.json",
    "note": "Released public test wells were used in earlier development; this is a public reproducibility check.",
    "style": "Restrained blue/teal encoding; absolute-zero RMSE bars and per-well point-line comparison.",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "fig_public_volve_paper.pdf", ROOT / "fig_public_volve_paper.png"]},
}
(ROOT / "paper_figure_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
