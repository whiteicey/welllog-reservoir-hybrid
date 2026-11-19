"""Style the verified full-v4 public comparison without altering experiment artifacts."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_style import apply_style, clean_axis, panel_label, NAVY, BLUE, TEAL, PURPLE, GREY, DARK

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "public_v4" / "results" / "v4_root_20261113" / "stages"
R = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
M = json.loads((SRC / "experiment_manifest.json").read_text(encoding="utf-8"))
assert hashlib.sha256((SRC / "metrics.json").read_bytes()).hexdigest() == M["outputs"]["metrics.json"]["sha256"]
OUT = ROOT / "paper" / "current" / "figures"
apply_style()
labels = ["RF five logs", "Tree + SW gate", "v4 root", "v4 full chain"]
values = [R["comparison"]["rf_five_log"], R["comparison"]["target_tree_gate"],
          R["root_only_rmse"], R["public_test"]["metrics"]["rmse_all3"]]
colors = [GREY, TEAL, BLUE, PURPLE]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.12]})
fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.88, wspace=.25)
x = np.arange(len(labels))
axes[0].bar(x, values, .62, color=colors, zorder=3)
axes[0].set_xticks(x, labels, rotation=20, ha="right")
axes[0].set_ylabel("Pooled RMSE")
axes[0].set_ylim(0, 0.178)
for i, v in enumerate(values):
    axes[0].text(i, v + .0025, f"{v:.4f}", ha="center", fontsize=8.0, color=DARK)
panel_label(axes[0], "a", "Released-test RMSE")

wells = list(R["public_test"]["per_well"])
root = json.loads((SRC.parent / "metrics.json").read_text(encoding="utf-8"))["public_test"]["per_well"]
x = np.arange(len(wells))
axes[1].plot(x, [R["public_test"]["per_well"][w]["rmse_all3"] for w in wells],
             color=PURPLE, marker="^", markersize=5.4, markeredgewidth=1.3, linewidth=1.35, label="v4 full chain")
axes[1].plot(x, [root[w]["v4_root_dual_backbone_mean"]["rmse_all3"] for w in wells],
             color=BLUE, marker="o", markersize=5.2, markeredgewidth=1.3, linewidth=1.35, label="v4 root")
axes[1].plot(x, [root[w]["target_tree_gate"]["rmse_all3"] for w in wells],
             color=TEAL, marker="D", markersize=5.2, markeredgewidth=1.3, linewidth=1.35, label="Tree + SW gate")
axes[1].set_xticks(x, wells)
axes[1].set_xlabel("Released test well")
axes[1].set_ylabel("RMSE")
axes[1].set_ylim(0.035, 0.178)
axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
panel_label(axes[1], "b", "The same ordering across wells")
for ax in axes:
    clean_axis(ax)
fig.savefig(OUT / "fig_v4_public_styled.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_v4_public_styled.png", dpi=300, bbox_inches="tight")
plt.close(fig)
meta = {
    "source": "experiments/public_v4/results/v4_root_20261113/stages/metrics.json and ../metrics.json",
    "source_sha256": M["outputs"]["metrics.json"]["sha256"],
    "note": "Presentation-only figure; original experiment metrics and predictions are unchanged.",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / "fig_v4_public_styled.pdf", OUT / "fig_v4_public_styled.png"]},
}
(OUT / "fig_v4_public_styled_data.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
