"""Restyle the verified blind/training well KS profile from saved evidence."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_style import apply_style, clean_axis, NAVY, TEAL, BLUE, DARK

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "experiments" / "reviewer_response" / "results" / "diag_divergence" / "inputs.json"
J = json.loads(source.read_text(encoding="utf-8"))
rows = sorted([(w[:8], float(v["mean_ks"]), int(v["rows"])) for w, v in J.items()], key=lambda r: -r[1])
OUT = ROOT / "paper" / "current" / "figures"
apply_style()
fig, ax = plt.subplots(figsize=(10.8, 3.8))
fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.88)
x = np.arange(len(rows))
ax.axhspan(.107, .440, color=BLUE, alpha=.14, zorder=0, label="Training leave-one-well-out range")
ax.axhline(.181, color=TEAL, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1, label="Training median (0.181)")
ax.bar(x, [r[1] for r in rows], .62, color=NAVY, zorder=3, label="Blind well")
for i, (_, ks, _) in enumerate(rows):
    ax.text(i, ks+.009, f"{ks:.3f}", ha="center", fontsize=7.8, color=DARK)
ax.set_xticks(x, [r[0] for r in rows], rotation=25, ha="right")
ax.set_ylabel("Mean KS distance to training wells")
ax.set_ylim(0, .50)
ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=8.5)
clean_axis(ax)
fig.savefig(OUT / "fig9_drift_profile.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig9_drift_profile_review.png", dpi=300, bbox_inches="tight")
plt.close(fig)
meta = {
    "source": "experiments/reviewer_response/results/diag_divergence/inputs.json",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "training_reference": {"median": .181, "range": [.107, .440]},
    "note": "Values unchanged; only visual encoding updated.",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / "fig9_drift_profile.pdf", OUT / "fig9_drift_profile_review.png"]},
}
(OUT / "fig9_drift_profile_data.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
