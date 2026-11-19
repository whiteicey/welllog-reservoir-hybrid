"""Draw the v1--v4 comparison as paired observations, not truncated rainbow bars."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_style import apply_style, clean_axis, NAVY, RED, GREY, DARK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "current" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "experiments" / "reviewer_response" / "results" / "model_family_comparison.json").read_text(encoding="utf-8"))["rows"]
names = [r["short_label"] for r in R]
oof = np.array([r["oof"] for r in R])
blind = np.array([r["blind"] for r in R])
apply_style()
fig, ax = plt.subplots(figsize=(10.8, 4.5))
fig.subplots_adjust(left=.19, right=.97, bottom=.18, top=.82)
y = np.arange(len(R))[::-1]
for yi, a, b in zip(y, oof, blind):
    ax.plot([a, b], [yi, yi], color="#D0D6DC", linewidth=1.25, zorder=1)
ax.scatter(oof, y, marker="o", s=52, color=NAVY, zorder=3, label="OOF (row-specific protocol)")
ax.scatter(blind, y, marker="D", s=42, color=RED, zorder=3, label="Historical blind feedback")
for yi, a, b in zip(y, oof, blind):
    ax.text(a-.07, yi+.18, f"{a:.2f}", ha="right", va="bottom", fontsize=7.8, color=NAVY, fontweight="semibold")
    ax.text(b+.07, yi+.18, f"{b:.2f}", ha="left", va="bottom", fontsize=7.8, color=RED, fontweight="semibold")
ax.set_yticks(y, names)
ax.set_ylim(-.65, len(R)-.35)
ax.set_xlim(78.8, 84.55)
ax.set_xlabel("Total score (row-specific protocols)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2, fontsize=8.5)
clean_axis(ax, grid="x")
ax.grid(axis="y", visible=False)
fig.savefig(OUT / "fig_model_family_comparison.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_model_family_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
meta = {
    "source": "experiments/reviewer_response/results/model_family_comparison.json",
    "note": "Only the v1–v4 comparison is shown. OOF and historical blind records are paired observations, not a conversion or a common-protocol ranking.",
    "layout": "Horizontal paired observations; value labels sit above points with extra bottom-axis clearance.",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / "fig_model_family_comparison.pdf", OUT / "fig_model_family_comparison.png"]},
}
(OUT / "fig_model_family_comparison_data.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
