"""Figure 14: the five final in-province amendments of the campaign (OOF vs blind).
Reproducible from the sealed submission ledger; run: python make_fig14.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "fig14_campaign_amendments.pdf"

# (label, OOF delta vs v97 baseline, blind delta vs v97 baseline, colour)
PTS = [
    ("4-family ensemble", 0.129, -0.0405, "#E8A33D"),
    ("triple-chain ensemble", 0.204, -0.1626, "#E8A33D"),
    ("external gate", 0.081, -0.0126, "#2E8B57"),
    ("distillation (fused teacher)", 0.138, -0.1411, "#7B4FA0"),
    ("distillation (mega teacher)", 0.192, -0.1653, "#7B4FA0"),
]
BASE_OFFSET = 1.86  # blind - OOF offset of the unchanged single-family baseline

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axhspan(-0.078, 0.078, color="0.85", zorder=0, label="noise floor (±0.078)")
ax.axhline(0.0, color="0.4", lw=1, ls="--", zorder=1)
ax.scatter([0], [0], marker="*", s=220, color="k", zorder=5, label="v97 baseline")
for label, x, y, col in PTS:
    ax.scatter([x], [y], s=70, color=col, zorder=4, edgecolor="k", linewidth=0.5)
    off = BASE_OFFSET + (y - x)
    ax.annotate(f"offset {off:.2f}", (x, y), textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color="0.15")
# trend of the four reference points (baseline + three decreasing offsets)
ax.annotate("blind best (least amended morphology)\n83.9361", (0, 0), textcoords="offset points",
            xytext=(18, -50), fontsize=7.5, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
ax.set_xlabel("honest OOF gain over the v97 baseline (points)")
ax.set_ylabel("blind-leaderboard change vs baseline (points)")
ax.set_xlim(-0.02, 0.24)
ax.set_ylim(-0.21, 0.10)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="lower left", fontsize=7, frameon=True)
ax.set_title("Final in-province amendments: larger OOF gains, smaller blind-score offsets", fontsize=9)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("SAVED", OUT)
