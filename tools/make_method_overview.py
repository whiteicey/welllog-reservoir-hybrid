"""Create a clean, publication-grade conceptual method overview diagram.

Elsevier Geodata and AI style:
- Clean, uncluttered 4-stage horizontal pipeline
- Symmetrical layout and perfectly aligned orthogonal connection arrows
- Crisp typography with ample margins and zero text overflow
- Pruned, concise academic tags for immediate visual clarity
- Accurate mathematical formulations for hierarchical fusion and expected utility decoding
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from paper_style import (apply_style, NAVY, BLUE, TEAL, PURPLE, RED, GREY, DARK, PALE,
                         TINT_NAVY, TINT_TEAL, TINT_PURPLE, TINT_GREY, TINT_YELLOW,
                         TINT_BLUE, BORDER_GREY)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "current" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
apply_style()


def create_figure():
    # Canvas: 13.0 inches x 4.70 inches (tightened vertically for cohesive, elegant proportions)
    fig, ax = plt.subplots(figsize=(13.0, 4.70), dpi=300)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.70)
    ax.axis("off")

    # Helper: Main Stage Container Card
    def card(x, y, w, h, title, num, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                                    linewidth=1.1, edgecolor=color, facecolor="#FCFDFE"))
        header_h = 0.36
        ax.add_patch(FancyBboxPatch((x, y + h - header_h), w, header_h,
                                    boxstyle="round,pad=0.015,rounding_size=0.06",
                                    linewidth=0, facecolor=color))
        ax.add_patch(Rectangle((x, y + h - header_h), w, 0.15, linewidth=0, facecolor=color))
        ax.text(x + w / 2, y + h - header_h / 2, f"Stage {num}: {title}",
                ha="center", va="center", fontsize=8.6, fontweight="bold", color="white")

    # Helper: Sub-module Block
    def block(x, y, w, h, title, color, face="white", pad=0.012):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.04",
                               linewidth=0.85, edgecolor=color, facecolor=face)
        ax.add_patch(patch)
        if title:
            ax.text(x + 0.12, y + h - 0.15, title, ha="left", va="center",
                    fontsize=7.4, fontweight="bold", color=color)
        return patch

    # Helper: Arrow (strictly no penetration: caller passes clearance coordinates)
    def arrow(x1, y1, x2, y2, color=GREY, lw=0.95, dashed=False, mscale=9.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mscale,
                                     linewidth=lw, color=color, linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=0, zorder=4))

    # =========================================================================
    # STAGE 1: MULTI-VIEW INPUT (x: 0.20 -> 2.90, w: 2.70)
    # =========================================================================
    card(0.20, 0.35, 2.70, 4.15, "Multi-View Input", "1", NAVY)

    # Top: Raw logging input pill
    block(0.33, 3.42, 2.44, 0.58, "Wireline Logging Curves", NAVY, TINT_NAVY)
    ax.text(0.44, 3.68, "• 10 curves (GR, RT, RHOB, NPHI, etc.)", fontsize=6.4, color=DARK)
    ax.text(0.44, 3.52, "• 0.125 m continuous depth sampling", fontsize=6.2, color="#52606A")

    # Downward arrow to views (enlarged, distinct arrow scale)
    arrow(1.55, 3.39, 1.55, 3.19, NAVY, lw=1.1, mscale=11.5)

    # View A: Tabular features
    block(0.33, 2.30, 2.44, 0.86, "Tabular View (145 Features)", NAVY, "#FFFFFF")
    ax.text(0.44, 2.82, "• Lithology interaction ratios", fontsize=6.3, color=DARK)
    ax.text(0.44, 2.64, "• Mineral cutoffs & extreme points", fontsize=6.3, color=DARK)
    ax.text(0.44, 2.46, "• Multi-scale depth lags & differences", fontsize=6.3, color=DARK)

    # Arrow from Tabular to Sequence
    arrow(1.55, 2.27, 1.55, 2.11, TEAL, lw=0.95, mscale=10.0)

    # View B: Sequence tensor
    block(0.33, 1.28, 2.44, 0.80, "Sequence View (32 Channels)", TEAL, "#FFFFFF")
    ax.text(0.44, 1.74, "• Continuous depth tensor [B, L, 32]", fontsize=6.3, color=DARK)
    ax.text(0.44, 1.56, "• Missing-sensor validity masks", fontsize=6.3, color=DARK)
    ax.text(0.44, 1.38, "• Stratigraphic depth context", fontsize=6.3, color=DARK)

    # Bottom summary tag (concise text with generous right margin)
    block(0.33, 0.45, 2.44, 0.72, "Stratigraphic Decoupling", GREY, TINT_GREY)
    ax.text(0.44, 0.84, "• Point mineralogy decoupled from strata", fontsize=6.2, color=DARK)
    ax.text(0.44, 0.65, "• Unifies tabular & sequence priors", fontsize=6.2, color=NAVY, fontweight="semibold")

    # =========================================================================
    # STAGE 2: DUAL BACKBONES (x: 3.25 -> 6.05, w: 2.80)
    # =========================================================================
    card(3.25, 0.35, 2.80, 4.15, "Dual Backbones", "2", TEAL)

    # Stream A: Tree Engine (Top)
    block(3.38, 2.30, 2.54, 1.70, "Stream A: Tree Backbone", NAVY, TINT_NAVY)
    ax.text(3.50, 3.65, "CatBoost & LightGBM Ensembles", fontsize=6.6, fontweight="bold", color=NAVY)
    ax.text(3.50, 3.42, "• Direct mapping on 145 tabular features", fontsize=6.3, color=DARK)
    ax.text(3.50, 3.20, "• Quantile loss for predictive uncertainty", fontsize=6.3, color=DARK)
    ax.text(3.50, 2.98, "• Excels on non-default pay zones", fontsize=6.3, fontweight="semibold", color="#1F5B42")
    ax.text(3.50, 2.68, r"• Output: $p_{\mathrm{tree}}(y \mid x)$ and gate $g_{\mathrm{tree}}$", fontsize=6.5, color=NAVY)

    # Stream B: Sequence Engine (Bottom)
    block(3.38, 0.45, 2.54, 1.70, "Stream B: Sequence Backbone", TEAL, TINT_TEAL)
    ax.text(3.50, 1.80, "WellNet / ResU-Net Deep Networks", fontsize=6.6, fontweight="bold", color=TEAL)
    ax.text(3.50, 1.57, "• 1D depthwise multiscale convolutions", fontsize=6.3, color=DARK)
    ax.text(3.50, 1.35, "• Global context & boundary attention", fontsize=6.3, color=DARK)
    ax.text(3.50, 1.13, "• Excels on baseline default cutoffs", fontsize=6.3, fontweight="semibold", color="#1F5B42")
    ax.text(3.50, 0.83, r"• Output: $p_{\mathrm{seq}}(y \mid x)$ and gate $g_{\mathrm{seq}}$", fontsize=6.5, color=TEAL)

    # Straight horizontal connection arrows from Stage 1 into Stage 2 (stopping cleanly 0.04 before Stage 2)
    arrow(2.80, 2.73, 3.34, 2.73, NAVY, lw=1.1, mscale=9.5)
    arrow(2.80, 1.68, 3.34, 1.68, TEAL, lw=1.1, mscale=9.5)

    # =========================================================================
    # STAGE 3: PROBABILISTIC FUSION (x: 6.40 -> 9.35, w: 2.95)
    # =========================================================================
    card(6.40, 0.35, 2.95, 4.15, "Probabilistic Fusion", "3", PURPLE)

    # Stream A State Block (Top)
    block(6.53, 2.96, 2.69, 1.04, "Stream A: Tabular State", NAVY, "#FFFFFF")
    ax.text(6.65, 3.65, r"• Gate Logit: $g_{\mathrm{tree}} \in [0, 1]$", fontsize=6.4, color=DARK)
    ax.text(6.65, 3.43, r"• Quantile Density: $p_{\mathrm{tree}}(y \mid x)$", fontsize=6.4, color=DARK)
    ax.text(6.65, 3.21, "• Sharp pay boundaries & mineral peaks", fontsize=6.2, color="#4A5660")

    # Stream B State Block (Middle)
    block(6.53, 1.70, 2.69, 1.04, "Stream B: Sequence State", TEAL, "#FFFFFF")
    ax.text(6.65, 2.39, r"• Gate Logit: $g_{\mathrm{seq}} \in [0, 1]$", fontsize=6.4, color=DARK)
    ax.text(6.65, 2.17, r"• Gaussian Mixture: $p_{\mathrm{seq}}(y \mid x)$", fontsize=6.4, color=DARK)
    ax.text(6.65, 1.95, "• Vertical stratigraphic continuity", fontsize=6.2, color="#4A5660")

    # Hierarchical Convex Fusion Node (Bottom)
    block(6.53, 0.45, 2.69, 1.03, "Hierarchical Distribution Fusion", PURPLE, TINT_PURPLE)
    eq_patch = FancyBboxPatch((6.63, 0.85), 2.49, 0.32, boxstyle="round,pad=0.01",
                              linewidth=0.75, edgecolor=PURPLE, facecolor="white")
    ax.add_patch(eq_patch)
    ax.text(6.63 + 2.49/2, 1.01, r"$p_{\mathrm{fused}}(y) = \alpha \cdot p_{\mathrm{tree}} + (1-\alpha) \cdot p_{\mathrm{seq}}$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=PURPLE)
    ax.text(6.65, 0.65, r"• Gate Fusion: $g^* = \sigma(w_t g_{\mathrm{tree}} + w_s g_{\mathrm{seq}})$", fontsize=6.4, color=TEAL)

    # Connections from Stage 2 into Stage 3 (stopping cleanly at 6.49 before 6.53)
    arrow(5.95, 3.48, 6.49, 3.48, NAVY, lw=1.1, mscale=9.5)
    arrow(5.95, 2.22, 6.49, 2.22, TEAL, lw=1.1, mscale=9.5)

    # Internal vertical arrows inside Stage 3 into blocks (stopping cleanly before target blocks)
    arrow(7.87, 2.93, 7.87, 2.78, NAVY, lw=0.95, mscale=9.0)
    arrow(7.87, 1.67, 7.87, 1.52, TEAL, lw=0.95, mscale=9.0)

    # =========================================================================
    # STAGE 4: METRIC-ALIGNED DECODING (x: 9.70 -> 12.80, w: 3.10)
    # =========================================================================
    card(9.70, 0.35, 3.10, 4.15, "Metric-Aligned Decoding", "4", RED)

    # Upper: Loss Kernel comparison
    block(9.83, 3.08, 2.84, 0.92, "1. Industrial Utility Metric", DARK, "#FFFFFF")
    eq_u = FancyBboxPatch((9.95, 3.36), 2.60, 0.30, boxstyle="round,pad=0.01",
                          linewidth=0.7, edgecolor=RED, facecolor="#FDF4F3")
    ax.add_patch(eq_u)
    ax.text(9.95 + 2.60/2, 3.51, r"$U(a, y) = \max\left(0, \, 1 - \frac{|a - y|}{\delta}\right)$",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=RED)
    ax.text(9.95, 3.20, r"• Metric tolerance band: zero utility outside $\pm\delta$", fontsize=6.2, color="#4A5660")

    # Middle: Action Decoders
    block(9.83, 1.66, 2.84, 1.30, "2. Dual Action Decoders", NAVY, TINT_NAVY)
    ax.text(9.95, 2.64, "RMSE Benchmark (Conditional Mean):", fontsize=6.4, fontweight="bold", color=DARK)
    ax.text(9.95 + 2.60/2, 2.46, r"$\hat{a}_{\mathrm{RMSE}} = \mathbb{E}[y] = \int y \, p_{\mathrm{fused}}(y) \, dy$",
            ha="center", va="center", fontsize=6.6, color=NAVY)
    ax.text(9.95, 2.24, "Industrial Utility Maximizer:", fontsize=6.4, fontweight="bold", color=DARK)
    ax.text(9.95 + 2.60/2, 2.04, r"$\hat{a}^* = \arg\max_{a} \, \mathbb{E}_{y \sim p_{\mathrm{fused}}}[U(a, y)]$",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=RED)
    ax.text(9.95, 1.80, "• Utility-optimal action resolves asymmetry", fontsize=6.1, color="#4A5660")

    # Bottom: Final Petrophysical Predictions
    block(9.83, 0.45, 2.84, 1.08, "3. Final Petrophysical Targets", PURPLE, "#FFFFFF")
    targets = [
        ("POR", "Porosity (0 – 40%)", NAVY, TINT_NAVY),
        ("PERM", "Permeability (mD, log)", TEAL, TINT_TEAL),
        ("SW", "Water Saturation (0 – 100%)", PURPLE, TINT_PURPLE),
    ]
    for i, (tname, tdesc, tcol, tbg) in enumerate(targets):
        ty = 1.18 - i * 0.26
        ax.add_patch(FancyBboxPatch((9.98, ty - 0.08), 0.65, 0.20, boxstyle="round,pad=0.01",
                                    linewidth=0.75, edgecolor=tcol, facecolor=tbg))
        ax.text(10.30, ty + 0.02, tname, ha="center", va="center", fontsize=6.4, fontweight="bold", color=tcol)
        ax.text(10.75, ty + 0.02, tdesc, ha="left", va="center", fontsize=6.3, color=DARK)

    # Clean continuous orthogonal connection from Stage 3 Fusion to Stage 4 Decoding (stopping at 9.79 before 9.83)
    ax.plot([9.25, 9.55, 9.55, 9.79], [0.96, 0.96, 2.25, 2.25], color=PURPLE, lw=1.1,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    arrow(9.65, 2.25, 9.79, 2.25, PURPLE, lw=1.1, mscale=9.5)

    # Internal arrow in Stage 4 from Decoder down to Predictions (stopping before 1.53 at 1.56)
    arrow(11.25, 1.63, 11.25, 1.56, DARK, lw=0.9, mscale=8.5)

    # Bottom single unified caption bar
    ax.text(6.50, 0.12,
            "Workflow summary: Multi-view inputs feed dual backbones; outputs are fused on the probability simplex; metric-aligned decoding yields optimal scalar parameters.",
            ha="center", va="center", fontsize=7.8, fontweight="semibold", color="#3E4C59")

    # Save
    fig.savefig(OUT / "fig_method_overview.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_method_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    meta = {
        "source": "main_v3.tex Sections 4.1--4.7; clean academic 4-stage methodology pipeline",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / "fig_method_overview.pdf", OUT / "fig_method_overview.png"]},
    }
    (OUT / "fig_method_overview_data.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("fig_method_overview generated successfully.")


if __name__ == "__main__":
    create_figure()
