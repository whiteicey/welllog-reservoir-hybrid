"""Shared, restrained visual style for the reservoir-property manuscript."""
import matplotlib as mpl

# Unified academic palette for Geodata and AI (Elsevier)
NAVY = "#2F5D8A"      # Primary / Trees / Baseline models / Main structures
BLUE = "#7FA8C9"      # Distribution models / Secondary variants / Context
TEAL = "#4C8C82"      # Joint SW Gate / Sequence networks / Refined heads
PURPLE = "#756A9C"    # Full fusion / Multi-component mixture / 18-stage chain
RED = "#B55245"       # Blind test platform / External records / Highlights
GREY = "#9DA5AF"      # Reference models (RF) / Baseline grids / Auxiliary lines
DARK = "#26323B"      # Text / Primary axes / Labels
PALE = "#EEF2F5"      # Light neutral fill

# Tints for diagram container boxes (clean, subtle pastels)
TINT_NAVY = "#F2F6FA"
TINT_TEAL = "#EFF7F5"
TINT_PURPLE = "#F5F3F9"
TINT_GREY = "#F5F7F8"
TINT_YELLOW = "#FAF5E8"
TINT_BLUE = "#EEF4F9"
BORDER_GREY = "#8A949C"
GRID_COLOR = "#E2E7EC"

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": "#7A8690",
        "axes.linewidth": 0.75,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_GREY)
    ax.spines["bottom"].set_color(BORDER_GREY)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if grid:
        ax.grid(axis=grid, color=GRID_COLOR, linewidth=0.55, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5, width=0.65, color=BORDER_GREY)


def panel_label(ax, label, title=None):
    text = f"({label})" if not title else f"({label}) {title}"
    ax.set_title(text, loc="left", pad=8, fontsize=9.5, fontweight="semibold", color=DARK)
