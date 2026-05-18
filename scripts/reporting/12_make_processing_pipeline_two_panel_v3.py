from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs/figures"
PRES = ROOT / "docs/presentation_assets"
FIGDIR.mkdir(parents=True, exist_ok=True)
PRES.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
GRAY = "#555555"
LIGHT = "#F8F8F8"
WHITE = "#FFFFFF"
EDGE = "#222222"
PANEL_EDGE = "#D8D8D8"

def add_panel(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            linewidth=1.0,
            edgecolor=PANEL_EDGE,
            facecolor=WHITE,
        )
    )

def add_box(ax, x, y, w, h, text, fs=8.9, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.05,
            edgecolor=EDGE,
            facecolor=LIGHT,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight="bold" if bold else "normal",
        color=BLACK,
        linespacing=1.15,
    )

def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.15,
            color=BLACK,
            shrinkA=5,
            shrinkB=5,
        )
    )

def add_row(ax, labels, y, x0, x1, h, fs=8.6, bold_last=True):
    n = len(labels)
    gap = 0.018
    w = (x1 - x0 - gap * (n - 1)) / n
    xs = [x0 + i * (w + gap) for i in range(n)]

    for i, (x, label) in enumerate(zip(xs, labels)):
        add_box(ax, x, y, w, h, label, fs=fs, bold=(bold_last and i == n - 1))
        if i < n - 1:
            add_arrow(ax, x + w, y + h / 2, xs[i + 1], y + h / 2)

def main():
    fig = plt.figure(figsize=(13.33, 7.5), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.045, 0.955,
        "Processing and evaluation pipeline",
        fontsize=24,
        weight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )
    ax.text(
        0.045, 0.912,
        "Pre-processing and residual-map post-processing adapted from Behrendt et al.; implemented for our healthy T2 and BraTS21 T2 workflow.",
        fontsize=10.4,
        ha="left",
        va="top",
        color=GRAY,
    )

    # Panels
    add_panel(ax, 0.040, 0.505, 0.920, 0.355)
    add_panel(ax, 0.040, 0.145, 0.920, 0.285)

    # ---------------- Panel A ----------------
    ax.text(
        0.060, 0.820,
        "A  Pre-processing and model-space preparation",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        color=BLACK,
    )

    healthy_labels = [
        "Curated healthy\nT2 cohort",
        "Resample 1.0 mm\n+ atlas registration",
        "Skull strip\n+ brain mask",
        "N4 bias-field\ncorrection",
        "Standardize size\nand crop/pad",
        "Final tensor\n96×96×50",
    ]

    brats_labels = [
        "BraTS21 T2\n+ tumour mask",
        "Brain-mask\ncrop",
        "Pad/truncate\n192×192×160",
        "Downsample ×2\nkeep z=15:65",
        "Shape-checked\nprepared cases",
        "Final tensor\n96×96×50",
    ]

    add_row(
        ax,
        healthy_labels,
        y=0.690,
        x0=0.070,
        x1=0.930,
        h=0.090,
        fs=8.4,
        bold_last=True,
    )

    add_row(
        ax,
        brats_labels,
        y=0.555,
        x0=0.070,
        x1=0.930,
        h=0.090,
        fs=8.4,
        bold_last=True,
    )

    ax.text(
        0.070, 0.520,
        "Healthy scans undergo full preprocessing. BraTS21 cases are prepared and shape-checked into the same model-space tensor geometry.",
        fontsize=8.8,
        color=GRAY,
        ha="left",
        va="center",
    )

    # ---------------- Panel B ----------------
    ax.text(
        0.060, 0.395,
        "B  Inference, residual map, and evaluation",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        color=BLACK,
    )

    inference_labels = [
        "Prepared tensor\n96×96×50",
        "3D-context\nconditioned\n2D DDPM",
        "Pseudo-healthy\nreconstruction",
        "Residual map\nR = |x₀ − x̂₀|",
        "Median filter\n+ mask erosion",
        "Threshold sweep\n+ component filter",
        "AUPRC\n+\nBest possible Dice",
    ]

    add_row(
        ax,
        inference_labels,
        y=0.255,
        x0=0.070,
        x1=0.930,
        h=0.100,
        fs=8.15,
        bold_last=True,
    )

    ax.text(
        0.070, 0.190,
        "Best possible Dice = best-threshold/oracle Dice from a threshold sweep over residual maps; not fixed-threshold deployment Dice.",
        fontsize=8.8,
        color=GRAY,
        ha="left",
        va="center",
    )

    out_png = FIGDIR / "01_processing_evaluation_pipeline.png"
    out_svg = FIGDIR / "01_processing_evaluation_pipeline.svg"
    pres_png = PRES / "01_processing_evaluation_pipeline.png"
    pres_svg = PRES / "01_processing_evaluation_pipeline.svg"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(pres_png, bbox_inches="tight")
    fig.savefig(pres_svg, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_svg)
    print("Saved:", pres_png)
    print("Saved:", pres_svg)

if __name__ == "__main__":
    main()
