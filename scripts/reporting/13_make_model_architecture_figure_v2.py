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

def add_box(ax, x, y, w, h, text, fs=8.8, bold=False):
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

def add_arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.10,
            linestyle="--" if dashed else "-",
            color=GRAY if dashed else BLACK,
            shrinkA=5,
            shrinkB=5,
        )
    )

def main():
    fig = plt.figure(figsize=(13.33, 7.5), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.045, 0.955,
        "3D-context conditioned diffusion model",
        fontsize=24,
        weight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )
    ax.text(
        0.045, 0.912,
        "A 3D encoder summarizes healthy-volume context, which conditions 2D DDPM slice reconstruction.",
        fontsize=10.5,
        ha="left",
        va="top",
        color=GRAY,
    )

    add_panel(ax, 0.040, 0.545, 0.920, 0.305)
    add_panel(ax, 0.040, 0.145, 0.920, 0.340)

    # ------------------------------------------------------------------
    # Panel A: pretraining
    # ------------------------------------------------------------------
    ax.text(
        0.060, 0.810,
        "A  Stage 1: self-supervised 3D context encoder pretraining",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        color=BLACK,
    )

    y = 0.665
    h = 0.105
    boxes_a = [
        (0.080, y, 0.165, h, "Healthy T2\nvolumes\n96×96×50"),
        (0.305, y, 0.205, h, "SparK-style\nmasked reconstruction\npretraining"),
        (0.570, y, 0.175, h, "3D MONAI\nResNet-50\nencoder"),
        (0.805, y, 0.120, h, "Pretrained\nencoder\nweights"),
    ]

    for i, (x, yy, w, hh, text) in enumerate(boxes_a):
        add_box(ax, x, yy, w, hh, text, fs=8.7, bold=(i == len(boxes_a) - 1))

    for i in range(len(boxes_a) - 1):
        x, yy, w, hh, _ = boxes_a[i]
        nx, ny, nw, nh, _ = boxes_a[i + 1]
        add_arrow(ax, x + w, yy + hh / 2, nx, ny + nh / 2)

    ax.text(
        0.080, 0.600,
        "Goal: learn healthy 3D anatomical context before conditioning the diffusion model.",
        fontsize=9.0,
        color=GRAY,
        ha="left",
        va="center",
    )

    # ------------------------------------------------------------------
    # Panel B: conditioned DDPM
    # ------------------------------------------------------------------
    ax.text(
        0.060, 0.445,
        "B  Stage 2: 3D-context conditioned 2D DDPM reconstruction",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        color=BLACK,
    )

    # Top context branch
    top_y = 0.330
    bot_y = 0.210
    h2 = 0.085

    top_boxes = [
        (0.090, top_y, 0.145, h2, "Full 3D\nT2 volume"),
        (0.295, top_y, 0.185, h2, "3D encoder\n+ global pooling\nfine-tuned"),
        (0.545, top_y, 0.170, h2, "128-d context\nvector"),
    ]

    for i, (x, yy, w, hh, text) in enumerate(top_boxes):
        add_box(ax, x, yy, w, hh, text, fs=8.4, bold=(i == len(top_boxes) - 1))

    for i in range(len(top_boxes) - 1):
        x, yy, w, hh, _ = top_boxes[i]
        nx, ny, nw, nh, _ = top_boxes[i + 1]
        add_arrow(ax, x + w, yy + hh / 2, nx, ny + nh / 2)

    # Bottom reconstruction branch
    bottom_boxes = [
        (0.090, bot_y, 0.145, h2, "2D slice\n+ diffusion\nnoise"),
        (0.295, bot_y, 0.185, h2, "2D conditioned\nDDPM U-Net"),
        (0.545, bot_y, 0.170, h2, "Pseudo-healthy\nreconstruction"),
        (0.775, bot_y, 0.145, h2, "Residual map\n|input − reco|"),
    ]

    for i, (x, yy, w, hh, text) in enumerate(bottom_boxes):
        add_box(ax, x, yy, w, hh, text, fs=8.4, bold=(i == len(bottom_boxes) - 1))

    for i in range(len(bottom_boxes) - 1):
        x, yy, w, hh, _ = bottom_boxes[i]
        nx, ny, nw, nh, _ = bottom_boxes[i + 1]
        add_arrow(ax, x + w, yy + hh / 2, nx, ny + nh / 2)

    # Clean conditioning arrow: context vector down to U-Net
    context_x, context_y, context_w, context_h, _ = top_boxes[-1]
    unet_x, unet_y, unet_w, unet_h, _ = bottom_boxes[1]

    add_arrow(
        ax,
        context_x + context_w / 2,
        context_y,
        unet_x + unet_w / 2,
        unet_y + unet_h,
        dashed=True,
    )

    ax.text(
        0.755,
        0.352,
        "Conditioning signal:\nscale-shift / FiLM-like",
        fontsize=8.6,
        color=GRAY,
        ha="left",
        va="center",
        linespacing=1.2,
    )

    ax.text(
        0.090, 0.165,
        "At inference, tumour-containing slices are reconstructed as healthy; residuals form the anomaly map.",
        fontsize=9.0,
        color=GRAY,
        ha="left",
        va="center",
    )

    out_png = FIGDIR / "02_model_architecture.png"
    out_svg = FIGDIR / "02_model_architecture.svg"
    pres_png = PRES / "02_model_architecture.png"
    pres_svg = PRES / "02_model_architecture.svg"

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
