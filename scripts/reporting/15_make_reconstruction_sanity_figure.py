from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs/figures"
PRES = ROOT / "docs/presentation_assets"
FIGDIR.mkdir(parents=True, exist_ok=True)
PRES.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
GRAY = "#555555"
BAR = "#6E6E6E"

def load_metrics():
    p = ROOT / "docs/tables/healthy_reconstruction_gold700_t500_mean_std.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing reconstruction summary: {p}")
    df = pd.read_csv(p)
    return df

def main():
    df = load_metrics()

    # Expected rows:
    # mean_l1_brain, mean_psnr_brain, mean_ssim_brain_bbox
    wanted = [
        ("mean_l1_brain", "L1 brain error", "lower is better"),
        ("mean_psnr_brain", "PSNR brain", "higher is better"),
        ("mean_ssim_brain_bbox", "SSIM brain bbox", "higher is better"),
    ]

    vals = []
    sds = []
    labels = []
    notes = []

    for metric, label, note in wanted:
        row = df[df["metric"] == metric]
        if len(row) == 0:
            raise ValueError(f"Metric not found in summary CSV: {metric}")
        vals.append(float(row["mean_across_folds"].iloc[0]))
        sds.append(float(row["std_across_folds"].iloc[0]))
        labels.append(label)
        notes.append(note)

    fig = plt.figure(figsize=(13.33, 7.5), dpi=240)
    fig.patch.set_facecolor("white")

    fig.text(
        0.055, 0.950,
        "Healthy reconstruction sanity check",
        fontsize=24,
        weight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )
    fig.text(
        0.055, 0.905,
        "Gold_700 fixed healthy test set, evaluated at t=500; secondary analysis, not the main endpoint.",
        fontsize=10.5,
        color=GRAY,
        ha="left",
        va="top",
    )

    ax = fig.add_axes([0.095, 0.265, 0.580, 0.500])

    x = np.arange(len(vals))
    bars = ax.bar(
        x,
        vals,
        yerr=sds,
        capsize=5,
        color=BAR,
        edgecolor=BLACK,
        linewidth=0.8,
        width=0.56,
    )

    for b, v, sd in zip(bars, vals, sds):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + max(vals)*0.045,
            f"{v:.3f} ± {sd:.3f}" if v < 1 else f"{v:.2f} ± {sd:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            color=BLACK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("Mean ± SD across 5 folds", fontsize=13.2, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    # Right note box
    ax_note = fig.add_axes([0.715, 0.300, 0.235, 0.430])
    ax_note.set_axis_off()
    ax_note.text(
        0.0, 1.0,
        "How to read this",
        fontsize=12,
        weight="bold",
        color=BLACK,
        ha="left",
        va="top",
    )
    ax_note.text(
        0.0, 0.84,
        "These metrics assess whether the\n"
        "model can reconstruct healthy\n"
        "Gold_700 test scans at t=500.\n\n"
        "They are useful as a sanity check,\n"
        "but are not directly comparable\n"
        "to Behrendt et al. Table 1 because\n"
        "their reconstruction table used a\n"
        "different IXI setup and DDPM\n"
        "ensemble protocol.\n\n"
        "Main endpoint remains BraTS21\n"
        "anomaly localization.",
        fontsize=9.1,
        color=GRAY,
        ha="left",
        va="top",
        linespacing=1.18,
    )

    fig.text(
        0.095,
        0.160,
        "Reported values: L1 brain error 0.3495 ± 0.0263, PSNR brain 8.39 ± 0.46, SSIM brain bbox 0.353 ± 0.059.",
        fontsize=10.5,
        weight="bold",
        color=BLACK,
        ha="left",
        va="bottom",
    )

    fig.text(
        0.095,
        0.110,
        "Use this as a secondary reconstruction-quality sanity check only; do not describe it as model accuracy.",
        fontsize=9.2,
        color=GRAY,
        ha="left",
        va="bottom",
    )

    out_png = FIGDIR / "05_healthy_reconstruction_sanity.png"
    out_svg = FIGDIR / "05_healthy_reconstruction_sanity.svg"
    pres_png = PRES / "05_healthy_reconstruction_sanity.png"
    pres_svg = PRES / "05_healthy_reconstruction_sanity.svg"

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
