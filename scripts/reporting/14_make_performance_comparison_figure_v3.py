from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs/figures"
PRES = ROOT / "docs/presentation_assets"
TABLES = ROOT / "docs/tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
PRES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
GRAY = "#555555"
LIGHT = "#F7F7F7"
DICE_COLOR = "#4D4D4D"
AUPRC_COLOR = "#A6A6A6"

def load_ours():
    rows = []
    for f in range(5):
        p = ROOT / f"docs/brats_eval/finn_style_fold{f}_finnstyle_t500/brats_ddpm3denc_finnstyle_summary.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing fold summary: {p}")
        r = pd.read_csv(p).iloc[0].to_dict()
        r["fold"] = f
        rows.append(r)

    df = pd.DataFrame(rows)

    return {
        "dice_mean": df["mean_best_dice_oracle"].mean() * 100,
        "dice_sd": df["mean_best_dice_oracle"].std(ddof=1) * 100,
        "auprc_mean": df["mean_auprc"].mean() * 100,
        "auprc_sd": df["mean_auprc"].std(ddof=1) * 100,
        "fold_dice": df["mean_best_dice_oracle"].to_numpy() * 100,
        "fold_auprc": df["mean_auprc"].to_numpy() * 100,
        "n_ok": int(df["n_ok"].iloc[0]),
        "n_error_total": int(df["n_error"].sum()),
    }, df

def add_callout(fig):
    # Right-side interpretive box.
    ax_note = fig.add_axes([0.685, 0.405, 0.265, 0.315])
    ax_note.set_axis_off()

    patch = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#CCCCCC",
        facecolor="#FAFAFA",
    )
    ax_note.add_patch(patch)

    ax_note.text(
        0.05, 0.86,
        "Interpretation",
        fontsize=11,
        weight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )

    text = (
        "Contextual comparison, not a\n"
        "perfect one-to-one benchmark.\n\n"
        "This work differs by:\n"
        "• Gold_700 healthy T2 cohort\n"
        "• 3D-context encoder extension\n"
        "• single t_test = 500\n"
        "• no t_test ensemble\n\n"
        "Behrendt strongest cDDPM row\n"
        "uses SSL + ENS."
    )

    ax_note.text(
        0.05, 0.74,
        text,
        fontsize=8.35,
        ha="left",
        va="top",
        color=GRAY,
        linespacing=1.18,
    )

def main():
    ours, fold_df = load_ours()

    # Behrendt et al. Table 2, BraTS21 T2 selected DDPM-family rows.
    comp = pd.DataFrame([
        {
            "model": "DDPM\nENS",
            "protocol": "Behrendt et al.",
            "best_possible_dice_mean": 50.27,
            "best_possible_dice_sd": 2.67,
            "auprc_mean": 50.61,
            "auprc_sd": 2.92,
        },
        {
            "model": "pDDPM\nENS",
            "protocol": "Behrendt et al.",
            "best_possible_dice_mean": 53.61,
            "best_possible_dice_sd": 0.51,
            "auprc_mean": 55.08,
            "auprc_sd": 0.54,
        },
        {
            "model": "cDDPM\nSSL+ENS",
            "protocol": "Behrendt et al.",
            "best_possible_dice_mean": 56.30,
            "best_possible_dice_sd": 1.25,
            "auprc_mean": 58.82,
            "auprc_sd": 1.56,
        },
        {
            "model": "Ours\n3D-context\ncDDPM",
            "protocol": "This work",
            "best_possible_dice_mean": ours["dice_mean"],
            "best_possible_dice_sd": ours["dice_sd"],
            "auprc_mean": ours["auprc_mean"],
            "auprc_sd": ours["auprc_sd"],
        },
    ])

    comp.to_csv(TABLES / "brats21_t2_performance_comparison.csv", index=False)
    fold_df.to_csv(TABLES / "brats21_t2_ours_fold_summaries_for_plot.csv", index=False)

    fig = plt.figure(figsize=(13.33, 7.5), dpi=240)
    fig.patch.set_facecolor("white")

    fig.text(
        0.055, 0.950,
        "BraTS21 T2 segmentation performance",
        fontsize=24,
        weight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )
    fig.text(
        0.055, 0.905,
        "Selected DDPM-family rows from Behrendt et al. Table 2 vs. our 3D-context conditioned DDPM.",
        fontsize=10.5,
        ha="left",
        va="top",
        color=GRAY,
    )

    # Main plot, narrowed to leave room for caveat box.
    ax = fig.add_axes([0.075, 0.230, 0.575, 0.575])

    x = np.arange(len(comp))
    width = 0.34

    dice = comp["best_possible_dice_mean"].to_numpy()
    dice_sd = comp["best_possible_dice_sd"].to_numpy()
    auprc = comp["auprc_mean"].to_numpy()
    auprc_sd = comp["auprc_sd"].to_numpy()

    bars1 = ax.bar(
        x - width / 2,
        dice,
        width,
        yerr=dice_sd,
        capsize=4,
        label="Best possible Dice",
        color=DICE_COLOR,
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=3,
    )

    bars2 = ax.bar(
        x + width / 2,
        auprc,
        width,
        yerr=auprc_sd,
        capsize=4,
        label="AUPRC",
        color=AUPRC_COLOR,
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=3,
    )

    ours_idx = len(comp) - 1
    ax.axvspan(ours_idx - 0.52, ours_idx + 0.52, color="#F2F2F2", zorder=0)

    # Overlay fold-level values for our model.
    rng = np.random.default_rng(42)
    jitter_dice = rng.normal(0, 0.015, size=len(ours["fold_dice"]))
    jitter_auprc = rng.normal(0, 0.015, size=len(ours["fold_auprc"]))

    ax.scatter(
        np.full(len(ours["fold_dice"]), ours_idx - width / 2) + jitter_dice,
        ours["fold_dice"],
        s=28,
        color="white",
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=5,
    )
    ax.scatter(
        np.full(len(ours["fold_auprc"]), ours_idx + width / 2) + jitter_auprc,
        ours["fold_auprc"],
        s=28,
        color="white",
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=5,
    )

    # Value labels.
    for bar, val in zip(bars1, dice):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4.0,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=BLACK,
            weight="bold" if bar.get_x() > 2.3 else "normal",
        )

    for bar, val in zip(bars2, auprc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4.0,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=BLACK,
            weight="bold" if bar.get_x() > 2.3 else "normal",
        )

    ax.set_ylim(0, 72)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"], fontsize=9.4)
    ax.set_title("BraTS21 T2: best possible Dice and AUPRC", fontsize=13.2, weight="bold")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    add_callout(fig)

    # Summary band.
    fig.text(
        0.075,
        0.130,
        f"Our result: AUPRC {ours['auprc_mean']:.2f}% ± {ours['auprc_sd']:.2f}%   |   "
        f"Best possible Dice {ours['dice_mean']:.2f}% ± {ours['dice_sd']:.2f}%   "
        f"({ours['n_ok']}/1251 cases per fold; {ours['n_error_total']} failed evaluations)",
        fontsize=11.5,
        weight="bold",
        color=BLACK,
        ha="left",
        va="bottom",
    )

    fig.text(
        0.075,
        0.075,
        "Best possible Dice = threshold-sweep/oracle Dice over residual maps; not fixed-threshold deployment Dice. "
        "Bars show mean ± SD across folds.",
        fontsize=9.1,
        color=GRAY,
        ha="left",
        va="bottom",
    )

    out_png = FIGDIR / "04_brats21_t2_performance_comparison.png"
    out_svg = FIGDIR / "04_brats21_t2_performance_comparison.svg"
    pres_png = PRES / "04_brats21_t2_performance_comparison.png"
    pres_svg = PRES / "04_brats21_t2_performance_comparison.svg"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(pres_png, bbox_inches="tight")
    fig.savefig(pres_svg, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_svg)
    print("Saved:", pres_png)
    print("Saved:", pres_svg)
    print("Saved:", TABLES / "brats21_t2_performance_comparison.csv")
    print("Saved:", TABLES / "brats21_t2_ours_fold_summaries_for_plot.csv")

if __name__ == "__main__":
    main()
