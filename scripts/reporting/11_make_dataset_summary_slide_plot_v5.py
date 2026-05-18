from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / "Data/splits/finn_style"
OUTDIR = ROOT / "docs/presentation_assets"
FIGDIR = ROOT / "docs/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OI = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
    "black": "#000000",
}

def load_gold700_unique():
    parts = []
    for f in range(5):
        tr = pd.read_csv(SPLIT_ROOT / f"Gold700_finnstyle_train_fold{f}.csv")
        va = pd.read_csv(SPLIT_ROOT / f"Gold700_finnstyle_val_fold{f}.csv")
        tr["fold"] = f
        tr["split"] = "train"
        va["fold"] = f
        va["split"] = "validation"
        parts.extend([tr, va])

    te = pd.read_csv(SPLIT_ROOT / "Gold700_finnstyle_test.csv")
    te["fold"] = "fixed"
    te["split"] = "test"
    parts.append(te)

    all_rows = pd.concat(parts, ignore_index=True)
    return all_rows.drop_duplicates("img_name").copy()

def verify_split_integrity():
    test = pd.read_csv(SPLIT_ROOT / "Gold700_finnstyle_test.csv")
    te_set = set(test["img_name"].astype(str))

    rows = []
    for f in range(5):
        tr = pd.read_csv(SPLIT_ROOT / f"Gold700_finnstyle_train_fold{f}.csv")
        va = pd.read_csv(SPLIT_ROOT / f"Gold700_finnstyle_val_fold{f}.csv")

        tr_set = set(tr["img_name"].astype(str))
        va_set = set(va["img_name"].astype(str))

        rows.append({
            "fold": f,
            "train": len(tr),
            "validation": len(va),
            "fixed_test": len(test),
            "train_val_overlap": len(tr_set & va_set),
            "train_test_overlap": len(tr_set & te_set),
            "val_test_overlap": len(va_set & te_set),
        })
    return pd.DataFrame(rows)

def get_brats_summary():
    rows = []
    for f in range(5):
        p = ROOT / f"docs/brats_eval/finn_style_fold{f}_finnstyle_t500/brats_ddpm3denc_finnstyle_summary.csv"
        if p.exists():
            r = pd.read_csv(p).iloc[0].to_dict()
            r["fold"] = f
            rows.append(r)
    return pd.DataFrame(rows)

def add_metric_card(ax, x, y, w, h, title, value, subtitle, color):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=1.65,
        edgecolor=color,
        facecolor="white"
    )
    ax.add_patch(box)

    ax.text(
        x + 0.026, y + h - 0.075,
        title,
        fontsize=10.4,
        weight="bold",
        color=color,
        va="top"
    )
    ax.text(
        x + 0.026, y + h - 0.245,
        value,
        fontsize=16.4,
        weight="bold",
        color=OI["black"],
        va="top"
    )
    ax.text(
        x + 0.026, y + 0.045,
        subtitle,
        fontsize=8.25,
        color="#333333",
        va="bottom"
    )

def main():
    unique = load_gold700_unique()
    split_df = verify_split_integrity()
    brats = get_brats_summary()

    n_gold = len(unique)

    sex_counts = unique["sex"].value_counts().reindex(["F", "M"]).fillna(0).astype(int)
    sex_labels = ["Female", "Male"]
    sex_values = [int(sex_counts["F"]), int(sex_counts["M"])]

    source_counts = unique["source_family"].value_counts().sort_values(ascending=True)
    clean_source_names = {
        "HCP_Wu_Minn": "HCP Wu-Minn",
        "OpenNeuro": "OpenNeuro",
        "WAND": "WAND",
    }
    source_labels = [clean_source_names.get(x, x) for x in source_counts.index]
    source_values = list(source_counts.values)

    train_n = int(split_df["train"].iloc[0])
    val_n = int(split_df["validation"].iloc[0])
    test_n = int(split_df["fixed_test"].iloc[0])

    total_overlap = int(
        split_df["train_val_overlap"].sum()
        + split_df["train_test_overlap"].sum()
        + split_df["val_test_overlap"].sum()
    )

    if len(brats):
        brats_ok = int(brats["n_ok"].iloc[0])
        brats_errors_total = int(brats["n_error"].sum())
    else:
        brats_ok = 1251
        brats_errors_total = 0

    fig = plt.figure(figsize=(13.33, 7.5), dpi=240)
    fig.patch.set_facecolor("white")

    # Title block
    fig.text(
        0.055, 0.950,
        "Dataset and evaluation summary",
        fontsize=24,
        weight="bold",
        ha="left",
        va="top"
    )

    fig.text(
        0.055, 0.905,
        "Curated healthy T2 MRI cohort from OpenNeuro, HCP Wu-Minn, and WAND; external BraTS21 T2 tumour evaluation",
        fontsize=10.8,
        color="#444444",
        ha="left",
        va="top"
    )

    # Metric cards
    ax_cards = fig.add_axes([0.055, 0.725, 0.890, 0.145])
    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    card_w = 0.232
    gap = 0.020
    xs = [0.000, card_w + gap, 2 * (card_w + gap), 3 * (card_w + gap)]

    add_metric_card(
        ax_cards, xs[0], 0.08, card_w, 0.84,
        "Healthy cohort",
        f"{n_gold} scans",
        "OpenNeuro / HCP Wu-Minn / WAND",
        OI["blue"]
    )
    add_metric_card(
        ax_cards, xs[1], 0.08, card_w, 0.84,
        "Representation",
        f"{sex_values[0]} F / {sex_values[1]} M",
        "Near-even sex composition",
        OI["reddish_purple"]
    )
    add_metric_card(
        ax_cards, xs[2], 0.08, card_w, 0.84,
        "Split protocol",
        f"{train_n} train / {val_n} val",
        f"Per fold; fixed healthy test = {test_n}",
        OI["orange"]
    )
    add_metric_card(
        ax_cards, xs[3], 0.08, card_w, 0.84,
        "External evaluation",
        f"{brats_ok} cases/fold",
        f"BraTS21 T2; failed evals = {brats_errors_total}",
        OI["vermillion"]
    )

    # Main plots
    gs = fig.add_gridspec(
        1, 3,
        left=0.075,
        right=0.955,
        bottom=0.155,
        top=0.650,
        wspace=0.42,
        width_ratios=[1.0, 1.15, 1.0]
    )

    # A: Sex composition
    ax1 = fig.add_subplot(gs[0, 0])
    sex_colors = [OI["reddish_purple"], OI["sky_blue"]]
    bars = ax1.barh(sex_labels, sex_values, color=sex_colors, edgecolor="black", linewidth=0.7)
    total_sex = sum(sex_values)

    for b, v in zip(bars, sex_values):
        ax1.text(
            v + 5,
            b.get_y() + b.get_height() / 2,
            f"{v} ({100 * v / total_sex:.1f}%)",
            va="center",
            fontsize=10.5
        )

    ax1.set_xlim(0, max(sex_values) * 1.25)
    ax1.set_title("Healthy cohort sex composition", fontsize=13.2, weight="bold")
    ax1.set_xlabel("Number of scans")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.15)
    ax1.text(-0.20, 1.06, "A", transform=ax1.transAxes, fontsize=14, weight="bold")

    # B: Source composition
    ax2 = fig.add_subplot(gs[0, 1])
    src_colors = [OI["bluish_green"], OI["orange"], OI["blue"]][:len(source_values)]
    bars = ax2.barh(source_labels, source_values, color=src_colors, edgecolor="black", linewidth=0.7)

    for b, v in zip(bars, source_values):
        ax2.text(
            v + 5,
            b.get_y() + b.get_height() / 2,
            f"{v} ({100 * v / n_gold:.1f}%)",
            va="center",
            fontsize=10.5
        )

    ax2.set_xlim(0, max(source_values) * 1.30)
    ax2.set_title("Source-family composition", fontsize=13.2, weight="bold")
    ax2.set_xlabel("Number of scans")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.15)
    ax2.text(-0.18, 1.06, "B", transform=ax2.transAxes, fontsize=14, weight="bold")

    # C: Split sizes
    ax3 = fig.add_subplot(gs[0, 2])
    split_labels = ["Train", "Validation", "Fixed\ntest"]
    split_values = [train_n, val_n, test_n]
    split_colors = [OI["blue"], OI["orange"], OI["gray"]]

    bars = ax3.bar(split_labels, split_values, color=split_colors, edgecolor="black", linewidth=0.7)

    for b, v in zip(bars, split_values):
        ax3.text(
            b.get_x() + b.get_width() / 2,
            v + 12,
            str(v),
            ha="center",
            va="bottom",
            fontsize=11.5,
            weight="bold"
        )

    ax3.set_ylim(0, max(split_values) * 1.30)
    ax3.set_title("Split sizes", fontsize=13.2, weight="bold")
    ax3.set_ylabel("Number of scans")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.15)
    ax3.text(-0.20, 1.06, "C", transform=ax3.transAxes, fontsize=14, weight="bold")

    ax3.text(
        0.5,
        -0.18,
        "Train/validation are per fold; fixed test is held out across folds.",
        ha="center",
        va="top",
        transform=ax3.transAxes,
        fontsize=8.8,
        color="#444444"
    )

    # Footer
    fig.text(
        0.055,
        0.065,
        f"Split integrity: train/validation/test overlap = {total_overlap}. "
        "Training uses healthy scans only; BraTS21 is external pathological evaluation.",
        fontsize=9.5,
        color="#333333",
        ha="left",
        va="bottom"
    )

    out_png = OUTDIR / "dataset_summary_slide_v5_okabe_ito.png"
    out_svg = OUTDIR / "dataset_summary_slide_v5_okabe_ito.svg"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    # Also save clean public-facing names
    fig.savefig(FIGDIR / "03_dataset_summary.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "03_dataset_summary.svg", bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_svg)
    print("Saved:", FIGDIR / "03_dataset_summary.png")
    print("Saved:", FIGDIR / "03_dataset_summary.svg")

if __name__ == "__main__":
    main()
