from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs/presentation_assets"
TABLE_DIR = ROOT / "docs/tables"
NOTE_DIR = ROOT / "docs/run_notes"

ASSET_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Fold result aggregation
# -----------------------------
fold_rows = []
for fold in range(5):
    p = ROOT / f"docs/brats_eval/finn_style_fold{fold}_finnstyle_t500/brats_ddpm3denc_finnstyle_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        rec = df.iloc[0].to_dict()
        rec["fold"] = fold
        rec["status"] = "complete"
    else:
        rec = {
            "fold": fold,
            "status": "pending",
            "n_total": np.nan,
            "n_ok": np.nan,
            "n_error": np.nan,
            "mean_auprc": np.nan,
            "median_auprc": np.nan,
            "std_auprc": np.nan,
            "mean_best_dice_oracle": np.nan,
            "median_best_dice_oracle": np.nan,
            "std_best_dice_oracle": np.nan,
            "mean_tumour_prevalence_evalmask": np.nan,
        }
    fold_rows.append(rec)

fold_df = pd.DataFrame(fold_rows)
completed = fold_df[fold_df["status"] == "complete"].copy()

fold_table_path = TABLE_DIR / "current_finn_style_fold_results.csv"
fold_df.to_csv(fold_table_path, index=False)

if len(completed):
    result_summary = {
        "n_folds_completed": len(completed),
        "mean_auprc": completed["mean_auprc"].mean(),
        "std_auprc": completed["mean_auprc"].std() if len(completed) > 1 else np.nan,
        "mean_oracle_dice": completed["mean_best_dice_oracle"].mean(),
        "std_oracle_dice": completed["mean_best_dice_oracle"].std() if len(completed) > 1 else np.nan,
        "mean_auprc_percent": 100 * completed["mean_auprc"].mean(),
        "std_auprc_percent": 100 * completed["mean_auprc"].std() if len(completed) > 1 else np.nan,
        "mean_oracle_dice_percent": 100 * completed["mean_best_dice_oracle"].mean(),
        "std_oracle_dice_percent": 100 * completed["mean_best_dice_oracle"].std() if len(completed) > 1 else np.nan,
    }
else:
    result_summary = {
        "n_folds_completed": 0,
        "mean_auprc": np.nan,
        "std_auprc": np.nan,
        "mean_oracle_dice": np.nan,
        "std_oracle_dice": np.nan,
        "mean_auprc_percent": np.nan,
        "std_auprc_percent": np.nan,
        "mean_oracle_dice_percent": np.nan,
        "std_oracle_dice_percent": np.nan,
    }

summary_path = TABLE_DIR / "current_finn_style_mean_std.csv"
pd.DataFrame([result_summary]).to_csv(summary_path, index=False)

# -----------------------------
# 2. Plot current fold performance
# -----------------------------
if len(completed):
    x = completed["fold"].astype(str).tolist()
    auprc = completed["mean_auprc"].to_numpy() * 100
    dice = completed["mean_best_dice_oracle"].to_numpy() * 100

    ind = np.arange(len(x))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ind - width/2, auprc, width, label="AUPRC")
    ax.bar(ind + width/2, dice, width, label="Oracle Dice")
    ax.set_xticks(ind)
    ax.set_xticklabels([f"Fold {v}" for v in x])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("BraTS21 Finn-style Evaluation by Completed Fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "current_fold_performance.png", dpi=200)
    plt.close(fig)

# -----------------------------
# 3. Finn-style split plot
# -----------------------------
split_summary = ROOT / "audit_reports/finn_style_cv/gold700_finnstyle_split_summary.csv"
if split_summary.exists():
    ss = pd.read_csv(split_summary)
    fig, ax = plt.subplots(figsize=(9, 5))
    ind = np.arange(len(ss))
    ax.bar(ind - 0.25, ss["train_rows"], width=0.25, label="Train")
    ax.bar(ind, ss["val_rows"], width=0.25, label="Validation")
    ax.bar(ind + 0.25, ss["test_rows"], width=0.25, label="Fixed healthy test")
    ax.set_xticks(ind)
    ax.set_xticklabels([f"Fold {i}" for i in ss["fold"]])
    ax.set_ylabel("Number of Gold_700 scans")
    ax.set_title("Finn-style Gold_700 Split Sizes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "gold700_finn_style_split_sizes.png", dpi=200)
    plt.close(fig)

# -----------------------------
# 4. Source / sex composition from fixed test + fold 0 train/val
# -----------------------------
train0 = ROOT / "Data/splits/finn_style/Gold700_finnstyle_train_fold0.csv"
val0 = ROOT / "Data/splits/finn_style/Gold700_finnstyle_val_fold0.csv"
test = ROOT / "Data/splits/finn_style/Gold700_finnstyle_test.csv"

def safe_read(p):
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

parts = []
for label, p in [("Fold0 train", train0), ("Fold0 val", val0), ("Fixed healthy test", test)]:
    df = safe_read(p)
    if len(df):
        df = df.copy()
        df["split"] = label
        parts.append(df)

if parts:
    comp = pd.concat(parts, ignore_index=True)

    if "cv_source_family" in comp.columns:
        source_counts = comp.groupby(["split", "cv_source_family"]).size().unstack(fill_value=0)
        source_counts.to_csv(TABLE_DIR / "gold700_source_counts_fold0_and_test.csv")

        fig, ax = plt.subplots(figsize=(9, 5))
        source_counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Number of scans")
        ax.set_title("Gold_700 Source Composition")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ASSET_DIR / "gold700_source_composition.png", dpi=200)
        plt.close(fig)

    if "cv_sex" in comp.columns:
        sex_counts = comp.groupby(["split", "cv_sex"]).size().unstack(fill_value=0)
        sex_counts.to_csv(TABLE_DIR / "gold700_sex_counts_fold0_and_test.csv")

        fig, ax = plt.subplots(figsize=(8, 5))
        sex_counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Number of scans")
        ax.set_title("Gold_700 Sex Composition")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ASSET_DIR / "gold700_sex_composition.png", dpi=200)
        plt.close(fig)

# -----------------------------
# 5. BraTS tumour voxel distribution
# -----------------------------
qa_path = ROOT / "audit_reports/brats_eval/brats21_goldstyle_full_QA.csv"
if qa_path.exists():
    qa = pd.read_csv(qa_path)
    if "seg_voxels" in qa.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(qa["seg_voxels"], bins=40)
        ax.set_xlabel("Tumour voxels in prepared 96×96×50 space")
        ax.set_ylabel("Number of BraTS cases")
        ax.set_title("BraTS21 Tumour Burden Distribution")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ASSET_DIR / "brats21_tumour_voxel_distribution.png", dpi=200)
        plt.close(fig)

        qa["seg_voxels"].describe().to_csv(TABLE_DIR / "brats21_tumour_voxel_stats.csv")

# -----------------------------
# 6. README rewrite
# -----------------------------
n_completed = int(result_summary["n_folds_completed"])
if n_completed:
    auprc_txt = f'{result_summary["mean_auprc_percent"]:.2f}%'
    dice_txt = f'{result_summary["mean_oracle_dice_percent"]:.2f}%'
    if n_completed > 1:
        auprc_txt += f' ± {result_summary["std_auprc_percent"]:.2f}%'
        dice_txt += f' ± {result_summary["std_oracle_dice_percent"]:.2f}%'
else:
    auprc_txt = "pending"
    dice_txt = "pending"

per_fold_lines = []
for _, r in fold_df.iterrows():
    if r["status"] == "complete":
        per_fold_lines.append(
            f'| {int(r["fold"])} | complete | {int(r["n_ok"])}/{int(r["n_total"])} | '
            f'{100*r["mean_auprc"]:.2f}% | {100*r["mean_best_dice_oracle"]:.2f}% |'
        )
    else:
        per_fold_lines.append(f'| {int(r["fold"])} | pending/running | — | — | — |')

readme = f"""# 3D-Context Conditioned DDPM for T2 Brain MRI Anomaly Detection

This repository contains a 3D-context extension/adaptation of a conditioned diffusion model framework for unsupervised anomaly detection in brain MRI.

The model uses a **3D MONAI ResNet-50 encoder with SparK-style masked reconstruction pretraining** to extract volumetric context from a full T2 volume and conditions a **2D DDPM reconstruction model**. Anomaly maps are generated from reconstruction residuals and evaluated on BraTS21 T2 tumour scans.

## Project Summary

- Healthy training cohort: Gold_700 T2 MRI
- Final healthy cohort size: 691 scans
- Main protocol: Finn-style repeated train/validation folds with a fixed held-out healthy test set
- Fixed healthy test set: 195 Gold_700 scans
- Per-fold development split: 442 train / 54 validation scans
- External pathological evaluation: BraTS21 T2 tumour scans
- Prepared BraTS21 cases: 1251
- BraTS21 preprocessing QA: 0 errors over 1251 prepared cases
- Model: 3D MONAI ResNet-50 encoder with SparK-style masked reconstruction pretraining + 2D conditioned DDPM
- Current completed Finn-style folds: {n_completed}/5

## Method Overview

1. Curate healthy Gold_700 T2 volumes and standardize to the model space.
2. Create Finn-style splits: fixed held-out healthy test set plus five train/validation fold CSVs.
3. Pretrain a 3D MONAI ResNet-50 encoder with SparK-style masked reconstruction pretraining on each fold's healthy training set.
4. Fine-tune a 2D conditioned DDPM using the pretrained 3D encoder context.
5. Evaluate each fold's model on the fixed external BraTS21 T2 set.
6. Report AUPRC and best possible Dice using Finn-style post-processing.

## Data Splits

Finn-style split files:

- `Data/splits/finn_style/Gold700_finnstyle_train_fold0.csv`
- `Data/splits/finn_style/Gold700_finnstyle_val_fold0.csv`
- ...
- `Data/splits/finn_style/Gold700_finnstyle_train_fold4.csv`
- `Data/splits/finn_style/Gold700_finnstyle_val_fold4.csv`
- `Data/splits/finn_style/Gold700_finnstyle_test.csv`

Important: the fixed healthy test set is not used for model training or validation.

## BraTS21 Evaluation

BraTS21 is used only as an external pathological evaluation set. It is not mixed into Gold_700 training or validation.

Prepared BraTS21 model-space outputs:

- T2 image: `96×96×50`
- Binary tumour segmentation: `seg > 0`
- Binary brain mask

Evaluation uses:

- residual anomaly map: `|input - reconstruction|`
- fixed test timestep: `t_test = 500`
- Finn-style post-processing:
  - median filtering
  - brain-mask erosion
  - small connected-component filtering
- metrics:
  - AUPRC
  - best possible Dice over threshold sweep

Dice is **best possible Dice**, not fixed-threshold deployment Dice.

## Current Results

Current completed folds: **{n_completed}/5**

| Fold | Status | BraTS cases ok | Mean AUPRC | Mean oracle Dice |
|---:|---|---:|---:|---:|
{chr(10).join(per_fold_lines)}

Current aggregate over completed folds:

- Mean AUPRC: **{auprc_txt}**
- Mean oracle Dice: **{dice_txt}**

These results should be interpreted as interim until all requested folds finish.

## Key Scripts

Split creation:

- `scripts/finn_style_cv/01_make_gold700_finn_style_splits.py`

Fold training/evaluation runner:

- `scripts/finn_style_cv/run_finn_style_one_fold.sh`

Result aggregation:

- `scripts/finn_style_cv/02_aggregate_finn_style_results.py`

BraTS evaluation:

- `scripts/brats_eval/08_eval_brats_ddpm3denc_finnstyle.py`

Reporting assets:

- `scripts/reporting/make_current_report_assets.py`

## Key Configs

- `configs/datamodule/Gold_700_finn_fold0.yaml`
- `configs/datamodule/Gold_700_finn_fold1.yaml`
- `configs/datamodule/Gold_700_finn_fold2.yaml`
- `configs/datamodule/Gold_700_finn_fold3.yaml`
- `configs/datamodule/Gold_700_finn_fold4.yaml`
- `configs/model/Spark_3D.yaml`
- `configs/model/DDPM_2D_3DEnc.yaml`

## Presentation Assets

Generated assets are stored in:

- `docs/presentation_assets/`
- `docs/tables/`

Useful figures include:

- `docs/presentation_assets/current_fold_performance.png`
- `docs/presentation_assets/gold700_finn_style_split_sizes.png`
- `docs/presentation_assets/gold700_source_composition.png`
- `docs/presentation_assets/gold700_sex_composition.png`
- `docs/presentation_assets/brats21_tumour_voxel_distribution.png`

## Reproducibility and Audit

Audit reports and run notes are stored under:

- `audit_reports/`
- `docs/run_notes/`
- `docs/code_snapshots/`
- `docs/reproducibility/`

## Limitations

- This is a 3D-encoder extension/adaptation, not an exact reproduction of the original cDDPM implementation.
- Gold_700 replaces IXI as the healthy training cohort.
- BraTS21 preprocessing was matched to the Gold_700/model-space pipeline and is not guaranteed to be identical to Finn et al.'s preprocessing.
- Dice is best possible Dice over threshold sweep.
- Final claims should use the completed fold aggregate, not a single fold.
"""

(ROOT / "README.md").write_text(readme)

note = f"""# Current Reporting Asset Generation

Generated presentation/reporting assets.

Completed Finn-style folds: {n_completed}/5

Current aggregate:
- Mean AUPRC: {auprc_txt}
- Mean oracle Dice: {dice_txt}

Outputs:
- {fold_table_path}
- {summary_path}
- {ASSET_DIR}
"""

(NOTE_DIR / "current_reporting_assets_summary.md").write_text(note)

print("Saved:")
print(fold_table_path)
print(summary_path)
print(ASSET_DIR)
print(ROOT / "README.md")
print(NOTE_DIR / "current_reporting_assets_summary.md")
