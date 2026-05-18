from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/run_notes"
OUT.mkdir(parents=True, exist_ok=True)

github_upload_plan = dedent("""
# GitHub Upload Plan

## Principle

Do not use `git add .`.

This repository should upload code, configs, small CSV summaries, audit notes, and presentation assets. It should not upload raw medical images, preprocessed NIfTI files, BraTS archives, checkpoints, or large training logs.

## Do not upload

- Raw medical image data
- Preprocessed NIfTI files
- BraTS archive files
- Gold_700 image files
- Model checkpoints
- Large Hydra / Lightning logs
- Local scratch logs
- Personal/local machine paths when avoidable

## Excluded by `.gitignore`

- `<path_to_logs>/`
- `checkpoints/`
- `*.ckpt`
- `*.pt`
- `*.pth`
- `*.nii`
- `*.nii.gz`
- `*.tar`
- `*.zip`
- `Data/preprocessed/`
- `Data/brats21_raw/`
- `Data/Test/Brats21/t2/`
- `Data/Test/Brats21/seg/`
- `Data/Test/Brats21/mask/`

## Safe to upload

- Source code
- Model configs
- Datamodule configs
- Training/evaluation scripts
- Split-generation scripts
- Reporting scripts
- Audit summary CSVs
- Result summary CSVs
- Presentation PNGs
- README
- MODEL_CARD
- Reproducibility notes

## Caution on split CSVs

Split CSVs are useful for reproducibility, but check whether they contain absolute local paths.

If generated files contain machine-specific paths, keep them internal or create sanitized relative-path copies before public release.

Do not upload image data itself.

## Safe staging workflow

Use explicit staging only.

Recommended dry-run first:

    git status --short
    git add -n README.md MODEL_CARD.md scripts configs docs audit_reports
    git diff --cached --stat
    git diff --cached --name-only

Only commit after confirming that no checkpoints, NIfTI files, archives, raw data, or huge logs are staged.
""").strip() + "\n"

script_manifest = dedent("""
# Script Manifest

This document lists the core scripts used in the final Finn-style training/evaluation workflow.

## Core final workflow

### `scripts/finn_style_cv/01_make_gold700_finn_style_splits.py`

Creates Finn-style Gold_700 splits:

- fixed healthy test set
- five train/validation folds
- per-fold 442 train / 54 validation scans
- fixed healthy test set of 195 scans

Outputs:

- `Data/splits/finn_style/Gold700_finnstyle_train_fold*.csv`
- `Data/splits/finn_style/Gold700_finnstyle_val_fold*.csv`
- `Data/splits/finn_style/Gold700_finnstyle_test.csv`
- `audit_reports/finn_style_cv/gold700_finnstyle_split_summary.csv`

### `scripts/finn_style_cv/run_finn_style_one_fold.sh`

Runs one full Finn-style fold:

1. Train Spark_3D encoder.
2. Create no-space `/tmp/cddpm_finn/foldX/encoder_best.ckpt` symlink.
3. Train DDPM_2D_3DEnc using the fold encoder.
4. Create no-space `/tmp/cddpm_finn/foldX/ddpm_best.ckpt` symlink.
5. Evaluate on BraTS21 using Finn-style post-processing.

Important fixes:

- avoids Hydra path parsing issues using `/tmp` checkpoint symlinks
- uses `PYTHONPATH="$PWD:${PYTHONPATH:-}"` for BraTS eval import stability

### `scripts/brats_eval/08_eval_brats_ddpm3denc_finnstyle.py`

Evaluates trained DDPM_2D_3DEnc checkpoints on prepared BraTS21 T2 cases.

Evaluation settings:

- `t_test = 500`
- residual score: `|input - reconstruction|`
- AUPRC over evaluation mask
- oracle best Dice over threshold sweep
- Finn-style post-processing:
  - median filtering
  - brain-mask erosion
  - connected-component filtering

Important fix:

- inserts project root into `sys.path` so local `src` imports work

### `scripts/finn_style_cv/02_aggregate_finn_style_results.py`

Aggregates fold-level BraTS summaries into:

- `docs/tables/finn_style_gold700_braTS_t500_per_fold_summary.csv`
- `docs/tables/finn_style_gold700_braTS_t500_mean_std.csv`

### `scripts/reporting/make_current_report_assets.py`

Generates presentation/reporting assets:

- current fold result tables
- current mean/std table
- fold performance PNG
- Gold_700 split-size PNG
- Gold_700 source/sex composition PNGs
- BraTS tumour voxel distribution PNG
- regenerated README reflecting currently completed folds

Rerun this script after additional folds finish.

## BraTS preprocessing scripts

### `scripts/brats_eval/05_prepare_brats_goldstyle_full.py`

Prepared the full BraTS21 T2 evaluation set into Gold-style model space.

### `scripts/brats_eval/04_prepare_brats_goldstyle_onecase.py`

One-case preprocessing sanity script.

### `scripts/brats_eval/03_prepare_one_brats_case_safe.py`

Earlier safe one-case preprocessing check.

## Pilot / development evaluation scripts

These were useful during development but should be described as pilot/sanity scripts, not the final workflow:

- `scripts/brats_eval/06_eval_onecase_ddpm3denc.py`
- `scripts/brats_eval/07_eval_brats_ddpm3denc_full.py`

## Reporting / packaging scripts

### `scripts/reporting/write_packaging_docs.py`

Regenerates packaging documentation cleanly to avoid shell heredoc/Markdown corruption.
""").strip() + "\n"

repo_packaging_checklist = dedent("""
# Repo Packaging Checklist

## Before GitHub upload

- [ ] Fold run has finished or current interim results are clearly labeled.
- [ ] `README.md` regenerated with `scripts/reporting/make_current_report_assets.py`.
- [ ] `docs/run_notes/github_upload_plan.md` regenerated with `scripts/reporting/write_packaging_docs.py`.
- [ ] `docs/run_notes/script_manifest.md` regenerated with `scripts/reporting/write_packaging_docs.py`.
- [ ] No raw NIfTI files are staged.
- [ ] No checkpoints are staged.
- [ ] No `.tar` or `.zip` archives are staged.
- [ ] No large Hydra/Lightning logs are staged.
- [ ] No private/local absolute paths are presented as portable public paths unless clearly documented.
- [ ] `git add -n ...` dry run reviewed.
- [ ] `git diff --cached --name-only` reviewed before commit.

## Final project story

This project is a 3D-context extension/adaptation of a conditioned DDPM anomaly-detection framework.

Main protocol:

- healthy training data: Gold_700 T2 MRI
- external pathological evaluation: BraTS21 T2 tumour scans
- model: 3D Spark encoder + 2D conditioned DDPM
- evaluation: Finn-style post-processing, `t_test = 500`
- metrics: AUPRC and oracle best Dice

Important reporting rule:

Dice is oracle best Dice over a threshold sweep. Do not describe it as fixed-threshold deployment Dice.
""").strip() + "\n"

files = {
    "github_upload_plan.md": github_upload_plan,
    "script_manifest.md": script_manifest,
    "repo_packaging_checklist.md": repo_packaging_checklist,
}

for name, text in files.items():
    p = OUT / name
    p.write_text(text)
    print("wrote", p)

print("Done.")
