# Script Manifest

This document lists the core scripts used in the final Behrendt et al.-adapted training/evaluation workflow.

## Core final workflow

### `scripts/finn_style_cv/01_make_gold700_finn_style_splits.py`

Creates Behrendt et al.-adapted Gold_700 splits:

- fixed healthy test set
- five train/validation folds
- per-fold 442 train / 54 validation scans
- fixed healthy test set of 195 scans

Outputs:

- `Data/splits/finn_style/Gold700_Behrendt-et-al-adapted_train_fold*.csv`
- `Data/splits/finn_style/Gold700_Behrendt-et-al-adapted_val_fold*.csv`
- `Data/splits/finn_style/Gold700_Behrendt-et-al-adapted_test.csv`
- `audit_reports/finn_style_cv/gold700_Behrendt-et-al-adapted_split_summary.csv`

### `scripts/finn_style_cv/run_finn_style_one_fold.sh`

Runs one full Behrendt et al.-adapted fold:

1. Train Spark_3D encoder.
2. Create no-space `/tmp/cddpm_finn/foldX/encoder_best.ckpt` symlink.
3. Train DDPM_2D_3DEnc using the fold encoder.
4. Create no-space `/tmp/cddpm_finn/foldX/ddpm_best.ckpt` symlink.
5. Evaluate on BraTS21 using Behrendt et al.-adapted post-processing.

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
- Behrendt et al.-adapted post-processing:
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
