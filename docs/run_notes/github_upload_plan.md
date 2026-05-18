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

If they contain `PROJECT_ROOT`, either keep them as internal run artifacts or create sanitized relative-path copies before public release.

Do not upload image data itself.

## Safe staging workflow

Use explicit staging only.

Recommended dry-run first:

    git status --short
    git add -n README.md MODEL_CARD.md scripts configs docs audit_reports
    git diff --cached --stat
    git diff --cached --name-only

Only commit after confirming that no checkpoints, NIfTI files, archives, raw data, or huge logs are staged.
