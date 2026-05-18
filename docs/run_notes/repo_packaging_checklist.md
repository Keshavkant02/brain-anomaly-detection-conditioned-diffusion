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
- model: 3D MONAI ResNet-50 encoder with SparK-style masked reconstruction pretraining + 2D conditioned DDPM
- evaluation: Behrendt et al.-adapted post-processing, `t_test = 500`
- metrics: AUPRC and oracle best Dice

Important reporting rule:

Dice is oracle best Dice over a threshold sweep. Do not describe it as fixed-threshold deployment Dice.
