# Final Submission Status

## Experiment status

All five Behrendt et al.-adapted folds completed successfully.

- Folds completed: 5/5
- BraTS21 cases evaluated per fold: 1251/1251
- Failed BraTS evaluations: 0
- Test timestep: t = 500
- Post-processing: Behrendt et al.-adapted post-processing enabled
- Final mean AUPRC: 57.05% ± 4.85%
- Final mean oracle Dice: 60.31% ± 2.64%

## Final per-fold result

| Fold | Mean AUPRC | Mean oracle Dice | BraTS cases ok |
|---:|---:|---:|---:|
| 0 | 60.42% | 61.59% | 1251/1251 |
| 1 | 59.21% | 62.34% | 1251/1251 |
| 2 | 53.37% | 58.72% | 1251/1251 |
| 3 | 61.71% | 62.47% | 1251/1251 |
| 4 | 50.52% | 56.44% | 1251/1251 |

## Reporting wording

Across five Behrendt et al.-adapted folds, the 3D-context conditioned DDPM achieved 57.05% ± 4.85% AUPRC and 60.31% ± 2.64% oracle Dice on 1251 BraTS21 T2 tumour cases, with zero failed evaluations.

Dice is oracle best Dice over a threshold sweep, not fixed-threshold deployment Dice.

## Next packaging steps

- Final GitHub staging audit
- Ensure no image data/checkpoints/archives are staged
- Prepare final presentation/report
- Push clean code, configs, scripts, README, audit summaries, and presentation assets
