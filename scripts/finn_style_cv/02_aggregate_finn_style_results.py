from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = []

for fold in range(5):
    p = ROOT / f"docs/brats_eval/finn_style_fold{fold}_finnstyle_t500/brats_ddpm3denc_finnstyle_summary.csv"
    if not p.exists():
        rows.append({"fold": fold, "status": "missing"})
        continue

    df = pd.read_csv(p)
    rec = df.iloc[0].to_dict()
    rec["fold"] = fold
    rec["status"] = "ok"
    rows.append(rec)

out = pd.DataFrame(rows)

outdir = ROOT / "docs/tables"
outdir.mkdir(parents=True, exist_ok=True)

per_fold_path = outdir / "finn_style_gold700_braTS_t500_per_fold_summary.csv"
mean_std_path = outdir / "finn_style_gold700_braTS_t500_mean_std.csv"

out.to_csv(per_fold_path, index=False)

ok = out[out["status"] == "ok"].copy()

summary = {
    "n_folds_completed": len(ok),
    "mean_auprc": ok["mean_auprc"].mean() if len(ok) else np.nan,
    "std_auprc": ok["mean_auprc"].std() if len(ok) > 1 else np.nan,
    "mean_oracle_dice": ok["mean_best_dice_oracle"].mean() if len(ok) else np.nan,
    "std_oracle_dice": ok["mean_best_dice_oracle"].std() if len(ok) > 1 else np.nan,
    "mean_auprc_percent": 100 * ok["mean_auprc"].mean() if len(ok) else np.nan,
    "std_auprc_percent": 100 * ok["mean_auprc"].std() if len(ok) > 1 else np.nan,
    "mean_oracle_dice_percent": 100 * ok["mean_best_dice_oracle"].mean() if len(ok) else np.nan,
    "std_oracle_dice_percent": 100 * ok["mean_best_dice_oracle"].std() if len(ok) > 1 else np.nan,
}

pd.DataFrame([summary]).to_csv(mean_std_path, index=False)

print("\nPer-fold:")
print(out.to_string(index=False))

print("\nMean ± SD:")
print(pd.DataFrame([summary]).to_string(index=False))

print("\nSaved:")
print(per_fold_path)
print(mean_std_path)
