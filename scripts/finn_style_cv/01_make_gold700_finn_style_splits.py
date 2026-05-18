from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Data/splits/finn_style"
AUDIT = ROOT / "audit_reports/finn_style_cv"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

# Reconstruct full 691 Gold_700 table from existing audited 90/10 split.
train_csv = ROOT / "Data/splits/Gold700_train_strat90_seed42.csv"
val_csv = ROOT / "Data/splits/Gold700_val_strat10_seed42.csv"

df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
df = df.drop_duplicates(subset=["img_name"]).reset_index(drop=True)

if len(df) != 691:
    raise RuntimeError(f"Expected 691 unique Gold_700 rows, got {len(df)}")

def infer_source(row):
    s = " ".join(str(row.get(c, "")) for c in ["source", "img_name", "img_path"])
    if "OpenNeuro" in s or "PT030" in s:
        return "OpenNeuro"
    if "HCP" in s or "Wu_Minn" in s or "PT020" in s:
        return "HCP_Wu_Minn"
    if "WAND" in s or "PT033" in s:
        return "WAND"
    return "Unknown"

def infer_sex(row):
    for c in ["sex", "Sex", "gender", "Gender"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            v = str(row[c]).strip().upper()
            if v.startswith("F"):
                return "F"
            if v.startswith("M"):
                return "M"
            return v
    return "Unknown"

df["cv_source_family"] = df.apply(infer_source, axis=1)
df["cv_sex"] = df.apply(infer_sex, axis=1)
df["cv_stratum_sex_source"] = df["cv_sex"] + "__" + df["cv_source_family"]

# Finn proportions:
# IXI total = 560, fixed healthy test = 158
# remaining dev = 402, val per fold = 44
test_n = round(len(df) * 158 / 560)
test_frac = test_n / len(df)

# Prefer sex+source if all strata have enough examples; otherwise source only.
strat_col = "cv_stratum_sex_source"
if df[strat_col].value_counts().min() < 2:
    strat_col = "cv_source_family"

print("Total rows:", len(df))
print("Finn-style fixed healthy test rows:", test_n)
print("Stratification column:", strat_col)
print(df[strat_col].value_counts().to_string())

dev_df, test_df = train_test_split(
    df,
    test_size=test_n,
    random_state=3141,
    shuffle=True,
    stratify=df[strat_col],
)

dev_df = dev_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Validation size follows Finn val/dev ratio: 44 / 402.
val_n = round(len(dev_df) * 44 / 402)

print("\nDevelopment rows:", len(dev_df))
print("Validation rows per fold:", val_n)
print("Training rows per fold:", len(dev_df) - val_n)

test_out = OUT / "Gold700_finnstyle_test.csv"
test_df.to_csv(test_out, index=False)

summary_rows = []
all_fold_val_counts = {}

for fold in range(5):
    fold_strat_col = strat_col
    if dev_df[fold_strat_col].value_counts().min() < 2:
        fold_strat_col = "cv_source_family"

    train_df, val_df = train_test_split(
        dev_df,
        test_size=val_n,
        random_state=3141 + fold,
        shuffle=True,
        stratify=dev_df[fold_strat_col],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_names = set(train_df["img_name"])
    val_names = set(val_df["img_name"])
    test_names = set(test_df["img_name"])

    overlap_train_val = train_names & val_names
    overlap_train_test = train_names & test_names
    overlap_val_test = val_names & test_names

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(f"Overlap detected in fold {fold}")

    train_out = OUT / f"Gold700_finnstyle_train_fold{fold}.csv"
    val_out = OUT / f"Gold700_finnstyle_val_fold{fold}.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    for name in val_df["img_name"]:
        all_fold_val_counts[name] = all_fold_val_counts.get(name, 0) + 1

    summary_rows.append({
        "fold": fold,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "overlap_train_val": len(overlap_train_val),
        "overlap_train_test": len(overlap_train_test),
        "overlap_val_test": len(overlap_val_test),
        "train_csv": str(train_out),
        "val_csv": str(val_out),
        "test_csv": str(test_out),
    })

    print(f"\nFold {fold}")
    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
    print("Val source counts:")
    print(val_df["cv_source_family"].value_counts().to_string())
    print("Val sex counts:")
    print(val_df["cv_sex"].value_counts().to_string())

summary = pd.DataFrame(summary_rows)
summary.to_csv(AUDIT / "gold700_finnstyle_split_summary.csv", index=False)

val_reuse = pd.DataFrame({
    "img_name": list(all_fold_val_counts.keys()),
    "val_count_across_5_folds": list(all_fold_val_counts.values()),
}).sort_values(["val_count_across_5_folds", "img_name"])

val_reuse.to_csv(AUDIT / "gold700_finnstyle_validation_reuse_counts.csv", index=False)

print("\nSaved:")
print(test_out)
print(AUDIT / "gold700_finnstyle_split_summary.csv")
print(AUDIT / "gold700_finnstyle_validation_reuse_counts.csv")

print("\nImportant note:")
print("This emulates Finn's fixed healthy test + 5 train/val fold CSV style.")
print("It is not classical equal 5-fold CV, because validation rows may repeat across folds.")
