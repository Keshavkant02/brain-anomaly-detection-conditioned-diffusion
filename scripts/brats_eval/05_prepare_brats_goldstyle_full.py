from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

TARGET_PAD = (192, 192, 160)
FINAL_Z_SLICE = slice(15, 65)

def bbox_from_mask(mask, margin=0):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("empty brain mask")
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + margin, np.array(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))

def gold_standardize_array(arr, is_label=False):
    arr = np.asarray(arr)
    padded = np.zeros(TARGET_PAD, dtype=arr.dtype)

    d0, d1, d2 = arr.shape
    c0, c1, c2 = min(d0, 192), min(d1, 192), min(d2, 160)

    s0 = (192 - c0) // 2
    s1 = (192 - c1) // 2
    s2 = (160 - c2) // 2

    # matches verified Gold_700 standardize_size.py behavior
    padded[s0:s0+c0, s1:s1+c1, s2:s2+c2] = arr[:c0, :c1, :c2]

    order = 0 if is_label else 1
    down = ndimage.zoom(padded, zoom=0.5, order=order)
    return down[:, :, FINAL_Z_SLICE]

def process_one(row, out_root, margin=0, overwrite=False):
    subj = row["subject"]
    t2_path = Path(row["t2_path"])
    seg_path = Path(row["seg_path"])

    t2_out = out_root / "t2" / f"{subj}_t2.nii.gz"
    seg_out = out_root / "seg" / f"{subj}_seg.nii.gz"
    mask_out = out_root / "mask" / f"{subj}_mask.nii.gz"

    if (not overwrite) and t2_out.exists() and seg_out.exists() and mask_out.exists():
        return {
            "subject": subj,
            "status": "skipped_exists",
            "img_path": str(t2_out),
            "mask_path": str(mask_out),
            "seg_path": str(seg_out),
            "error": "",
        }

    try:
        t2_img = nib.load(str(t2_path))
        seg_img = nib.load(str(seg_path))

        t2 = t2_img.get_fdata().astype(np.float32)
        seg = seg_img.get_fdata().astype(np.int16)

        raw_mask = (t2 > 0).astype(np.uint8)
        raw_seg = (seg > 0).astype(np.uint8)

        crop = bbox_from_mask(raw_mask, margin=margin)

        t2_crop = t2[crop]
        mask_crop = raw_mask[crop]
        seg_crop = raw_seg[crop]

        t2_std = gold_standardize_array(t2_crop, is_label=False).astype(np.float32)
        t2_std = np.clip(t2_std, 0, None)

        mask_std = gold_standardize_array(mask_crop, is_label=True).astype(np.uint8)
        mask_std = (mask_std > 0).astype(np.uint8)

        seg_std = gold_standardize_array(seg_crop, is_label=True).astype(np.uint8)
        seg_std = (seg_std > 0).astype(np.uint8)
        seg_std = (seg_std & mask_std).astype(np.uint8)

        if t2_std.shape != (96, 96, 50):
            raise RuntimeError(f"bad t2 shape {t2_std.shape}")
        if seg_std.shape != (96, 96, 50):
            raise RuntimeError(f"bad seg shape {seg_std.shape}")
        if mask_std.shape != (96, 96, 50):
            raise RuntimeError(f"bad mask shape {mask_std.shape}")

        affine = t2_img.affine.copy()
        affine[:3, :3] *= 2

        for d in [out_root / "t2", out_root / "seg", out_root / "mask"]:
            d.mkdir(parents=True, exist_ok=True)

        nib.save(nib.Nifti1Image(t2_std, affine), str(t2_out))
        nib.save(nib.Nifti1Image(seg_std, affine), str(seg_out))
        nib.save(nib.Nifti1Image(mask_std, affine), str(mask_out))

        return {
            "subject": subj,
            "status": "ok",
            "raw_shape": str(t2.shape),
            "crop_shape": str(t2_crop.shape),
            "t2_min": float(t2_std.min()),
            "t2_max": float(t2_std.max()),
            "seg_voxels": int(seg_std.sum()),
            "mask_voxels": int(mask_std.sum()),
            "img_path": str(t2_out),
            "mask_path": str(mask_out),
            "seg_path": str(seg_out),
            "error": "",
        }

    except Exception as e:
        return {
            "subject": subj,
            "status": "error",
            "img_path": "",
            "mask_path": "",
            "seg_path": "",
            "error": repr(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="audit_reports/brats_eval/brats21_raw_t2seg_manifest.csv")
    ap.add_argument("--out-root", default="Data/Test/Brats21")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["has_seg"] == True].copy()
    manifest = manifest.sort_values("subject")

    if args.limit and args.limit > 0:
        manifest = manifest.head(args.limit)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_one, row.to_dict(), out_root, args.margin, args.overwrite)
            for _, row in manifest.iterrows()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rows.append(fut.result())

    results = pd.DataFrame(rows).sort_values("subject")
    Path("audit_reports/brats_eval").mkdir(parents=True, exist_ok=True)
    results_out = Path("audit_reports/brats_eval/brats21_goldstyle_preprocess_results.csv")
    results.to_csv(results_out, index=False)

    ok = results[results["status"].isin(["ok", "skipped_exists"])].copy()
    eval_rows_abs = []
    eval_rows_rel = []

    for _, r in ok.iterrows():
        subj = r["subject"]

        eval_rows_abs.append({
            "img_path": str(Path(r["img_path"]).resolve()),
            "img_name": subj,
            "age": "",
            "label": 1,
            "setname": "Brats21",
            "settype": "test",
            "mask_path": str(Path(r["mask_path"]).resolve()),
            "seg_path": str(Path(r["seg_path"]).resolve()),
        })

        eval_rows_rel.append({
            "img_path": f"/Test/Brats21/t2/{subj}_t2.nii.gz",
            "img_name": subj,
            "age": "",
            "label": 1,
            "setname": "Brats21",
            "settype": "test",
            "mask_path": f"/Test/Brats21/mask/{subj}_mask.nii.gz",
            "seg_path": f"/Test/Brats21/seg/{subj}_seg.nii.gz",
        })

    Path("Data/splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_rows_abs).to_csv("Data/splits/Brats21_goldstyle_t2_all_absolute.csv", index=False)
    pd.DataFrame(eval_rows_rel).to_csv("Data/splits/Brats21_goldstyle_t2_all_relative.csv", index=False)

    print("Processed rows:", len(results))
    print(results["status"].value_counts(dropna=False).to_string())
    print("Results:", results_out)
    print("Absolute CSV: Data/splits/Brats21_goldstyle_t2_all_absolute.csv")
    print("Relative CSV: Data/splits/Brats21_goldstyle_t2_all_relative.csv")

if __name__ == "__main__":
    main()
