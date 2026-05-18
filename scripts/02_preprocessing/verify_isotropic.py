import os
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_DIR = Path("Data/preprocessed/gold_700_resampled")

def verify_isotropic():
    print("🔬 --- VERIFYING ISOTROPIC RESOLUTION (1.0mm³) --- 🔬")
    
    files = [f for f in os.listdir(TARGET_DIR) if f.endswith('.nii.gz')]
    
    if not files:
        print("❌ No files found in the resampled directory!")
        return

    failed_files = []
    
    print(f"🔍 Scanning {len(files)} files for perfect 1.0mm spacing...")
    for f in tqdm(files, desc="Verifying"):
        try:
            img = nib.load(TARGET_DIR / f)
            zooms = img.header.get_zooms()[:3]
            
            # We use np.allclose because floating-point math in NIfTI headers 
            # can sometimes record 1.0 as 0.99999999
            if not np.allclose(zooms, [1.0, 1.0, 1.0], atol=1e-3):
                failed_files.append((f, zooms))
        except Exception as e:
            print(f"\n❌ Error reading {f}: {e}")

    print("\n==================================================")
    if len(failed_files) == 0:
        print(f"💎 GREEN LIGHT: 100% of your {len(files)} scans are perfectly isotropic (1.0mm³).")
        print("The standardization phase is a complete success.")
    else:
        print(f"🚨 RED LIGHT: Found {len(failed_files)} non-isotropic scans!")
        for name, z in failed_files[:10]:
            print(f"  - {name}: Voxel Size {[round(float(v), 3) for v in z]}")
    print("==================================================")

if __name__ == "__main__":
    verify_isotropic()
