import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import scipy.ndimage as ndimage

INPUT_DIR = "Data/preprocessed/gold_700_final/t2"
OUTPUT_DIR = "Data/preprocessed/gold_700_standardized/t2"

# Target sizes from the paper
TARGET_PAD = (192, 192, 160)

def standardize():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]

    print(f"Starting standardization for {len(files)} brains...")
    
    for fname in tqdm(files):
        img_nib = nib.load(os.path.join(INPUT_DIR, fname))
        data = img_nib.get_fdata()

        # 1. Pad to 192x192x160 (centers the brain in a blank array)
        padded_data = np.zeros(TARGET_PAD)
        
        # Get current dimensions
        d0, d1, d2 = data.shape
        
        # Safety check: ensure it doesn't exceed the target box
        c0, c1, c2 = min(d0, TARGET_PAD[0]), min(d1, TARGET_PAD[1]), min(d2, TARGET_PAD[2])

        # Calculate centering coordinates
        start0 = (TARGET_PAD[0] - c0) // 2
        start1 = (TARGET_PAD[1] - c1) // 2
        start2 = (TARGET_PAD[2] - c2) // 2
        
        # Place the brain in the center of the bounding box
        padded_data[start0:start0+c0, start1:start1+c1, start2:start2+c2] = data[:c0, :c1, :c2]

        # 2. Reduce resolution by factor of 2 (Downsample: 192x192x160 -> 96x96x80)
        # Using order=1 (bilinear interpolation) to balance speed and quality
        downsampled = ndimage.zoom(padded_data, zoom=0.5, order=1)

        # 3. Remove 15 top and bottom slices (Z-axis)
        # Slices 0-14 are removed, 15-64 are kept, 65-79 are removed.
        final_data = downsampled[:, :, 15:65]

        # Update the affine matrix (spacing doubles from 1mm to 2mm)
        new_affine = img_nib.affine.copy()
        new_affine[:3, :3] *= 2 
        
        new_img = nib.Nifti1Image(final_data, new_affine)
        nib.save(new_img, os.path.join(OUTPUT_DIR, fname))

if __name__ == "__main__":
    standardize()
