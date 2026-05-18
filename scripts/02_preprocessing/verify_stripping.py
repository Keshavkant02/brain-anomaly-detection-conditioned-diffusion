import os
import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

STRIPPED_DIR = Path("Data/preprocessed/gold_700_skullstripped")
OUTPUT_PNG = "stripping_check.png"

def verify_stripping():
    print("🔍 --- VISUAL VERIFICATION OF SKULL STRIPPING --- 🔍")
    
    # Grab files, but explicitly IGNORE the mask files
    files = [f for f in os.listdir(STRIPPED_DIR) if f.endswith('.nii.gz') and not f.endswith('_mask.nii.gz')]
    
    if len(files) == 0:
        print("❌ No skull-stripped files found!")
        return
        
    print(f"📂 Found {len(files)} skull-stripped brains.")
    
    # Pick 5 random patients
    sample_files = random.sample(files, 5)
    
    # Set up the plot (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Plot the 5 random patients
    for i, f in enumerate(sample_files):
        img = nib.load(STRIPPED_DIR / f)
        data = img.get_fdata()
        
        # Find the middle slice
        mid_z = data.shape[2] // 2
        
        # Squeeze out the empty dimension for Matplotlib
        patient_slice = np.squeeze(data[:, :, mid_z])
        axes[i].imshow(patient_slice.T, cmap='gray', origin='lower')
        
        # Shorten the filename for the title
        short_name = f.split('_sub-')[1][:6] if '_sub-' in f else f[:6]
        axes[i].set_title(f"Patient {short_name}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='black')
    
    print("\n==================================================")
    print(f"✅ SUCCESS: Saved visual grid to {OUTPUT_PNG}")
    print("Download or open this PNG. The bright outer skulls should be COMPLETELY GONE, leaving only the brain.")
    print("==================================================")

if __name__ == "__main__":
    verify_stripping()
