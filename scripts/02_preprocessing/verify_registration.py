import os
import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ATLAS_PATH = "Data/atlases/T2.nii"
REG_DIR = Path("Data/preprocessed/gold_700_registered")
OUTPUT_PNG = "registration_check.png"

def verify_registration():
    print("🔍 --- VISUAL VERIFICATION OF REGISTRATION --- 🔍")
    
    # Load the Master Atlas
    atlas_img = nib.load(ATLAS_PATH)
    atlas_data = atlas_img.get_fdata()
    
    # Safely find the middle slice of the Z-axis
    atlas_mid_z = atlas_data.shape[2] // 2
    
    files = [f for f in os.listdir(REG_DIR) if f.endswith('.nii.gz')]
    
    if len(files) == 0:
        print("❌ No files found in the registered directory!")
        return
        
    print(f"📂 Found {len(files)} registered files.")
    
    # Pick 5 random patients
    sample_files = random.sample(files, 5)
    
    # Set up the plot (1 row, 6 columns)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    # Plot the Atlas in the first column (Squeezed to fix the 1x240x240 error)
    atlas_slice = np.squeeze(atlas_data[:, :, atlas_mid_z])
    axes[0].imshow(atlas_slice.T, cmap='gray', origin='lower')
    axes[0].set_title("Master Atlas", fontweight='bold')
    axes[0].axis('off')
    
    # Plot the 5 random patients
    for i, f in enumerate(sample_files):
        img = nib.load(REG_DIR / f)
        data = img.get_fdata()
        
        # We use the atlas_mid_z so we are looking at the EXACT same slice depth
        patient_slice = np.squeeze(data[:, :, atlas_mid_z])
        axes[i+1].imshow(patient_slice.T, cmap='gray', origin='lower')
        
        # Shorten the filename for the title
        short_name = f.split('_sub-')[1][:6] if '_sub-' in f else f[:6]
        axes[i+1].set_title(f"Patient {short_name}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='black')
    
    print("\n==================================================")
    print(f"✅ SUCCESS: Saved visual grid to {OUTPUT_PNG}")
    print("Download or open this PNG to verify the brains are centered and facing the same way as the Atlas.")
    print("==================================================")

if __name__ == "__main__":
    verify_registration()
