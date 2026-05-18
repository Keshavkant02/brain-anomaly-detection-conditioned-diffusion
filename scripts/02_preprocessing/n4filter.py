import ants
import sys
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='N4 Bias Field Correction')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for preprocessed files')
    parser.add_argument('-m', '--mask-dir', type=str, required=False, default=None,
                        help='mask directory for preprocessed files')
    return parser

def main(args=None):
    args = arg_parser().parse_args(args)
    if not os.path.isdir(args.img_dir):
        raise ValueError('(-i / --img-dir) argument needs to be a directory.')
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Standard N4 settings from the paper
    n4_opts = {'iters': [200, 200, 200, 200], 'tol': 0.0005}
    
    files = [f for f in os.listdir(args.img_dir) if f.endswith('.nii.gz')]
    
    for file in tqdm(files):
        out_path = os.path.join(args.out_dir, file)
        if not os.path.isfile(out_path):
            file_path = os.path.join(args.img_dir, file)
            img = ants.image_read(file_path)
            
            smoothed_mask = None
            if args.mask_dir is not None:
                # FIX: Map brain filename to the HD-BET mask name (_bet.nii.gz)
                mask_name = file.replace('.nii.gz', '_bet.nii.gz')
                mask_path = os.path.join(args.mask_dir, mask_name)
                
                if os.path.isfile(mask_path):
                    mask = ants.image_read(mask_path)
                    smoothed_mask = ants.smooth_image(mask, 1)
                else:
                    print(f"\nWarning: Mask not found for {file} at {mask_path}")

            # Perform bias field correction
            # The mask helps the algorithm ignore the black background
            img_corr = ants.n4_bias_field_correction(img, convergence=n4_opts, weight_mask=smoothed_mask)
            
            ants.image_write(img_corr, out_path)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
