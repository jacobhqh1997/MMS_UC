import numpy as np
import sys
sys.path.append("./")
from utils.geodis_toolkits import randompoint, geodismap
import pandas as pd
import os
import SimpleITK as sitk
from scipy.ndimage import binary_closing, label
from scipy import ndimage as sp_ndimage

# --- 1. Update Path ---
# Initial binary segmentation result path
seg_path = 'path/to/seg'
# True Probability Graph (Pf) Path
prob_path = 'path/to/prob'
# Ground Truth Path
gd_path = 'path/to/gd'
# Original image path
img_path = 'path/to/imagesTr'
# Path to save the corrected results
save_dir = 'path/to/refined'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get a list of filenames from the binary segmentation results directory
seg_files = sorted(os.listdir(seg_path))

# Create a list to store the number of clicks for each file
click_records = []

for name in seg_files:
    print(f"Processing case: {name}")

    # --- 2. Load all necessary data ---

    # Load the binary segmentation map (for finding error regions)
    seg_sitk = sitk.ReadImage(os.path.join(seg_path, name))
    seg_arr = sitk.GetArrayFromImage(seg_sitk)

    # Load the true probability plot Pf (to serve as the basis for corrections)
    prob_name = name.replace('_rnet.nii.gz', '_rnet_prob.nii.gz')
    prob_file_path = os.path.join(prob_path, prob_name)
    if not os.path.exists(prob_file_path):
        print(f"  - Probability map not found for {name}, skipping.")
        continue
    prob_sitk = sitk.ReadImage(prob_file_path)
    Pf = sitk.GetArrayFromImage(prob_sitk)

    # Load Ground Truth and original image  
    gd_sitk = sitk.ReadImage(os.path.join(gd_path, name.replace('org_rnet.nii.gz', 'seg.nii.gz')))
    gd_arr = sitk.GetArrayFromImage(gd_sitk)
    img_name = name.replace('_rnet.nii.gz', '.nii.gz')
    img_sitk = sitk.ReadImage(os.path.join(img_path, img_name))
    img_arr = sitk.GetArrayFromImage(img_sitk)
    img_arr_processed = np.expand_dims(img_arr.astype(np.float32), axis=0)


    
    # --- 3. Correction Process ---

    # a. Find over-segmented and under-segmented regions (based on binary segmentation results)
    over_seg = np.where(seg_arr - gd_arr == 1, 1, 0)
    under_seg = np.where(seg_arr - gd_arr == -1, 1, 0)


    # b. Simulate corrective clicks on the error area.
    sb_refine, sb_clicks = randompoint(over_seg)
    sf_refine, sf_clicks = randompoint(under_seg)

    # Record the number of clicks on the current file
    click_records.append({'id': name, 'sb_clicks': sb_clicks, 'sf_clicks': sf_clicks})

    if sb_clicks == 0 and sf_clicks == 0:
        print(f"  - No refinement clicks needed. Saving original segmentation.")
        refined_seg_arr = seg_arr
    else:
        print(f"  - Refinement Clicks: Background={sb_clicks}, Foreground={sf_clicks}")

        # c. Calculate the corrected geodetic distance map (Ef, Eb)
        Ef, Eb = geodismap(sf_refine, sb_refine, img_arr_processed)

        # d. Core Fusion: Local Modulation Using Multiplication

        # Enhance Foreground: In the foreground click area, (1 + Ef) > 1, Pf is amplified.

        # Suppress Foreground: In the background click area, (1 - Eb) < 1, Pf is reduced.

        # In the no-click area, Ef=0, Eb=0, Pf remains unchanged.
        fore_enhancement_factor = 1.0  
        
        P_final_f = Pf * (1 + Ef * fore_enhancement_factor) * (1 - Eb)
        
    
        P_final_f = np.clip(P_final_f, 0, 1)

    
        segmentation_threshold = 0.5
        refined_seg_arr = (P_final_f > segmentation_threshold).astype(np.uint8)

        refined_seg_arr = binary_closing(refined_seg_arr, structure=np.ones((3,3,3))).astype(np.uint8)

       
        if seg_arr.sum() > 0 and refined_seg_arr.sum() == 0:
            print(f"  - Refinement resulted in empty segmentation. Reverting to original.")
            refined_seg_arr = seg_arr.copy()

  
    refined_sitk = sitk.GetImageFromArray(refined_seg_arr)
    refined_sitk.CopyInformation(seg_sitk) 
    sitk.WriteImage(refined_sitk, os.path.join(save_dir, name.replace('_rnet.nii.gz', '_rnet_refined.nii.gz')))


    clicks_df = pd.DataFrame(click_records)
    clicks_df.to_csv(os.path.join(save_dir, 'refinement_clicks_log.csv'), index=False)
    print(f"Click records saved to {os.path.join(save_dir, 'refinement_clicks_log.csv')}")


print("Refinement process finished.")