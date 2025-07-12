"""
Replication of SCAN network ROI identification using clustering methods on Vanderbilt
EEG-fMRI data. Analysis writes top, middle and bottom ROIs of the scan network to 
separate gifti (.label.gii) files. The pipeline is set with a deterministic random
seed to ensure reproducibility. Changing random seed may result in different cluster
assignments. 
"""
import os

import nibabel as nb
import numpy as np
import pandas as pd

from scan.io.load import DatasetLoad
from sklearn.cluster import BisectingKMeans, DBSCAN
from scan.io.write import ClusterResults

from nibabel.gifti import GiftiLabelTable, GiftiLabel, GiftiImage, GiftiDataArray

# template directory
TEMPLATE_DIRECTORY='template'

# SCAN ROI VERTEX NUMBERS
SCAN_TOP_LH = 5463
SCAN_TOP_RH = 5463
SCAN_MIDDLE_LH = 8230
SCAN_MIDDLE_RH = 8230
SCAN_BOTTOM_LH = 19285
SCAN_BOTTOM_RH = 19285

# SCAN ROI VERTEX COORDINATES (fs_LR)
SCAN_TOP_LH_COORDS = (16.1998, -10.9618, 64.7537)
SCAN_TOP_RH_COORDS = (-18.6646, -12.8767, 63.6935)
SCAN_MIDDLE_LH_COORDS = (-16.6407, 6.53068, 35.7849)
SCAN_MIDDLE_RH_COORDS = (13.2431, 5.67059, 39.5844)
SCAN_BOTTOM_LH_COORDS = (-42.7007, 27.2643, -9.17697)
SCAN_BOTTOM_RH_COORDS = (44.3305, 24.022, -6.89702)

def main():
    # load inflated gifti surface (fs_LR)
    gii_lh = nb.load(
        os.path.join(TEMPLATE_DIRECTORY, 'fsaverage.L.inflated.32k_fs_LR.surf.gii')
    )
    gii_rh = nb.load(
        os.path.join(TEMPLATE_DIRECTORY, 'fsaverage.R.inflated.32k_fs_LR.surf.gii')
    )

    from scipy.spatial import KDTree
    # get coordinates
    gii_lh_coords = gii_lh.agg_data('NIFTI_INTENT_POINTSET')
    gii_rh_coords = gii_rh.agg_data('NIFTI_INTENT_POINTSET')

    # create KDTree for left hemisphere
    kdtree_lh = KDTree(gii_lh_coords)
    # create KDTree for right hemisphere
    kdtree_rh = KDTree(gii_rh_coords)

    # get nearest neighbors within 6mm radius for each SCAN ROI vertex
    lh_top_neighbors = kdtree_lh.query_ball_point(SCAN_TOP_LH_COORDS, 6)
    lh_middle_neighbors = kdtree_lh.query_ball_point(SCAN_MIDDLE_LH_COORDS, 6)
    lh_bottom_neighbors = kdtree_lh.query_ball_point(SCAN_BOTTOM_LH_COORDS, 6)
    rh_top_neighbors = kdtree_rh.query_ball_point(SCAN_TOP_RH_COORDS, 6)
    rh_middle_neighbors = kdtree_rh.query_ball_point(SCAN_MIDDLE_RH_COORDS, 6)
    rh_bottom_neighbors = kdtree_rh.query_ball_point(SCAN_BOTTOM_RH_COORDS, 6)
    
    # create masks for each SCAN ROI
    lh_top_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)
    lh_top_mask[lh_top_neighbors] = True
    lh_middle_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)
    lh_middle_mask[lh_middle_neighbors] = True
    lh_bottom_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)
    lh_bottom_mask[lh_bottom_neighbors] = True
    rh_top_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)
    rh_top_mask[rh_top_neighbors] = True
    rh_middle_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)
    rh_middle_mask[rh_middle_neighbors] = True
    rh_bottom_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)
    rh_bottom_mask[rh_bottom_neighbors] = True

    # create gifti label table
    label_table = GiftiLabelTable()
    gifti_label = GiftiLabel(key=1, red=1, green=0, blue=0)
    gifti_label.label = str('SCAN')
    label_table.labels.append(gifti_label)


    # create gifti images for each SCAN ROI
    lh_masks = [lh_top_mask, lh_middle_mask, lh_bottom_mask]
    rh_masks = [rh_top_mask, rh_middle_mask, rh_bottom_mask]
    scan_roi_names = ['TOP', 'MIDDLE', 'BOTTOM']
    for lh_mask, rh_mask, scan_roi_name in zip(lh_masks, rh_masks, scan_roi_names):
        gii_lh_scan_roi = GiftiImage(labeltable=label_table)
        gii_lh_scan_roi.add_gifti_data_array(
            GiftiDataArray(
                data=lh_mask, 
                datatype=16, 
                intent='NIFTI_INTENT_LABEL'
            )
        )
        gii_rh_scan_roi = GiftiImage(labeltable=label_table)
        gii_rh_scan_roi.add_gifti_data_array(
            GiftiDataArray(
                data=rh_mask, 
                datatype=16, 
                intent='NIFTI_INTENT_LABEL'
            )
        )
        # write gifti images
        lh_fp_out = os.path.join(
            TEMPLATE_DIRECTORY,
            f'scan_roi_lh_{scan_roi_name}.label.gii'
        )
        rh_fp_out = os.path.join(
            TEMPLATE_DIRECTORY,
            f'scan_roi_rh_{scan_roi_name}.label.gii'
        )
        nb.save(gii_lh_scan_roi, lh_fp_out)
        nb.save(gii_rh_scan_roi, rh_fp_out)
        # set structure to cortext using connectome workbench
        os.system(f"""
            wb_command -set-structure {lh_fp_out} CORTEX_LEFT
        """)
        os.system(f"""
            wb_command -set-structure {rh_fp_out} CORTEX_RIGHT
        """)


if __name__ == '__main__':
    main()