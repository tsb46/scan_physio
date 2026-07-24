"""
Manual labeling of SCAN-like ROI identification from visualization of motor cortex activation
in response to physiological regressors. Manually identifed vertices and associated
coordinates on the motor cortex (in fsLR space) are used to create circular ROIs around the vertex coordinates.
The script writes separate top, middle and bottom ROIs of the scan network to separate gifti (.label.gii) files.
"""

import os

import nibabel as nb
import numpy as np


from nibabel.gifti.gifti import GiftiLabelTable, GiftiLabel, GiftiImage, GiftiDataArray

# template directory
TEMPLATE_DIRECTORY = "../template"

# SCAN ROI VERTEX NUMBERS
SCAN_TOP_LH = 5412
SCAN_TOP_RH = 5362
SCAN_MIDDLE_LH = 8245
SCAN_MIDDLE_RH = 8311
SCAN_BOTTOM_LH = 19046
SCAN_BOTTOM_RH = 19120

# SCAN ROI VERTEX COORDINATES (fs_LR)
SCAN_TOP_LH_COORDS = (16.458763, -12.976827, 65.59505)
SCAN_TOP_RH_COORDS = (-17.37413, -14.706095, 64.94013)
SCAN_MIDDLE_LH_COORDS = (-19.144726, 7.4526763, 33.055355)
SCAN_MIDDLE_RH_COORDS = (20.840258, 14.599702, 35.36756)
SCAN_BOTTOM_LH_COORDS = (-39.202225, 32.88463, -14.883785)
SCAN_BOTTOM_RH_COORDS = (40.029575, 30.3181, -10.898453)


def main():
    # load inflated gifti surface (fs_LR)
    gii_lh = nb.gifti.gifti.GiftiImage.load(
        os.path.join(TEMPLATE_DIRECTORY, "fsaverage.L.inflated.32k_fs_LR.surf.gii")
    )
    gii_rh = nb.gifti.gifti.GiftiImage.load(
        os.path.join(TEMPLATE_DIRECTORY, "fsaverage.R.inflated.32k_fs_LR.surf.gii")
    )

    from scipy.spatial import KDTree

    # get coordinates
    gii_lh_coords = gii_lh.agg_data("NIFTI_INTENT_POINTSET")
    gii_rh_coords = gii_rh.agg_data("NIFTI_INTENT_POINTSET")

    # create KDTree for left hemisphere
    kdtree_lh = KDTree(gii_lh_coords)
    # create KDTree for right hemisphere
    kdtree_rh = KDTree(gii_rh_coords)

    # get nearest neighbors within 6mm radius for each SCAN ROI vertex
    lh_top_neighbors = kdtree_lh.query_ball_point(SCAN_TOP_LH_COORDS, 6, p=2)
    lh_middle_neighbors = kdtree_lh.query_ball_point(SCAN_MIDDLE_LH_COORDS, 6, p=2)
    lh_bottom_neighbors = kdtree_lh.query_ball_point(SCAN_BOTTOM_LH_COORDS, 6, p=2)
    rh_top_neighbors = kdtree_rh.query_ball_point(SCAN_TOP_RH_COORDS, 6, p=2)
    rh_middle_neighbors = kdtree_rh.query_ball_point(SCAN_MIDDLE_RH_COORDS, 6, p=2)
    rh_bottom_neighbors = kdtree_rh.query_ball_point(SCAN_BOTTOM_RH_COORDS, 6, p=2)

    # create masks for each SCAN ROI
    lh_top_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)  # type: ignore
    lh_top_mask[lh_top_neighbors] = True
    lh_middle_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)  # type: ignore
    lh_middle_mask[lh_middle_neighbors] = True
    lh_bottom_mask = np.zeros(gii_lh_coords.shape[0], dtype=np.int32)  # type: ignore
    lh_bottom_mask[lh_bottom_neighbors] = True
    rh_top_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)  # type: ignore
    rh_top_mask[rh_top_neighbors] = True
    rh_middle_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)  # type: ignore
    rh_middle_mask[rh_middle_neighbors] = True
    rh_bottom_mask = np.zeros(gii_rh_coords.shape[0], dtype=np.int32)  # type: ignore
    rh_bottom_mask[rh_bottom_neighbors] = True

    # create gifti label table
    label_table = GiftiLabelTable()
    gifti_label = GiftiLabel(key=1, red=1, green=0, blue=0)
    gifti_label.label = str("SCAN")  # type: ignore
    label_table.labels.append(gifti_label)

    # create gifti images for each SCAN ROI
    lh_masks = [lh_top_mask, lh_middle_mask, lh_bottom_mask]
    rh_masks = [rh_top_mask, rh_middle_mask, rh_bottom_mask]
    scan_roi_names = ["TOP", "MIDDLE", "BOTTOM"]
    for lh_mask, rh_mask, scan_roi_name in zip(lh_masks, rh_masks, scan_roi_names):
        gii_lh_scan_roi = GiftiImage(labeltable=label_table)
        gii_lh_scan_roi.add_gifti_data_array(
            GiftiDataArray(data=lh_mask, datatype=16, intent="NIFTI_INTENT_LABEL")
        )
        gii_rh_scan_roi = GiftiImage(labeltable=label_table)
        gii_rh_scan_roi.add_gifti_data_array(
            GiftiDataArray(data=rh_mask, datatype=16, intent="NIFTI_INTENT_LABEL")
        )
        # write gifti images
        lh_fp_out = os.path.join(
            TEMPLATE_DIRECTORY, f"lh.SCAN_manual_{scan_roi_name}.label.gii"
        )
        rh_fp_out = os.path.join(
            TEMPLATE_DIRECTORY, f"rh.SCAN_manual_{scan_roi_name}.label.gii"
        )
        nb.save(gii_lh_scan_roi, lh_fp_out)  # type: ignore
        nb.save(gii_rh_scan_roi, rh_fp_out)  # type: ignore
        # set structure to cortext using connectome workbench
        os.system(f"""
            wb_command -set-structure {lh_fp_out} CORTEX_LEFT
        """)
        os.system(f"""
            wb_command -set-structure {rh_fp_out} CORTEX_RIGHT
        """)


if __name__ == "__main__":
    main()
