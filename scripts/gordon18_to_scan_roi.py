"""
The Gordon18 atlas contains the 17 canonical resting-state networks (RSNs) defined by Gordon et al. (2016), along with
the SCAN ROIs defined in Gordon et al. (2023). Each RSN, including the SCAN ROIs, are represented as separate surface
overlay files in .mgh format and are in fsaverage6 space.

This script extracts the SCAN ROIs and foot, hand and mouth RSNs from the Gordon18 atlas with the following steps:

1) Convert SCAN (Net018) and foot/hand/mouth ROIs from the Gordon18 atlas into a gifti format.
2) The gifti files are transformed to the fsLR 32k surface using the neuromaps transforms module.
3) For the SCAN ROIs, extract each separate ROI (top, middle and bottom) from the merged ROI files and save them as separate gifti files.

The Gordon18 atlas is available at:
https://github.com/pBFSLab/UNITE/tree/master
"""

import nibabel as nib
import numpy as np

from neuromaps import transforms

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# define network file paths
# net018 is the SCAN ROIs in the Gordon18 atlas
LH_SCAN_ROI_FILE = "../template/lh.Gordon17_with_SCAN_fs6_net018_fs6.mgh"
RH_SCAN_ROI_FILE = "../template/rh.Gordon17_with_SCAN_fs6_net018_fs6.mgh"
# net017 is the foot RSN in the Gordan18 atlas
LH_FOOT_ROI_FILE = "../template/lh.Gordon17_with_SCAN_fs6_net017_fs6.mgh"
RH_FOOT_ROI_FILE = "../template/rh.Gordon17_with_SCAN_fs6_net017_fs6.mgh"
# net010 is the hand RSN in the Gordan18 atlas
LH_HAND_ROI_FILE = "../template/lh.Gordon17_with_SCAN_fs6_net010_fs6.mgh"
RH_HAND_ROI_FILE = "../template/rh.Gordon17_with_SCAN_fs6_net010_fs6.mgh"
# net011 is the mouth RSN in the Gordan18 atlas
LH_MOUTH_ROI_FILE = "../template/lh.Gordon17_with_SCAN_fs6_net011_fs6.mgh"
RH_MOUTH_ROI_FILE = "../template/rh.Gordon17_with_SCAN_fs6_net011_fs6.mgh"

# Load inflated fs_LR surfaces
LH_FS_LR_SURFACE = "../template/fsaverage.L.inflated.32k_fs_LR.surf.gii"
RH_FS_LR_SURFACE = "../template/fsaverage.R.inflated.32k_fs_LR.surf.gii"

# define scan labels
SCAN_LABELS = ["top", "middle", "bottom"]

OUTPUT_DIR = "../template"


def main():
    # load the SCAN .mgh surface files
    lh_scan_gii = _convert_to_gifti(LH_SCAN_ROI_FILE)
    rh_scan_gii = _convert_to_gifti(RH_SCAN_ROI_FILE)
    # load the foot/hand/mouth .mgh surface files
    lh_foot_gii = _convert_to_gifti(LH_FOOT_ROI_FILE)
    rh_foot_gii = _convert_to_gifti(RH_FOOT_ROI_FILE)
    lh_hand_gii = _convert_to_gifti(LH_HAND_ROI_FILE)
    rh_hand_gii = _convert_to_gifti(RH_HAND_ROI_FILE)
    lh_mouth_gii = _convert_to_gifti(LH_MOUTH_ROI_FILE)
    rh_mouth_gii = _convert_to_gifti(RH_MOUTH_ROI_FILE)

    # load inflated fs_LR surfaces (for connected components)
    lh_surf = nib.gifti.gifti.GiftiImage.load(LH_FS_LR_SURFACE)
    rh_surf = nib.gifti.gifti.GiftiImage.load(RH_FS_LR_SURFACE)

    # convert all ROIs to fsLR 32k surface
    fslr_lh_scan_gii = transforms.fsaverage_to_fslr(
        lh_scan_gii, target_density="32k", method="nearest", hemi="L"
    )
    fslr_rh_scan_gii = transforms.fsaverage_to_fslr(
        rh_scan_gii, target_density="32k", method="nearest", hemi="R"
    )
    fslr_lh_foot_gii = transforms.fsaverage_to_fslr(
        lh_foot_gii, target_density="32k", method="nearest", hemi="L"
    )
    fslr_rh_foot_gii = transforms.fsaverage_to_fslr(
        rh_foot_gii, target_density="32k", method="nearest", hemi="R"
    )
    fslr_lh_hand_gii = transforms.fsaverage_to_fslr(
        lh_hand_gii, target_density="32k", method="nearest", hemi="L"
    )
    fslr_rh_hand_gii = transforms.fsaverage_to_fslr(
        rh_hand_gii, target_density="32k", method="nearest", hemi="R"
    )
    fslr_lh_mouth_gii = transforms.fsaverage_to_fslr(
        lh_mouth_gii, target_density="32k", method="nearest", hemi="L"
    )
    fslr_rh_mouth_gii = transforms.fsaverage_to_fslr(
        rh_mouth_gii, target_density="32k", method="nearest", hemi="R"
    )

    # write the transformed SCAN ROIs
    nib.save(fslr_lh_scan_gii[0], f"{OUTPUT_DIR}/lh.SCAN_fsLR.label.gii")  # type: ignore
    nib.save(fslr_rh_scan_gii[0], f"{OUTPUT_DIR}/rh.SCAN_fsLR.label.gii")  # type: ignore
    # write the transformed foot/hand/mouth ROIs
    nib.save(fslr_lh_foot_gii[0], f"{OUTPUT_DIR}/lh.FOOT_fsLR.label.gii")  # type: ignore
    nib.save(fslr_rh_foot_gii[0], f"{OUTPUT_DIR}/rh.FOOT_fsLR.label.gii")  # type: ignore
    nib.save(fslr_lh_hand_gii[0], f"{OUTPUT_DIR}/lh.HAND_fsLR.label.gii")  # type: ignore
    nib.save(fslr_rh_hand_gii[0], f"{OUTPUT_DIR}/rh.HAND_fsLR.label.gii")  # type: ignore
    nib.save(fslr_lh_mouth_gii[0], f"{OUTPUT_DIR}/lh.MOUTH_fsLR.label.gii")  # type: ignore
    nib.save(fslr_rh_mouth_gii[0], f"{OUTPUT_DIR}/rh.MOUTH_fsLR.label.gii")  # type: ignore

    # get the union of the foot/hand/mouth ROIs to create a mask of all effector-specific ROIs
    fslr_lh_effector_mask = np.logical_or.reduce(
        [
            fslr_lh_foot_gii[0].darrays[0].data == 1,  # type: ignore
            fslr_lh_hand_gii[0].darrays[0].data == 1,  # type: ignore
            fslr_lh_mouth_gii[0].darrays[0].data == 1,  # type: ignore
        ]
    )
    fslr_rh_effector_mask = np.logical_or.reduce(
        [
            fslr_rh_foot_gii[0].darrays[0].data == 1,  # type: ignore
            fslr_rh_hand_gii[0].darrays[0].data == 1,  # type: ignore
            fslr_rh_mouth_gii[0].darrays[0].data == 1,  # type: ignore
        ]
    )
    # create gifti files for the effector-specific ROI masks
    gii_lh_effector_mask = nib.gifti.gifti.GiftiImage()
    gii_lh_effector_mask.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(fslr_lh_effector_mask.astype(np.float32))
    )
    gii_rh_effector_mask = nib.gifti.gifti.GiftiImage()
    gii_rh_effector_mask.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(fslr_rh_effector_mask.astype(np.float32))
    )
    # save the effector-specific ROI mask gifti files
    nib.save(gii_lh_effector_mask, f"{OUTPUT_DIR}/lh.EFFECTOR_fsLR.label.gii")  # type: ignore
    nib.save(gii_rh_effector_mask, f"{OUTPUT_DIR}/rh.EFFECTOR_fsLR.label.gii")  # type: ignore

    # find connected components to separate merged ROIs
    lh_labels, lh_coords, n_lh_comp = find_components(lh_surf, fslr_lh_scan_gii[0])  # type: ignore
    rh_labels, rh_coords, n_rh_comp = find_components(rh_surf, fslr_rh_scan_gii[0])  # type: ignore

    # there should only be three components in each hemisphere (top, middle and bottom)
    assert n_lh_comp == 3, (
        f"Expected 3 components in left hemisphere, found {n_lh_comp}"
    )
    assert n_rh_comp == 3, (
        f"Expected 3 components in right hemisphere, found {n_rh_comp}"
    )

    # compute the order of the components (top, middle and bottom) based on their centroids
    lh_order = _compute_order(lh_labels, lh_coords)
    rh_order = _compute_order(rh_labels, rh_coords)

    # save each component as a separate gifti file
    for lh_comp_id, rh_comp_id, name in zip(lh_order, rh_order, SCAN_LABELS):
        roi_lh = (lh_labels == lh_comp_id).astype(np.float32)
        roi_rh = (rh_labels == rh_comp_id).astype(np.float32)

        gii_lh = nib.gifti.gifti.GiftiImage()
        gii_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(roi_lh))

        gii_rh = nib.gifti.gifti.GiftiImage()
        gii_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(roi_rh))

        nib.save(gii_lh, f"{OUTPUT_DIR}/lh.SCAN_{name}.label.gii")  # type: ignore
        nib.save(gii_rh, f"{OUTPUT_DIR}/rh.SCAN_{name}.label.gii")  # type: ignore


def find_components(surf, gii_label):
    """
    Separate merged SCAN ROIs on surface into separate ROIs based on connected components. This is tricky
    because each SCAN ROI has the same label value (1) in the merged surface, so we need to use connected
    components to separate them. This is done by using a connected components algorithm on the surface mesh.
    """
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data

    label = gii_label.darrays[0].data
    mask = label == 1

    adj = _build_adjacency_matrix(len(coords), faces)

    mask_idx = np.where(mask)[0]
    sub_adj = adj[mask_idx][:, mask_idx]

    n_comp, comp_labels = connected_components(sub_adj)

    full_labels = np.zeros(len(coords), dtype=int)
    full_labels[mask_idx] = comp_labels + 1  # 1..k

    return full_labels, coords, n_comp


def _build_adjacency_matrix(n_vertices, faces):
    """Build sparse adjacency matrix from triangular faces."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(len(rows))

    adj_mat = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    return adj_mat


def _compute_order(full_labels, coords):
    """Return component IDs sorted top → bottom."""
    comps = np.unique(full_labels)
    comps = comps[comps != 0]

    centroids = {}
    for c in comps:
        verts = np.where(full_labels == c)[0]
        centroids[c] = coords[verts].mean(axis=0)

    # Sort by Z descending
    ordered = sorted(centroids.items(), key=lambda x: x[1][2], reverse=True)

    return [c for c, _ in ordered]


def _convert_to_gifti(mgh_file):
    """Convert .mgh surface file to .gii format."""
    data = nib.freesurfer.mghformat.MGHImage.load(mgh_file)
    array = data.get_fdata()[:, 0, 0]

    gii = nib.gifti.gifti.GiftiImage()
    data_array = nib.gifti.gifti.GiftiDataArray(
        data=array.astype(np.float32), intent="NIFTI_INTENT_LABEL"
    )
    gii.add_gifti_data_array(data_array)

    return gii


if __name__ == "__main__":
    main()
