"""
Module for writing analysis results to func.gii.
"""
import pickle
import os

from typing import Literal, TypedDict

import nibabel as nb
import numpy as np
from nibabel.gifti import (
    GiftiImage, 
    GiftiDataArray, 
    GiftiLabel, 
    GiftiLabelTable,
)

from scan.io.load import Gifti


class ClusterResults:
    """
    Class for storing results of clustering. Provides
    utilities for writing cluster labels to pickle and func.gii files.
    """

    def __init__(self, cluster_labels: np.ndarray, cluster_params: dict):
        # if cluster_labels is a 1D array, convert to 2D array
        if cluster_labels.ndim == 1:
            cluster_labels = cluster_labels[np.newaxis,:]
        self.cluster_labels = cluster_labels
        self.cluster_params = cluster_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = 'cluster_out',
        out_dir: str = None
    ) -> None:
        """
        Write out cluster labels to pickle and func.gii files.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f'{out_dir}/{file_prefix}'
        # write out cluster params
        with open(f'{out_prefix}.pkl', 'wb') as f:
            pickle.dump(self.cluster_params, f)

        # write out cluster labels to label.gii
        write_label_gii(self.cluster_labels, gii_params, out_prefix)

        
class DistributedLagModelPredResults:
    """
    Class for storing predictions of distributed lag modeling. Provides
    utilities for writing predicted time courses to func.gii files.

    Parameters
    ----------
    pred_func: np.ndarray
        predicted time courses from dlm model represented as an ndarray
        with predicted time points in the rows and vertices in columns.

    dlm_params: dict
        the parameters used to fit the dlm model

    Methods
    -------
    write(out_fp, out_dir=None):
        write predicted time coureses to func.gii and dlm params to pickle

    """
    def __init__(self, pred_func: np.ndarray, dlm_params: dict):
        self.pred_func = pred_func
        self.dlm_params = dlm_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = 'dlm_pred_out',
        out_dir: str = None
    ) -> None:
        """
        Write out prediction results from dlm model to func.gii and pickle
        file. The func.gii displayed the predicted fMRI values over the
        predicted time span, and the pickle contains params passed to the
        dlm class.

        Parameters
        ----------
        gii_params: Gifti
            Gifti class that contains a loaded func.gii file. Used for
            writing out func.gii in the same format as the input func.gii.
            If running group-level analysis, this is returned in the
            Dataset.load() method.
        out_fp_prefix: str
            Optional - file path prefix for pickle and func.gii file
        out_dir: str
            Optional - output directory for writing files. If None (default),
            write out to current working directory.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f'{out_dir}/{file_prefix}'
        # write out dlm pred params
        with open(f'{out_prefix}.pkl', 'wb') as f:
            pickle.dump(self.dlm_params, f)

        # write predicted time courses to func.gii
        write_func_gii(self.pred_func, gii_params, out_prefix)


class ComplexPCAResults:
    """
    Class for storing results of complex-valued PCA. Provides
    utilities for writing complex-valued PCA results to func.gii files.

    Parameters
    ----------
    pc_scores: np.ndarray
        the PC scores of the complex-valued PCA model

    loadings: np.ndarray
        the loadings of the complex-valued PCA model

    explained_variance: np.ndarray
        the explained variance of the complex-valued PCA model
    """

    def __init__(
        self,
        pc_scores: np.ndarray,
        loadings: np.ndarray,
        explained_variance: np.ndarray
    ):
        self.pc_scores = pc_scores
        self.loadings = loadings
        self.explained_variance = explained_variance

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = None,
        out_dir: str = None
    ) -> None:
        """
        Write out complex-valued PCA results to func.gii file.

        Parameters
        ----------
        gii_params: Gifti
            Gifti class that contains a loaded func.gii file. Used for
            writing out func.gii in the same format as the input func.gii.
        file_prefix: str
            Optional - file path prefix for pickle and func.gii file
        out_dir: str
            Optional - output directory for writing files. If None (default),
            write out to current working directory.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()
        
        out_prefix = f'{out_dir}/{file_prefix}'
        out_prefix_phase = f'{out_prefix}_phase'
        out_prefix_amp = f'{out_prefix}_amp'
        # write out pca params
        out_params = {
            'pc_scores': self.pc_scores,
            'loadings': self.loadings,
            'explained_variance': self.explained_variance
        }
        with open(f'{out_prefix}.pkl', 'wb') as f:
            pickle.dump(out_params, f)
        

        # write out phase and amplitude maps from complex-valued loadings
        write_func_gii(np.angle(self.loadings).T, gii_params, out_prefix_phase)
        write_func_gii(np.abs(self.loadings).T, gii_params, out_prefix_amp)


class ComplexPCAReconResults:
    """
    Class for storing results of complex-valued PCA reconstruction of spatiotemporal patterns. 
    Provides utilities for writing reconstructed time courses to func.gii files.

    Parameters
    ----------
    bin_timepoints: np.ndarray
        the reconstructed time courses of the complex-valued PCA model

    bin_centers: np.ndarray
        the bin centers of the complex-valued PCA model
    """

    def __init__(self, bin_timepoints: np.ndarray, bin_centers: np.ndarray):
        self.bin_timepoints = bin_timepoints
        self.bin_centers = bin_centers

    def write(self, gii_params: Gifti, file_prefix: str = None, out_dir: str = None) -> None:
        """
        Write out reconstructed time courses to func.gii file.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f'{out_dir}/{file_prefix}'
        # write out bin centers to pickle
        with open(f'{out_prefix}_bin_centers.pkl', 'wb') as f:
            pickle.dump(self.bin_centers, f)
        
        # write out reconstructed time courses to func.gii
        write_func_gii(self.bin_timepoints, gii_params, out_prefix)
    
class WindowAverageResults:
    """
    Class for storing results of windowed averaging. Provides
    utilities for writing averaged time courses to func.gii files.
    """

    def __init__(self, avg_func: np.ndarray, window_average_params: dict):
        self.avg_func = avg_func
        self.window_average_params = window_average_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = 'window_avg_out',
        out_dir: str = None
    ) -> None:
        """
        Write out averaged time courses to func.gii file.

        Parameters
        ----------
        gii_params: Gifti
            Gifti class that contains a loaded func.gii file. Used for
            writing out func.gii in the same format as the input func.gii.
            If running group-level analysis, this is returned in the
            Dataset.load() method.
        file_prefix: str
            Optional - file path prefix for pickle and func.gii file
        out_dir: str
            Optional - output directory for writing files. If None (default),
            write out to current working directory.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f'{out_dir}/{file_prefix}'
        # write out window average params
        with open(f'{out_prefix}.pkl', 'wb') as f:
            pickle.dump(self.window_average_params, f)

        # write out averaged time courses to func.gii
        write_func_gii(self.avg_func, gii_params, out_prefix)

def write_func_gii(data: np.ndarray, gii_params: Gifti, fp_out: str) -> None:
    """
    Write out functional data in 2D format (# of time points, # of vertices)
    to a func.gii file. Use parameters in Gifti class (gii_params) to write
    out in consistent format.
    """
    # split data into left and right hemispheres
    data_lh, data_rh = _separate_gii_hemispheres(data, gii_params.split_indx)
    # Create new GiftiDataArrays for the left and right hemispheres
    gii_lh = GiftiImage()
    gii_rh = GiftiImage()
    for row_i in range(data_lh.shape[0]):
        gii_data_array_lh = GiftiDataArray(
            data=data_lh[row_i,:], datatype=16
        )
        gii_data_array_rh = GiftiDataArray(
            data=data_rh[row_i,:], datatype=16
        )
        gii_lh.add_gifti_data_array(gii_data_array_lh)
        gii_rh.add_gifti_data_array(gii_data_array_rh)

    # Save the new GIFTI files
    nb.save(gii_lh, f'{fp_out}_lh.func.gii')
    nb.save(gii_rh, f'{fp_out}_rh.func.gii')

    # set structure as left or right cortex to view in connectome workbench
    _set_structure(fp_out, 'func')



def write_label_gii(data: np.ndarray, gii_params: Gifti, fp_out: str) -> None:
    """
    Write out label data in 1D format (# of vertices) to a label.gii file.
    Use parameters in Gifti class (gii_params) to write out in consistent
    format.
    """
    # split data into left and right hemispheres
    data_lh, data_rh = _separate_gii_hemispheres(data, gii_params.split_indx)

    # squeeze data to 1D array
    data_lh = data_lh.squeeze()
    data_rh = data_rh.squeeze()

    # get unique labels
    unique_labels = np.unique(data)

    # generate unique RGB colors for each label
    unique_colors = _generate_unique_rgb_colors(len(unique_labels))

    # create label table
    label_table = GiftiLabelTable()
    for label_i, label in enumerate(unique_labels):
        # associate with unique r, g, b values
        r, g, b = unique_colors[label_i]
        gifti_label = GiftiLabel(key=label, red=r, green=g, blue=b)
        gifti_label.label = str(label)
        label_table.labels.append(gifti_label)

    # Create new GiftiDataArrays for the left and right hemispheres
    gii_lh = GiftiImage(labeltable=label_table)
    gii_rh = GiftiImage(labeltable=label_table)
    gii_lh.add_gifti_data_array(
        GiftiDataArray(
            data=data_lh, 
            datatype=16, 
            intent='NIFTI_INTENT_LABEL'
        )
    )
    gii_rh.add_gifti_data_array(
        GiftiDataArray(
            data=data_rh, 
            datatype=16, 
            intent='NIFTI_INTENT_LABEL'
        )
    )

    # Save the new GIFTI files
    nb.save(gii_lh, f'{fp_out}_lh.label.gii')
    nb.save(gii_rh, f'{fp_out}_rh.label.gii')
    
    # set structure as left or right cortex to view in connectome workbench
    _set_structure(fp_out, 'label')


def _separate_gii_hemispheres(
    data: np.ndarray,
    split_indx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate data into left and right hemispheres. Return as separate arrays
    with left hemisphere first and right hemisphere second.
    """
    data_lh = data[:,:split_indx]
    data_rh = data[:,split_indx:]
    return data_lh, data_rh


def _generate_unique_rgb_colors(n_colors: int) -> np.ndarray:
    """
    Generate unique RGB colors for labels.
    
    Parameters
    ----------
    n_colors : int
        Number of unique colors to generate
        
    Returns
    -------
    np.ndarray
        Array of shape (n_colors, 3) with RGB values between 0 and 1
    """
    if n_colors <= 0:
        return np.array([])
    
    # For small number of colors, use predefined distinct colors
    if n_colors <= 12:
        # Use distinct colors that are visually distinguishable
        distinct_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.0],  # Dark Green
            [0.5, 0.5, 0.0],  # Olive
            [0.5, 0.0, 0.5],  # Dark Magenta
            [0.0, 0.5, 0.5],  # Teal
        ])
        return distinct_colors[:n_colors]
    
    # For larger numbers, generate colors using golden ratio method
    # This ensures good distribution in color space
    colors = np.zeros((n_colors, 3))
    golden_ratio = 0.618033988749895
    
    for i in range(n_colors):
        hue = (i * golden_ratio) % 1.0
        # Convert HSV to RGB (simplified version)
        h = hue * 6
        c = 1.0
        x = c * (1 - abs(h % 2 - 1))
        m = 0.3  # Minimum brightness
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        colors[i] = [r + m, g + m, b + m]
    
    return colors

def _set_structure(fp_out: str, type: Literal['func', 'label']) -> None:
    """
    Set structure as left or right cortex to view in connectome workbench
    """
    os.system(f"""
        wb_command -set-structure {fp_out}_lh.{type}.gii CORTEX_LEFT
    """)
    os.system(f"""
        wb_command -set-structure {fp_out}_rh.{type}.gii CORTEX_RIGHT
    """)
