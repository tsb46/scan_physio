"""
Module for writing analysis results to func.gii.
"""
import pickle
import os

from typing import Literal, TypedDict

import nibabel as nb
import numpy as np
from nibabel.gifti import GiftiImage, GiftiDataArray

from scan.io.load import Gifti


class ClusterResults:
    """
    Class for storing results of clustering. Provides
    utilities for writing cluster labels to pickle and func.gii files.
    """

    def __init__(self, cluster_labels: np.ndarray, cluster_params: dict):
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

        # write out cluster labels to func.gii
        write_gii(self.cluster_labels, gii_params, out_prefix)

        
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
        write_gii(self.pred_func, gii_params, out_prefix)


class DimReductionResults:
    """
    Class for storing results of dimension reduction. Provides
    utilities for writing dimension reduction results to func.gii files.

    Parameters
    ----------
    model: Literal['pca', 'ica']
        the dimension reduction model used

    params: dict
        the parameters used to fit the dimension reduction model

    coef: np.ndarray
        the coefficients of the dimension reduction model

    ts: np.ndarray
        the time series of the dimension reduction model
    """

    def __init__(
        self,
        model: Literal['pca', 'ica'],
        params: dict,
        coef: np.ndarray,
        ts: np.ndarray
    ):
        self.model = model
        self.params = params
        self.coef = coef
        self.ts = ts

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = None,
        out_dir: str = None
    ) -> None:
        """
        Write out dimension reduction results to func.gii file.

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
        
        # set file prefix based on model type if not provided
        if file_prefix is None:
            file_prefix = f'{self.model}_out'

        out_prefix = f'{out_dir}/{file_prefix}'
        # write out pca params
        out_params = {
            'model': self.model,
            'params': self.params,
            'coef': self.coef,
            'ts': self.ts
        }
        with open(f'{out_prefix}.pkl', 'wb') as f:
            pickle.dump(out_params, f)

        # write out dim red coef
        write_gii(self.coef, gii_params, out_prefix)


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
        write_gii(self.avg_func, gii_params, out_prefix)


def write_gii(data: np.ndarray, gii_params: Gifti, fp_out: str) -> None:
    """
    Write out functional data in 2D format (# of time points, # of vertices)
    to a func.gii file. Use parameters in Gifti class (gii_params) to write
    out in consistent format.
    """
    # split data into left and right hemispheres
    data_lh = data[:, :gii_params.split_indx]
    data_rh = data[:, gii_params.split_indx:]

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
    os.system(f"""
        wb_command -set-structure {fp_out}_lh.func.gii CORTEX_LEFT
    """)
    os.system(f"""
        wb_command -set-structure {fp_out}_rh.func.gii CORTEX_RIGHT
    """)