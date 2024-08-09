"""
Module for loading and concatenating functional (func.gii),
eeg and physio.
"""
from typing import Literal, List, Tuple

import nibabel as nb
from nibabel.gifti import GiftiImage, GiftiDataArray
import numpy as np

from scan.io.file import Participant

class DatasetLoad:
    """
    Load and concatenate files (eeg, func or physio) for a dataset.

    Attributes
    ----------
    dataset: Literal['vanderbilt']
        dataset label
    data: List[str]
        Data modality to load. Can be multiple of - func, physio or eeg
        (default: ('func', 'physio') )
    subj_filt: List[str]
        filepath to right hemisphere func.gii
    """
    def __init__(
        dataset: Literal['vanderbilt'],
        data: Tuple[str] = ('func', 'physio'),
        subj_filt: List[str] = None
    ):
        pass




class Gifti:
    """
    Class for loading left and right hemisphere
    func.gii files for analysis

    Attributes
    ----------
    fp_gii_lh: str
        filepath to left hemisphere func.gii
    fp_gii_rh: str
        filepath to right hemisphere func.gii

     Methods
    -------
    load():
        load left and right hemisphere func.gii and concatenate into array
    split()
        split concatenated array into left and right hemisphere arrays (in
        that order)

    """
    def __init__(
        self,
        fp_gii_lh: str,
        fp_gii_rh: str
    ):
        # Load the GIFTI files for both hemispheres
        self.gii_lh = nb.load(fp_gii_lh)
        self.gii_rh = nb.load(fp_gii_rh)
        self.split_indx = self.gii_lh.darrays[0].data.shape[0]
        # get # of vertices per hemisphere
        self.lh_nvert = self.gii_lh.darrays[0].data.shape[0]
        self.rh_nvert = self.gii_rh.darrays[0].data.shape[0]
        if self.lh_nvert != self.rh_nvert:
            raise ValueError(
                'left and right hemispheres should have same number of vertices'
            )

    def load(self) -> np.ndarray:
        """
        load left and right hemisphere func.gii and concatenate into array
        """
        # loop through samples and concatenate into one array
        combined_data = []
        for lh_d, rh_d in zip(self.gii_lh.darrays, self.gii_rh.darrays):
            # Access the data arrays in the GIFTI files
            data_left = lh_d.data
            data_right = rh_d.data
            combined_data.append(np.hstack((data_left, data_right)))

        return np.vstack(combined_data)

    def split(self, combined_data: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """
        split concatenated array into left and right hemisphere arrays (in
        that order). The array is expected to have # of samples in rows
        and # of vertices in the columns
        """
        if combined_data.shape[1] != (self.lh_nvert + self.rh_nvert):
            raise ValueError(
                'the # of vertices in combined_data does not match the number '
                'left and right hemisphere vertices'
            )
        data_left = combined_data[:, :self.split_indx]
        data_right = combined_data[:, self.split_indx:]
        return data_left, data_right






