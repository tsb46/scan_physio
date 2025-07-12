"""
Module for loading and concatenating functional (func.gii),
eeg and physio.
"""
import json
import os
import warnings

from typing import Literal, List, Tuple, Dict

import nibabel as nb
import nilearn
import numpy as np

from scan.io.file import Participant
from scan.io import utils
from scipy.stats import zscore


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
    
    def load_separate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        load left and right hemisphere func.gii and return as separate arrays
        """
        # loop through arrays and return as separate arrays
        lh_data = []
        rh_data = []
        for lh_d, rh_d in zip(self.gii_lh.darrays, self.gii_rh.darrays):
            lh_data.append(lh_d.data)
            rh_data.append(rh_d.data)
        return np.array(lh_data), np.array(rh_data)
    
    def merge(self, lh_data: np.ndarray, rh_data: np.ndarray) -> np.ndarray:
        """
        Concatenate left and right hemisphere arrays into single array
        """
        if lh_data.shape[1] != self.lh_nvert:
            raise ValueError(
                'the # of vertices in the input left hemisphere data does'
                'not match the number of expected left hemisphere vertices'
            )
        if rh_data.shape[1] != self.rh_nvert:
            raise ValueError(
                'the # of vertices in the input right hemisphere data does'
                'not match the number of expected right hemisphere vertices'
            )
        return np.hstack((lh_data, rh_data))
    
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


class DatasetLoad:
    """
    Class for loading scan data, functional MRI (.gii) or physio, and
    concatenation across scans (optional).

    Attributes
    ----------
    dataset: Literal['vanderbilt']
        dataset label
    subj_filt: List[str] |
        list of subject labels or (subject, session) labels pairs
        (in a tuple) to exclude from load
    physio_dir:
        physio directory name for preprocessing output. Should be
        'proc1_physio' (default: 'proc1_physio')
    func_dir
        functional directory name for preprocessing output. If you
        want to analyze in native space, specify 'proc5_surface_smooth'.
        Otherwise, data is loaded from the last preprocessing step where
        functional data is in fsLR space: 'proc6_surfacelr'.
        (default: 'proc6_surfacelr')

    Methods
    -------
    load(concat = True):
        Iterate through scan data and concatenate (optional)
    load_scan(data, subj, ses)
        Load data for individual scan
    """
    def __init__(
        self,
        dataset: Literal['vanderbilt'],
        subj_filt: List[str] | List[Tuple[str,str]] = None,
        physio_dir: str = 'proc1_physio', # last output of physio pipeline
        func_dir: str = 'proc6_surfacelr', # last output of func pipeline
    ):
        self.dataset = dataset
        self.subj_filt = subj_filt
        # get dataset parameters
        # get data formatting
        with open('scan/meta/params.json', 'rb') as f:
            self.params = json.load(f)[dataset]
        # define output directories to search for files
        self.func_dir = f"{self.params['directory']['func']}/{func_dir}"
        self.physio_dir = f"{self.params['directory']['physio']}/{physio_dir}"
        # get scan iterator
        self.iter = Participant(dataset)
        # check if multiple sessions per subject
        self.session_flag = 'session' in self.iter.fields

    def load(
        self,
        data_type: Tuple[str, str] | Literal['func', 'physio'] = None,
        concat: bool = True,
        verbose: bool = True,
        norm: Literal['zscore', 'demean', None] = 'zscore',
        func_low_pass: bool = False,
        func_high_pass: bool = False,
        physio_low_pass: bool = False,
        physio_high_pass: bool = False,
        input_mask: bool = False,
        lh_roi_masks: List[str] = None,
        rh_roi_masks: List[str] = None
    ) -> Tuple[dict, Gifti]:
        """
        Iteratively load scan data and concatenate for group
        analysis (optional). Data can be functional (gii) or physio, or both.
        If concatenation is set to false, return the data for individual
        subjects in a list. Two outputs are returned, inluding the data in a
        dictionary with the key as the data modality (e.g. 'physio'), and a
        Gifti class from the last scan (this is needed for writing out
        outputs to func.gii after analysis). This assumes that the func.gii
        are consistent in shape (# of vertices in the left and right hemispheres
        are consistent across scans). This should be the case when performing
        group analysis.

        Parameters
        ----------
        concat: bool
            Whether to temporally concatenate scan data into
            a single array. Otherwise, return data from indvidual
            scans in a list (default: True).
        data: Tuple[str] | Literal['func', 'physio'] = ('func', 'physio')
            data modality. Can be functional or physio, or both. If both
            pass as a tuple (default: ('func', 'physio'))
        norm: Literal['zscore', 'demean', None]:
            The type of normalization to perform on the time courses. This
            is important if performing concatenation to remove differences
            in baseline signal between scans. Using zscore, differences in signal
            variability between scan are also removed (default: zscore). Note,
            sample weight physio files are not normalized before concatenation.
        func_high_pass: bool
            whether to perform a high-pass (>0.01Hz) 5th order butterworth filter
            to both physio and func time courses with nilearn.signal.butterworth.
            This is performed at the individual scan level before concatenation
            (default: False)
        func_low_pass: bool
            whether to perform a low-pass (<0.15Hz) 5th order butterworth filter
            to both physio and func time courses with nilearn.signal.butterworth.
            This is performed at the individual scan level before concatenation
            (default: False)
        physio_high_pass: bool
            high-pass filtering on physio time courses (default: False)
        physio_low_pass: bool
            low-pass filtering on physio time courses (default: False)
        input_mask: bool
            whether to apply roi masks to func.gii data (default: False). 
            Masks should have have a value of 1 for vertices within the mask,
            and 0 for vertices outside the mask. Time courses of vertices within the mask 
            are averaged together, and returned instead of the vertex time courses.
            BOTH left and right hemisphere roi masks should be passed as a list, with
            matching ROIs in the left and right hemisphers in the same order.
        lh_roi_masks: List[str]
            list of left hemisphere roi mask file paths to apply to
            func.gii data. See input_mask for more details. If input_mask is False,
            this parameter is ignored.
        rh_roi_masks: List[str]
            list of right hemisphere roi mask file paths to apply to
            func.gii data. See input_mask for more details. If input_mask is False,
            this parameter is ignored.
        verbose: bool
            print progress (default: True)

        Returns
        -------
        output: dict[str, List[np.ndarray]]
            group data in a list or one concatenated array packaged
            in a dictionary where the key is the data modality.
        gii: Gifti
            Gifti class for storing gifti parameters, this is needed for
            writing out outputs to func.gii after analysis. Gifti class
            is returned only if 'func' is passed to data parameter. Otherwise,
            returns None
        """
        # if data_type is not passed, set to all data types
        if data_type is None:
            data_type = ['func', 'physio']
        # if data_type is passed as str, convert to list
        if isinstance(data_type, str):
            data_type = [data_type]
        # check if data_type is valid
        if not all(d in ['func', 'physio'] for d in data_type):
            raise ValueError(f'data type {data_type} is not available')

        # if roi_masks are passed, load them
        if input_mask:
            lh_roi, rh_roi = self._load_masks(lh_roi_masks, rh_roi_masks)
        else:
            if lh_roi_masks is not None or rh_roi_masks is not None:
                warnings.warn('roi masks are passed, but input_mask is False. ROI masks will be ignored.')
            lh_roi, rh_roi = None, None
            
        # initalize output dictionary
        output = {
            'func': [],
            'physio': {
                p_out: [] for p in self.params['physio']['out']
                for p_out in self.params['physio']['out'][p]
            }
        }
        # set func_gii as None (returns None if 'physio' is set as data)
        func_gii = None
        for subj_ses in self.iter:
            # if multiple sessions per participant, split into subj and session
            if self.session_flag:
                subj = subj_ses[0]
                ses = subj_ses[1]
            else:
                subj = subj_ses[0]
                ses = None
            # print progress
            if verbose:
                print(f'loading scan: subj: {subj} ses: {ses}')
            # loop through data modalities, load data and append to list
            data_out, func_gii = self.load_scan(
                subj=subj, ses=ses, data=data_type,
                norm=norm,
                func_low_pass=func_low_pass,
                func_high_pass=func_high_pass,
                physio_low_pass=physio_low_pass,
                physio_high_pass=physio_high_pass,
                roi_lh_masks=lh_roi,
                roi_rh_masks=rh_roi,
                input_mask=input_mask
            )
            output['func'].append(data_out['func'])

            # loop through physio signals and append to list
            for p in self.params['physio']['out']:
                for p_out in self.params['physio']['out'][p]:
                    output['physio'][p_out].append(data_out['physio'][p_out])

        # if concatenate is True, stack along the temporal dimension
        if concat:
            output = self._concat(data=data_type, data_dict=output)

        return output, func_gii

    def load_scan(
        self,
        subj: str,
        ses: str,
        data: Tuple[Literal['func', 'physio']]  = ('func', 'physio'),
        norm: Literal['zscore', 'demean', None] = 'zscore',
        func_low_pass: bool = False,
        func_high_pass: bool = False,
        physio_low_pass: bool = False,
        physio_high_pass: bool = False,
        roi_lh_masks: Dict[str,np.ndarray] = None,
        roi_rh_masks: Dict[str, np.ndarray] = None,
        input_mask: bool = False
    ) -> Tuple[dict, Gifti]:
        """
        given subject and session label, load func or physio data. Data is
        returned in a dictionary with 'func' and 'physio' as separate keys (
        if both data modalities chosen). Functional time courses are
        returned as a 2D np.ndarray (# of timepoints, # of vertices),
        physio data is returned as a dictionary with keys as the physio signal
        label and values as the physio signal in a 2D np.ndarray (# of
        timepoints, 1). In addition, Gifti class for storing gifti parameters
        are returned; this is needed for writing out outputs to func.gii after
        analysis. Gifti class is returned only if 'func' is included in the
        list passed to data parameter. Otherwise, returns None.

        Parameters
        ----------
        data: Tuple[Literal['func', 'physio']]
            list of data modalities - can only be 'func' and/or physio
        subj: str
            subject label
        ses: str
            subject
        norm: Literal['zscore', 'demean', None]
            whether to normalize the data (default: zscore)
        func_low_pass: bool
            whether to perform low-pass filtering on func.gii data
        func_high_pass: bool
            whether to perform high-pass filtering on func.gii data
        physio_low_pass: bool
            whether to perform low-pass filtering on physio data
        physio_high_pass: bool
            whether to perform high-pass filtering on physio data
        roi_lh_masks: Dict[str, np.ndarray]
            left hemisphere roi mask with keys as the roi name
        roi_rh_masks: Dict[str, np.ndarray]
            right hemisphere roi mask with keys as the roi name
        input_mask: bool
            whether to apply roi masks to func.gii data

        Returns
        -------
        output: dict
            func and/or physio data returned as a dictionary. Top-level
            keys are 'func' and 'physio'. Within 'physio', different physio
            signals are returned as a dictionary with physio labels as keys (
            e.g. 'eog1').
        """
        # if data is passed as str, convert to list
        if isinstance(data, str):
            data = [data]
        # check data modality labels
        for d in data:
            if d not in ['func', 'physio']:
                raise ValueError(f'data modality {data} is not available')

        # set func_gii as None (returns None if 'physio' is set as data)
        func_gii = None
        # initialize output dictionary
        output = {
            'func': [],
            'physio': {
                p_out: [] for p in self.params['physio']['out']
                for p_out in self.params['physio']['out'][p]
            }
        }

        for d in data:
            if d == 'func':
                # get data from left and right hemispheres
                fp_lh = self.iter.to_file(
                    data=d, subject=subj, session=ses,
                    basedir=self.func_dir, file_ext='lh.func.gii'
                )
                fp_rh = self.iter.to_file(
                    data=d, subject=subj, session=ses,
                    basedir=self.func_dir, file_ext='rh.func.gii'
                )
                # load gifti data
                func_gii = Gifti(fp_lh, fp_rh)
                if input_mask:
                    func_data = self._extract_roi(
                        func_gii,
                        roi_lh_masks,
                        roi_rh_masks
                    )
                else:
                    func_data = func_gii.load()

                # signal filtering, if specified in init
                func_data_proc = utils.filter(
                    func_data,
                    low_pass=func_low_pass,
                    high_pass=func_high_pass,
                    tr=self.params['func']['tr']
                )
                # normalize data, if specified in init
                func_data_proc = utils.norm(func_data_proc, norm=norm)
                output[d] = func_data_proc

            if d == 'physio':
                # loop through physio signals of dataset
                output[d] = {}
                for p in self.params['physio']['out']:
                    for p_out in self.params['physio']['out'][p]:
                        # all physio preprocessing outputs are .txt files
                        physio_fp = self.iter.to_file(
                            data=d, subject=subj, session=ses,
                            basedir=self.physio_dir, physio=p_out,
                            file_ext='txt', physio_type='out'
                        )
                        physio = np.loadtxt(physio_fp, ndmin=2)
                        # signal filtering, if specified in init
                        # do not filter or normalize if physio is sample weights
                        if p_out != 'weight':
                            physio = utils.filter(
                                physio, 
                                low_pass=physio_low_pass, 
                                high_pass=physio_high_pass,
                                tr=self.params['func']['tr']
                            )
                            # normalize data, if specified in init
                            physio = utils.norm(physio, norm=norm)
                        output[d][p_out] = physio

        return output, func_gii

    def _extract_roi(
        self,
        gifti: Gifti,
        left_roi_masks: Dict[str,np.ndarray],
        right_roi_masks: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        extract roi time courses from func.gii data

        Parameters
        ----------
        gifti: Gifti
            Gifti class for storing gifti parameters
        left_roi_masks: Dict[str,np.ndarray]
            left hemisphere roi mask with keys as the roi name
        right_roi_masks: Dict[str, np.ndarray]
            right hemisphere roi mask with keys as the roi name

        Returns
        -------
        roi_data: np.ndarray
            roi time courses arranged in column-order (left hemisphere time courses
            followed by right hemisphere time courses)
        """
        lh_func, rh_func = gifti.load_separate()
        # loop through roi masks and extract roi time courses
        roi_data = []
        roi_names = []
        for lh_roi_name, lh_roi in left_roi_masks.items():
            roi_data.append(lh_func[:, lh_roi].mean(axis=1)[:, np.newaxis])
            roi_names.append(lh_roi_name)
        for rh_roi_name, rh_roi in right_roi_masks.items():
            roi_data.append(rh_func[:, rh_roi].mean(axis=1)[:, np.newaxis])
            roi_names.append(rh_roi_name)
        roi_data = np.hstack(roi_data)
        # set roi names for future reference
        self.roi_names = roi_names
        return roi_data

        
    def _load_masks(
        self, 
        lh_roi_mask_fps: List[str], 
        rh_roi_mask_fps: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        check if roi masks are valid and load them
        """
        # check if roi_masks are passed as a list
        if not isinstance(lh_roi_mask_fps, list):
            raise ValueError('lh_roi_masks must be passed as a list')
        # check if roi_masks are passed as a list
        if not isinstance(rh_roi_mask_fps, list):
            raise ValueError('rh_roi_masks must be passed as a list')
        # check if roi masks are valid
        if not all(os.path.exists(roi_mask) for roi_mask in lh_roi_mask_fps):
            raise ValueError('lh_roi_masks must be valid file paths')
        if not all(os.path.exists(roi_mask) for roi_mask in rh_roi_mask_fps):
            raise ValueError('rh_roi_masks must be valid file paths')
        # load roi masks
        lh_roi = [nb.load(roi_mask).darrays[0].data for roi_mask in lh_roi_mask_fps]
        rh_roi = [nb.load(roi_mask).darrays[0].data for roi_mask in rh_roi_mask_fps]
        # check if roi masks are valid
        utils.check_roi_masks(lh_roi, rh_roi)
        # convert to boolean mask
        lh_roi_mask = {
            roi_name: roi_mask == 1 
            for roi_name, roi_mask in zip(lh_roi_mask_fps, lh_roi)
        }
        rh_roi_mask = {
            roi_name: roi_mask == 1 
            for roi_name, roi_mask in zip(rh_roi_mask_fps, rh_roi)
        }
        return lh_roi_mask, rh_roi_mask

    def _concat(
        self,
        data: Tuple[Literal['func', 'physio']],
        data_dict: dict
    ) -> dict:
        """
        temorally concatenate func and/or physio data across scans
        """
        for d in data:
            if d == 'func':
                data_dict['func'] = np.concatenate(data_dict['func'], axis=0)
            elif d == 'physio':
                for p in self.params['physio']['out']:
                    for p_out in self.params['physio']['out'][p]:
                        data_dict['physio'][p_out] = np.concatenate(
                            data_dict['physio'][p_out], axis=0
                        )
        return data_dict



