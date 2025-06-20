"""
Module for loading and concatenating functional (func.gii),
eeg and physio.
"""
import json
from typing import Literal, List, Tuple

import nibabel as nb
import nilearn
import numpy as np

from scan.io.file import Participant
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
        physio_high_pass: bool = False
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
                physio_high_pass=physio_high_pass
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
        physio_high_pass: bool = False
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
                    basedir=self.func_dir, file_ext='lh.func.gii'
                )
                # load gifti data
                func_gii = Gifti(fp_lh, fp_rh)
                # signal filtering, if specified in init
                func_gii_data = self._filter(
                    func_gii.load(),
                    low_pass=func_low_pass,
                    high_pass=func_high_pass
                )
                # normalize data, if specified in init
                func_gii_data = self._norm(func_gii_data, norm=norm)
                output[d] = func_gii_data
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
                            physio = self._filter(
                                physio, 
                                low_pass=physio_low_pass, 
                                high_pass=physio_high_pass
                            )
                            # normalize data, if specified in init
                            physio = self._norm(physio, norm=norm)
                        output[d][p_out] = physio

        return output, func_gii

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

    def _filter(
        self, 
        signals: np.ndarray, 
        low_pass: bool, 
        high_pass: bool
    ) -> np.ndarray:
        """
        perform low- (< 0.15), high- (>0.01) or band-pass filtering
        of time courses with nilearn.signal.butterworth (5th order butterworth
        filter with padding of 100 samples). If no filtering
        is specified, return original signal
        """
        if low_pass or high_pass:
            # specify low- and high-pass cutoffs
            if high_pass:
                highpass = 0.01
            else:
                highpass = None

            if low_pass:
                lowpass = 0.15
            else:
                lowpass = None

            # get sampling frequency
            sf = 1/self.params['func']['tr']
            # perform signal filtering
            signals = nilearn.signal.butterworth(
                signals, sampling_rate=sf, low_pass=lowpass,
                high_pass=highpass, padlen=100
            )

        return signals

    def _norm(
        self, 
        signals: np.ndarray, 
        norm: Literal['zscore', 'demean', None] = 'zscore'
    ) -> np.ndarray:
        """
        normalize signals, either with zscoring or mean centering. If no
        normalization is specified, return original signal
        """
        if norm == 'zscore':
            # zscore along temporal dimension
            signals = zscore(signals, axis=0)
        elif norm == 'demean':
            signals -= np.mean(signals, axis=0)
        return signals


