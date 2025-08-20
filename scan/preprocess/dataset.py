"""
Dataset-specific utilities for loading and preprocessing physio/EEG data
"""
import json
import os
from typing import Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd

from scipy.io import loadmat


VANDERBILT_EEG_CHAN_LABELS = [
    'P3', 'P4', 'P7', 'P8', 'Pz', 'O1', 'O2', 'Oz'
]

def load_physio_newcastle(
    blink_fp: str,
    saccade_fp: str,
    physio_fp: str,
) -> Tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Load raw physio signals collected from respiratory belt, ppg monitor,
    and blink onsets from eye-tracking.

    Parameters
    ----------
        blink_fp: str
            filepath to blink onsets in .csv format.
        saccade_fp: str
            filepath to saccade data in .csv format.
        physio_fp: str
            filepath to physio (resp) data in .tsv.gz format.

    Returns
    -------
    physio: dict[str, np.ndarray]
        physio signals
    sf: dict[str, float]
        sampling frequency of physio signals
    """
    # load saccade data from eye-tracking .csv file
    saccade_raw = pd.read_csv(saccade_fp)
    # load blink onsets from eye-tracking .csv file
    blink_raw = pd.read_csv(blink_fp)
    # get json file path from physio_fp
    physio_json = os.path.splitext(os.path.splitext(physio_fp)[0])[0] + '.json'
    # load physio parameters from json file
    with open(physio_json, 'r') as f:
        physio_params = json.load(f)
    # load physio data from tab separated values file
    physio_raw = pd.read_csv(physio_fp, sep='\t')
    # get respiratory and ppg signals (different column names across files)
    try:
        resp = physio_raw['Respiration'].to_numpy()
        ppg = physio_raw['Pulse'].to_numpy()
    except KeyError:
        # assume resp and ppg are in the first two columns
        physio_raw = pd.read_csv(physio_fp, sep='\t', header=None)
        resp = physio_raw.iloc[:,0].to_numpy()
        ppg = physio_raw.iloc[:,1].to_numpy()
    # get sampling frequency
    sf_physio = physio_params['SamplingFrequency']
    # get blink onset periods
    
    # package into dictionaries
    physio = {
        'resp': resp,
        'ppg': ppg
    }
    sf = {
        'resp': sf_physio,
        'ppg': sf_physio
    }
    return physio, sf


def load_physio_vanderbilt(
    physio_fp: str,
    eeg_fp: str
) -> Tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Load raw physio signals collected from eeg and respiratory belt (Philips
    scanner or Biopac hardware)

    Parameters
    ----------
        physio_fp: str
            filepath to physio (resp) data in .mat format.
        eeg_fp: str
            filepath to gradient-artifact corrected EEG data in .mat format.

    Returns
    -------
    signals: dict[str, np.ndarray]
        physio and eeg signals
    sf: dict[str, float]
        sampling frequency of physio and eeg signals
    """
    ## load respiratory belt signals
    physio_raw = loadmat(physio_fp, squeeze_me=True)
    # handle Philips scanner recordings differently from BIOPAC
    if 'OUT_p' in physio_raw:
        # Philips scanner recordings
        sf_physio = 1/physio_raw['OUT_p']['dt_phys'].item()
        resp = physio_raw['OUT_p']['resp_sync'].item()
        # load annotated spans of respiratory belt recordings marked as bad
        weights_idx = physio_raw['OUT_p']['resp_bad_ind'].item()
        # create time course of sample weights
        weights = np.ones(len(resp))
        # - 1 to account for zero-indexing in Python
        weights[weights_idx-1] = 0
    else:
        # BIOPAC recordings
        resp, sf_physio = _biopac_load_vanderbilt(physio_raw)
        weights = np.ones(len(resp))

    # load eog and emg signals
    eeg_mat = loadmat(eeg_fp, squeeze_me=True)
    eeg_data = eeg_mat['EEG']['data'].item()
    sf_eeg = eeg_mat['EEG']['srate'].item()
    chan_labels = [chan[0] for chan in eeg_mat['EEG']['chanlocs'].item()]

    # load eeg data averaged across parietal/occipital channels
    eeg_signals = _extract_eeg_vanderbilt(eeg_data, chan_labels)

    # package into dictionaries
    physio = {
        'resp': resp,
        'weight': weights,
        'eog1': eeg_data[chan_labels.index('EOG1'), :],
        'eog2': eeg_data[chan_labels.index('EOG2'), :],
        'emg1': eeg_data[chan_labels.index('EMG1'), :],
        'emg2': eeg_data[chan_labels.index('EMG2'), :],
        'emg3': eeg_data[chan_labels.index('EMG3'), :],
        'eeg': eeg_signals
    }
    sf = {
        'resp': sf_physio,
        'weight': sf_physio,
        'eog1': sf_eeg,
        'eog2': sf_eeg,
        'emg1': sf_eeg,
        'emg2': sf_eeg,
        'emg3': sf_eeg,
        'eeg': sf_eeg
    }
    return physio, sf


def _extract_eeg_vanderbilt(
    eeg_data: np.ndarray,
    chan_labels: list[str]
) -> np.ndarray:
    """
    Extract partial/occipital eeg signals from vanderbilt dataset.

    Parameters
    ----------
        eeg_data: np.ndarray
            eeg data
        chan_labels: list[str]
            channel labels
        sf_eeg: float
            sampling frequency of eeg data

    Returns
    -------
    eeg_data: np.ndarray
        eeg data
    """
    eeg_chan = VANDERBILT_EEG_CHAN_LABELS
    eeg_chan_idx = [chan_labels.index(chan) for chan in eeg_chan]
    eeg_data = eeg_data[eeg_chan_idx, :]
    # transpose to time x channel
    return eeg_data.T


def _biopac_load_vanderbilt(
    physio: dict
) -> Tuple[np.ndarray, float]:
    """
    During scanning, vanderbilt acquired a BIOPAC system for recording
    respiratory recordings. These signals need to be handled a little
    differently from the respiratory recordings from the Phillips scanner.
    BIOPAC recordings are sampled at 2000Hz and are resampled to 200 Hz to
    ease later computations.

    Parameters
    ----------
        physio: dict
            .mat file structure of BIOPAC recordings as a dictionary

    Returns
    -------
    resp_resample: np.ndarray
        resampled respiratory signal from the BIOPAC system
    sf: float
        new sampling frequency of respiratory signal (200Hz)

    """
    # hardcoded trigger and respiratory field labels
    trigger_field = 'DTU100 - Trigger View, AMI / HLT - A 1'
    resp_field = 'Custom, DA100C'
    signals = physio['data']
    fields = physio['labels'].tolist()
    # interstimulus interval in milliseconds
    isi = physio['isi']
    # convert isi from ms to s
    isi = isi / 1000
    # convert to sampling frequency
    sf = 1/isi
    # get respiratory field
    resp = signals[:,fields.index(resp_field)]
    # get mr trigger
    mr_trigger = signals[:, fields.index(trigger_field)]
    # identify first and last trigger (threshold of 0.05 works)
    triggers = np.where(mr_trigger > 0.05)[0]
    first_trigger, last_trigger = triggers[0], triggers[-1]
    # crop respiratory signal based on MR triggers
    resp_crop = resp[first_trigger:last_trigger]
    # resample to 200 Hz
    sf_new = 200
    resp_resample = nk.signal_resample(
        resp_crop,
        sampling_rate=sf,
        desired_sampling_rate=sf_new
    )
    return np.asarray(resp_resample), sf_new






