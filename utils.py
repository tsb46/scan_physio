import neurokit2 as nk
import nibabel as nb
import numpy as np
import pandas as pd

from nibabel.gifti.gifti import GiftiDataArray
from scipy.signal import butter, hilbert, sosfiltfilt, sosfreqz
from sklearn.linear_model import LinearRegression


# metadata parameters
n_scan = 1200 # number of time points per scan
n_vert = 59412 # number of vertices of left and right hemisphere
tr = 0.72 # hcp sampling interval
sf_func = 1/tr # hcp sampling frequency


def butter_bandpass(lowcut, highcut, fs, order=5):
    # construct butterworth bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, npad=500):
    # bandpass filter signal
    # Median padding to reduce edge effects
    data_pad = np.pad(data,[(npad, npad), (0, 0)], 'median')
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # Use filtfilt to avoid phase delay
    data_filt = sosfiltfilt(sos, data, axis=0)
    return data_filt


def load_data(physio_label, seeds=None):
    # Master function for loading functional and physio files for all HCP subjects
    # load subject list
    subject_df = pd.read_csv('subject_list_hcp.csv')
    # create a template for getting file path
    func_template = 'data/func/{0}_{1}_clean_proc.dtseries.nii'
    physio_template = 'data/physio/{0}_{1}_physio.txt'
    # pre-allocate array for group concatenation
    if seeds is None:
        func = np.empty((n_scan*subject_df.shape[0], n_vert), np.float32)
    else:
        func = np.empty((n_scan*subject_df.shape[0], len(seeds)), np.float32)
    physio = []
    indx = 0
    for subj, lr in zip(subject_df.subject, subject_df.lr):
        print(subj)
        cifti_fp = func_template.format(subj, lr)
        physio_fp = physio_template.format(subj, lr)
        func[indx:(indx+n_scan), :], cifti = load_cifti(cifti_fp, seeds)
        if physio_label is not None:
            physio.append(load_physio(physio_fp, physio_label))
        indx += n_scan
    return func, cifti, physio


def load_cifti(cifti_fp, seeds):
    # From: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    cifti = nb.load(cifti_fp)
    cifti_data = cifti.get_fdata(dtype=np.float32)
    brain_models = cifti.header.get_axis(1)  # Assume we know this
    cortex_l = surf_data_from_cifti(cifti_data, brain_models, 
                                    "CIFTI_STRUCTURE_CORTEX_LEFT")
    cortex_r = surf_data_from_cifti(cifti_data, brain_models, 
                                    "CIFTI_STRUCTURE_CORTEX_RIGHT")
    l_indx = np.arange(cortex_l.shape[0])
    r_indx = np.arange(cortex_r.shape[0])+l_indx[-1]
    lr_indx = (l_indx, r_indx)
    cortex_data = np.concatenate([cortex_l, cortex_r], axis=0).T
    if seeds is not None:
        cortex_data = cortex_data[:, seeds]
    return cortex_data, (cifti, lr_indx)


def load_physio(physio_fp, label, sf=400, sf_resamp=100):
    # load physio and derive resp amplitude and ppg amp
    physio_signals = np.loadtxt(physio_fp)
    physio_signals_df = pd.DataFrame({ 
     'resp': physio_signals[:,1],
     'ppg': physio_signals[:,2]
     })
    # Downsample to 100Hz (400Hz is unnecessary)
    physio_signals_resamp = physio_signals_df.apply(
        nk.signal_resample, sampling_rate=sf, desired_sampling_rate=sf_resamp, 
        method='FFT', axis=0
    )
    physio_out = preprocess_physio(physio_signals_resamp, sf_resamp, label)
    return physio_out


def preprocess_physio(physio_df, sf, label):
    # Derive respiration amplitude or PPG peak-to-peak amplitude, then resample
    # to functional sampling rate
    if label == 'resp':
        # Band-pass filter respiration signal to wide respiratory frequency: 0.05 to 0.5Hz
        resp_filt = butter_bandpass_filter(
           physio_df['resp'].values[:, np.newaxis], 0.05, 0.5, sf
        )
        # Derive ampitude of band-passed signal via Hilbert transform
        ts_amp = np.abs(hilbert(resp_filt))
    elif label == 'ppg':
        # Derive peak amplitudes from PPG signal
        ppg_signals, ppg_info = nk.ppg_process(physio_df['ppg'], sampling_rate=sf)
        # Get locations of PPG peaks and interpolate (cubic) amplitude values
        ppg_peaks_loc = np.where(ppg_signals['PPG_Peaks'])[0]
        ppg_peaks_amp = np.abs(ppg_signals['PPG_Clean'].iloc[ppg_peaks_loc])
        ts_amp = nk.signal_interpolate(
            ppg_peaks_loc, ppg_peaks_amp.values, 
            np.arange(ppg_signals.shape[0]), method='cubic'
        )

    # bandpass filter to functional frequency band: 0.01 - 0.1 Hz
    ts_amp_filt = butter_bandpass_filter(ts_amp, 0.01, 0.1, sf)
    # Resampling amplitude signal to length of functional scan (n_scan)
    ts_amp_resamp = nk.signal_resample(ts_amp_filt, desired_length=n_scan, method='FFT')

    return ts_amp_resamp


def regress_global_signal(func):
    # regress out global signal (mean time course) from functional time courses
    global_signal = func.mean(axis=1)
    lin_reg = LinearRegression()
    lin_reg.fit(global_signal.reshape(-1, 1), func)
    func_pred = lin_reg.predict(global_signal.reshape(-1,1))
    func_residual = func - func_pred
    return func_residual


def surf_data_from_cifti(data, axis, surf_name):
    # From: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            return data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
    raise ValueError(f"No structure named {surf_name}")


def volume_from_cifti(data, axis):
    # From: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]                          # Assume brainmodels axis is last, move it to front
    volmask = axis.volume_mask                               # Which indices on this axis are for voxels?
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)      # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                        dtype=data.dtype)
    vol_data[vox_indices] = data                             # "Fancy indexing"
    return nb.Nifti1Image(vol_data, axis.affine)             # Add affine for spatial interpretation


def write_cifti(data, cifti_lr, output_name):
    # Given 2D dataset, use a subject's cifti header to write out to cifti file
    cifti = cifti_lr[0]
    cifti_shape = cifti.get_fdata().shape
    cifti_data = np.zeros((data.shape[0], cifti_shape[1]), dtype=np.float32)
    lr_indx = cifti_lr[1]
    brain_models = cifti.header.get_axis(1)  # Assume we know this
    lr_name = ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]
    for indx, s_name in zip(lr_indx, lr_name):
        for name, data_indices, model in brain_models.iter_structures():  # Iterates over volumetric and surface structures
            if name == s_name:
                cifti_data[:, data_indices] = data[:, indx]

    # Create new axes due to matrix shape alteration
    # https://neurostars.org/t/alter-size-of-matrix-for-new-cifti-header-nibabel/20903/2
    ax_0 = nb.cifti2.SeriesAxis(start = 0, step = tr, size = cifti_data.shape[0])
    ax_1 = cifti.header.get_axis(1)
    new_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
    cifti_new = nb.Cifti2Image(cifti_data, header=new_header)
    nb.save(cifti_new, output_name)
