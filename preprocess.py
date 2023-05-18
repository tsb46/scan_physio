import nibabel as nb
import numpy as np
import os
import pandas as pd

from scipy.stats import zscore
from utils import butter_bandpass_filter, tr

# Mask displays of some nibabel messages
nb.imageglobals.logger.setLevel(40)


# Command string for extraction of left and right hemisphere from cifti dense files
cifti_cmd_smooth = """
wb_command -cifti-smoothing {0} 4.0 4.0 COLUMN {1} \
-left-surface {2} -right-surface {3}
"""

data_dir = 'data/func'
subjects = pd.read_csv('subject_list_hcp.csv')
for s, lr in zip(subjects.subject, subjects.lr):
    print(s)
    # define file path strings
    func_fp = f'{data_dir}/{s}_{lr}_clean.dtseries.nii'
    func_fp_proc = f'{data_dir}/{s}_{lr}_clean_proc.dtseries.nii'
    # smoothing - 3mm FWHM
    fs_L = f'{data_dir}/{s}.L.midthickness.32k_fs_LR.surf.gii'
    fs_R = f'{data_dir}/{s}.R.midthickness.32k_fs_LR.surf.gii'
    os.system(cifti_cmd_smooth.format(func_fp, func_fp_proc, fs_L, fs_R))
    # load cifti
    cifti = nb.load(func_fp_proc)
    cifti_data = cifti.get_fdata()
    # Demean data 
    cifti_data = zscore(cifti_data)
    # Bandpass filter
    fs = 1 / tr
    cifti_data = butter_bandpass_filter(cifti_data, 0.01, 0.1, fs)
    # Write out to cifti
    cifti_out = nb.Cifti2Image(cifti_data, cifti.header)
    cifti_out.set_data_dtype('<f4')
    nb.save(cifti_out, func_fp_proc)

    
