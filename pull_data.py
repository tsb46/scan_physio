import boto3
import botocore
import os
import pandas as pd

# Create directories
os.makedirs('data/func', exist_ok=True)
os.makedirs('data/physio', exist_ok=True)

# Code borrowed and modified from:
# https://github.com/jokedurnez/HCP_download

# Set up S3 bucket
boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('hcp-openaccess')


# Load subject list and iterate through subjects
subjects = pd.read_csv('subject_list_hcp.csv')
for s, lr in zip(subjects.subject, subjects.lr):
    # Set up file path strings
    print(s)
    # define base directories
    s_dir = f'HCP_1200/{s}/MNINonLinear/Results/rfMRI_REST1_{lr}'
    fs_dir = f'HCP_1200/{s}/MNINonLinear/fsaverage_LR32k'

    # Pull hcp-fix cleaned cifti file
    func_fp = f'{s_dir}/rfMRI_REST1_{lr}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    func_out = f'data/func/{s}_{lr}_clean.dtseries.nii'
    bucket.download_file(func_fp, func_out)

    # Pull freesurfer files for cifti smoothing
    fs_L_fp = f'{fs_dir}/{s}.L.midthickness.32k_fs_LR.surf.gii'
    fs_R_fp = f'{fs_dir}/{s}.R.midthickness.32k_fs_LR.surf.gii'
    fs_L_out = f'data/func/{s}.L.midthickness.32k_fs_LR.surf.gii'
    fs_R_out = f'data/func/{s}.R.midthickness.32k_fs_LR.surf.gii'
    bucket.download_file(fs_L_fp, fs_L_out)
    bucket.download_file(fs_R_fp, fs_R_out)

    # Pull physio .txt file
    phys_fp = f'{s_dir}/rfMRI_REST1_{lr}_Physio_log.txt'
    phys_out = f'data/physio/{s}_{lr}_physio.txt'
    bucket.download_file(phys_fp, phys_out)







