#!/bin/bash

# set run variables
fmriprep_dir=/u/project/CCN/apps/fmriprep/rh7/23.2.0/fmriprep-23.2.0.sif
nproc=10

# bids dir
bids_dir=/u/home/t/tbolt/project-lucina/projects/scan/data/utrecht

# freesurfer license
freesurfer_license=/u/home/t/tbolt/project-lucina/projects/scan/env/license.txt

# fmriprep call
apptainer run --cleanenv ${fmriprep_dir} \
${bids_dir} \
${bids_dir}/derivatives \
participant \
--nprocs ${nproc} \
--fs-license-file ${freesurfer_license} 




