"""
Pipeline module for preprocessing anatomical, functional, EEG and physio
datasets.
"""

import json
import os

from multiprocessing import Pool
from typing import Literal


from scan.io.file import Participant
from scan.preprocess import fsl
from scan.preprocess import freesurfer as fs
from scan.preprocess import workbench as wb
from scan.preprocess.misc import tedana_denoise
from scan import utils


class FileMapper:
    """
    Utility class for mapping input and output files for
    anatomical and functional

    Attributes
    ----------
    dataset: str
        dataset label
    params: dict
        dataset params

    Methods
    -------
    map():
        map all input and output files for each scan

    """
    # get participant iterator
    def __init__(self, dataset: str, params: dict):
        self.dataset = dataset
        self.params = params
        # initialize participant iterator
        self.iter = Participant(self.dataset)

    def map(self) -> dict:
        """
        define output paths for all preprocessing outputs and create directories for storing them
        """
        # create output directories
        dir_map = self._create_directories()
        # store directory paths
        file_map = {
            'anat': {},
            'func': {}
        }

        # loop through subjects then sessions
        for subj, ses_list in self.iter.subject_by_session():
            file_map['anat'][subj] = {}
            file_map['func'][subj] = {}
            # get directory paths
            dir_anat = dir_map['anat']
            dir_func = dir_map['func']

            self._map_anat(subj, dir_anat, file_map['anat'][subj])
            # loop through sessions and create directories
            for ses in ses_list:
                file_map['func'][subj][ses] = {}
                self._map_func(subj, ses, dir_func, file_map['func'][subj][ses])

        return file_map

    def _map_anat(self, subj: str, dir_anat: str, file_map_subj: dict) -> None:
        """
        create output files for anatomical pipeline. Modify in-place
        """
        # raw anatomical
        file_map_subj['raw'] = self._filepath(
            data='anat', subject=subj,
            basedir=dir_anat['raw']
        )
        # freesurfer subject out dir
        file_map_subj['freesurfer'] = os.path.abspath(
            f"{dir_anat['freesurfer']}"
        )
        # freesurfer subject out dir
        file_map_subj['freesurfer_sdir'] = os.path.abspath(
            f"{dir_anat['freesurfer']}/{subj}"
        )
        # miscellanous anat outputs
        file_map_subj['misc'] = {}
        ## freesurfer brain mask
        file_map_subj['misc']['fs_brainmask'] = os.path.abspath(
            f"{file_map_subj['freesurfer_sdir']}/mri/brainmask.mgz"
        )
        ## freesurfer midthickness
        fslr_fp = utils.get_fp_base(
            self._filepath(
                data='anat', subject=subj,
                basedir=dir_anat['fslr']
            )
        )
        ### create files for each hemisphere (32k res)
        for hemi, hemi_lr in zip(['lh', 'rh'], ['L', 'R']):
            file_map_subj['misc'][f'fs_mid_{hemi}'] = \
            f'{fslr_fp}.{hemi}.midthickness.surf.gii'
            file_map_subj['misc'][f'fslr_mid_{hemi}'] = \
            f'{fslr_fp}.{hemi_lr}.midthickness.32k_fs_LR.surf.gii'
            file_map_subj['misc'][f'fs_sphere_{hemi}'] = \
            f'{fslr_fp}.{hemi}.sphere.reg.surf.gii'


    def _map_func(
        self,
        subj: str,
        ses: str,
        dir_func: str,
        file_map_subj: dict
    ) -> None:
        """
        create output files for functional pipeline. Modify in-place
        """
        # filepaths for tasks with gifti output
        # map files differently for multi- vs. single-echo datasets
        for task in ['raw', 'trim', 'slicetime', 'motion']:
            file_map_subj[task] = self._filepath(
                data='func', subject=subj, session=ses,
                basedir=dir_func[task], return_echos=True
            )
        # coregistration file handling is same for multi- vs. single-echo
        for task in ['coregister']:
            file_map_subj[task] = self._filepath(
                data='func', subject=subj, session=ses,
                basedir=dir_func[task]
            )
        # separate .gii outputs per hemisphere
        for task in ['surface_smooth', 'surface_lr']:
            file_map_subj[task] = {}
            for hemi in ['lh', 'rh']:
                file_map_subj[task][hemi] = self._filepath(
                    data='func', subject=subj, session=ses,
                    basedir=dir_func[task], func_ext='gii',
                    hemi=hemi
                )

        # tedana out directory
        file_map_subj['tedana'] = utils.get_fp_base(
            self._filepath(
                data='func', subject=subj, session=ses,
                basedir=dir_func['tedana']
            )
        )
        # miscellanous functional outputs
        file_map_subj['misc'] = {}
        ## mcflirt mean volume
        motion_fp = file_map_subj['motion']
        if self.params['multiecho']:
            motion_fp_base = utils.get_fp_base(motion_fp[0])
        else:
            motion_fp_base = utils.get_fp_base(motion_fp)
        file_map_subj['misc']['mean_vol'] = os.path.abspath(
            f"{motion_fp_base}_mean_reg.nii.gz"
        )
        ## mcflirt transform matrix
        file_map_subj['misc']['mcflirt_mat'] = os.path.abspath(
            f"{motion_fp_base}.mat"
        )
        ## coregistration transform matrix
        coreg_fp = utils.get_fp_base(file_map_subj['coregister'])
        file_map_subj['misc']['surfregister_mat'] = os.path.abspath(
            f'{coreg_fp}.lta'
        )
        ## coregistration inital FLIRT transform matrix
        file_map_subj['misc']['surfregister_fslmat'] = os.path.abspath(
            f'{coreg_fp}.mat'
        )
        ## coregistration reg file
        file_map_subj['misc']['surfregister_reg'] = os.path.abspath(
            f'{coreg_fp}_mean_reg.nii.gz'
        )
        ## mask file for tedana preprocessing
        file_map_subj['misc']['fs_func_mask'] = os.path.abspath(
            f'{coreg_fp}_brainmask.nii.gz'
        )
        ## tedana denoised file
        tedana_out = file_map_subj['tedana']
        tedana_out_fp = os.path.basename(file_map_subj['tedana'])
        file_map_subj['misc']['tedana_denoised'] = os.path.abspath(
            f'{tedana_out}/{tedana_out_fp}_desc-denoised_bold.nii.gz'
        )

    def _create_directories(self) -> dict:
        """
        create output directories and store
        paths in dictionary
        """
        # anatomical directory
        anat_dir = self.params['directory']['anat']
        # functional directory
        func_dir = self.params['directory']['func']
        # directory map
        directory_map = {
            'anat': {
                'raw': f'{anat_dir}/raw',
                'freesurfer': f'{anat_dir}/freesurfer',
                'fslr': f'{anat_dir}/proc1_fslr'
            },
            'func': {
                'raw': f'{func_dir}/raw',
                'trim': f'{func_dir}/proc1_trim',
                'slicetime': f'{func_dir}/proc2_slicetime',
                'motion': f'{func_dir}/proc3_motion',
                'coregister': f'{func_dir}/proc4_coregister',
                'surface_smooth': f'{func_dir}/proc5_surface_smooth',
                'surface_lr': f'{func_dir}/proc6_surfacelr',
            }
        }
        # include tedana directory, if multiecho dataset
        if self.params['multiecho']:
            directory_map['func']['tedana'] = f'{func_dir}/proc_tedana'

        # create directories, if doesn't exist
        for d in ['anat', 'func']:
            for p in directory_map[d]:
                os.makedirs(directory_map[d][p], exist_ok=True)
        return directory_map

    def _filepath(
        self,
        data: Literal['func', 'anat', 'eeg', 'physio'],
        basedir: str,
        subject: str,
        func_ext: Literal['nii', 'gii'] = 'nii',
        hemi: Literal['nii', 'gii'] = None,
        return_echos: bool = False,
        session: str = None,
    ) -> str:
        """
        take parameters and return file path

        Parameters
        ----------
        data: Literal['func', 'anat', 'eeg', 'physio']
            Choice of data modality
        basedir: str
            prepend filepath to directory
        subject: str
            subject label
        func_ext:  Literal['nii', 'gii']
            whether you want a .nii or .gii extension for the functional
            file path. Ignored if data modality not 'func' (default: nii)
        hemi: Literal[rh, lh]
            left or right hemisphere. Ignored if data modality not 'func'
        return_echos: bool
            whether to return filepaths for individual echos (default: False)
        session: str
            session label (optional)

        """
        if return_echos:
            fp = [
                self.iter.to_file(
                    data=data, subject=subject, session=session,
                    basedir=basedir, echo=e+1, func_ext=func_ext
                )
                for e in range(len(self.params['func']['echos']))
            ]
            # get full path
            fp = [os.path.abspath(f) for f in fp]
            if len(fp) < 2:
                fp = fp[0]
        else:
            fp = self.iter.to_file(
                data=data, subject=subject, session=session,
                basedir=basedir, func_ext=func_ext, hemi=hemi
            )
            # get full path
            fp = os.path.abspath(fp)
        return fp


class Pipeline:
    """
    Master pipeline for orchestrating preprocessing of functional,
    anatomical (T1w), physiological and EEG data.
    Dataset specific parameters are set in scan/meta/params.json.

    Attributes
    ----------
    dataset: str
        chosen dataset
    anat_skip: bool
        whether to skip anatomical preprocessing, if already run
        (default = False)
    reconall_skip: bool
        whether to skip reconall preprocessing, if already run
        (default: False)
    func_skip: bool
        whether to skip functional preprocessing, if already run
        (default = False)
    Methods
    -------

    run():
        Iterate through scans sequentially and execute preprocessing

    run_parallel(n_cores: int):
        Iterate through scans in parallel and execute preprocessing
    """

    def __init__(
        self,
        dataset: Literal['vanderbilt'],
        anat_skip: bool = False,
        func_skip: bool = False,
        reconall_skip: bool = False
    ):
        self.dataset = dataset
        self.anat_skip = anat_skip
        self.func_skip = func_skip
        self.reconall_skip = reconall_skip
        # dataset specific parameters
        with open('scan/meta/params.json', 'rb') as f:
            self.params = json.load(f)[dataset]
        self.multiecho = self.params['multiecho']
        # if multiecho, get echos
        if self.multiecho:
            self.echos = self.params['func']['echos']
        else:
            self.echos = [None]
        # get mapping to output files
        self.file_iter = FileMapper(dataset, self.params)
        self.file_map = self.file_iter.map()

    def run(self) -> None:
        """
        Execute full preprocessing pipeline sequentially over all subjects
        """
        # loop through subjects then sessions
        for subj, ses_list in self.file_iter.iter.subject_by_session():
            # initialized and execute anatomical pipeline
            anat_proc = AnatomicalPipeline(
                subj=subj, params=self.params, fmap=self.file_map,
                reconall_skip=self.reconall_skip
            )
            if not self.anat_skip:
                anat_proc.run()
            # loop through sessions and preprocess functionals
            for ses in ses_list:
                func_proc = FunctionalPipeline(
                    subj=subj, ses=ses, params=self.params,
                    fmap=self.file_map
                )
                if not self.func_skip:
                    func_proc.run()

    def run_parallel(self, n_cores: int = 4) -> None:
        """
        Execute full preprocessing pipeline in parallel over all subjects

        Parameters
        ----------
        n_cores: ints
            number of cores to run in parallel
        """
        # loop through subjects then sessions
        anat_proc_params = []
        func_proc_params = []
        for subj, ses_list in self.file_iter.iter.subject_by_session():
            # initialized and execute anatomical pipeline
            anat_p = {
                'subj': subj,
                'params': self.params,
                'fmap': self.file_map,
                'reconall_skip': self.reconall_skip
            }
            anat_proc_params.append(anat_p)
            # loop through sessions and preprocess functionals
            for ses in ses_list:
                func_p = {
                    'subj': subj,
                    'ses': ses,
                    'params': self.params,
                    'fmap': self.file_map
                }
                func_proc_params.append(func_p)

        # parallel execution
        # define preproc execution function
        # execute anatomical pipeline
        if not self.anat_skip:
            with Pool(n_cores) as pool:
                pool.map(_par_execute_anat, anat_proc_params)

        # execute functional pipeline
        if not self.func_skip:
            with Pool(n_cores) as pool:
                pool.map(_par_execute_func, func_proc_params)





class AnatomicalPipeline:
    """
    Freesurfer recon-all T1w preprocessing

    Attributes
    ----------
    subj: str
        subject label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths
    reconall_skip: bool
        whether to skip reconall preprocessing, if already run
        (default: False)

    Methods
    -------
    run():
        Execute anatomical preprocessing pipeline

    """
    def __init__(
        self,
        subj: str,
        params: dict,
        fmap: dict,
        reconall_skip: bool = False
    ):
        self.subj = subj
        self.fmap = fmap
        self.reconall_skip = reconall_skip
        # get anatomical preprocessing params
        self.anat_params = params['anat']
        # get anat directory
        self.anat_dir = params['directory']['anat']

    def run(self) -> None:
        """
        Execute anatomical preprocessing pipeline
        """
        # freesurfer recon-all
        if not self.reconall_skip:
            fs.recon_all(
                t1 = self.fmap['anat'][self.subj]['raw'],
                subj_label = self.subj,
                dir_out = self.fmap['anat'][self.subj]['freesurfer']
            )
        # create midthickness files for left and right hemispheres
        for hemi in ['lh', 'rh']:
            wb.create_midthickness(
                hemi = hemi,
                fs_mid = self.fmap['anat'][self.subj]['misc'][f'fs_mid_{hemi}'],
                lr_mid = self.fmap['anat'][self.subj]['misc'][f'fslr_mid_{hemi}'],
                sphere_out = self.fmap['anat'][self.subj]['misc'][f'fs_sphere_{hemi}'],
                fs_subj_dir = self.fmap['anat'][self.subj]['freesurfer_sdir']
            )


class FunctionalPipeline:
    """
    Functional preprocessing pipeline for a single scan.
    Steps:
        1. Head motion correction (mcflirt)
        2. Coregistration of mean functional volume to T1w (bbregister)
        3. Slice-timing correction
        4. If multiecho, tedana preprocessing
        5. Volume-to-surface sampling (SampleToSurface; mri_vol2surf)
        6. Surface smoothing

    Attributes
    ----------
    subj: str
        subject label
    ses: str
        session label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths

    """
    def __init__(
        self,
        subj: str,
        ses: str,
        params: dict,
        fmap: dict,
    ):
        self.subj = subj
        self.ses = ses
        self.fmap = fmap
        # get directory paths
        self.func_dir = params['directory']['func']
        # get functional preprocessing params
        self.func_params = params['func']
        # check whether multiecho dataset
        self.multiecho = params['multiecho']



    def run(self):
        """
        Execute functional preprocessing pipeline
        """
        # trim first N volumes
        if self.multiecho:
            fsl.trim_vol_multiecho(
                fps=self.fmap['func'][self.subj][self.ses]['raw'],
                fps_out=self.fmap['func'][self.subj][self.ses]['trim'],
                n_trim=self.func_params['trim']
            )
        else:
            fsl.trim_vol(
                fp=self.fmap['func'][self.subj][self.ses]['raw'],
                fp_out=self.fmap['func'][self.subj][self.ses]['trim'],
                n_trim=self.func_params['trim']
            )

        # slicetiming correction w/ FSL slicetimer
        if self.multiecho:
            fsl.slicetime_multiecho(
                fps=self.fmap['func'][self.subj][self.ses]['trim'],
                fps_out=self.fmap['func'][self.subj][self.ses]['slicetime'],
                slice_order=self.func_params['sliceorder'],
                tr = self.func_params['tr']
            )
        else:
            fsl.slicetime(
                fp=self.fmap['func'][self.subj][self.ses]['trim'],
                fp_out=self.fmap['func'][self.subj][self.ses]['slicetime'],
                slice_order=self.func_params['sliceorder'],
                tr = self.func_params['tr']
            )

        # motion correction with FSL MCLFIRT
        if self.multiecho:
            fsl.mcflirt_multiecho(
                fps=self.fmap['func'][self.subj][self.ses]['slicetime'],
                fps_out=self.fmap['func'][self.subj][self.ses]['motion'],
                fp_meanvol=self.fmap['func'][self.subj][self.ses]['misc']['mean_vol'],
                fp_mat=self.fmap['func'][self.subj][self.ses]['misc']['mcflirt_mat']
            )
        else:
            fsl.mcflirt(
                fp=self.fmap['func'][self.subj][self.ses]['slicetime'],
                fp_out=self.fmap['func'][self.subj][self.ses]['motion']
            )

        # coregistration of mean functional to T1w with Freesurfer BBRegister
        fs.surf_register(
            fp_reg_out = self.fmap['func'][self.subj][self.ses]['misc']['surfregister_reg'],
            fp_mat_out = self.fmap['func'][self.subj][self.ses]['misc']['surfregister_mat'],
            fp_fslmat_out = self.fmap['func'][self.subj][self.ses]['misc']['surfregister_fslmat'],
            func_mean = self.fmap['func'][self.subj][self.ses]['misc']['mean_vol'],
            subj=self.subj,
            subjects_dir=self.fmap['anat'][self.subj]['freesurfer']
        )

        # if multiecho, run tedana pipeline
        if self.multiecho:
            # create binary brain mask from Freesurfer in functional space
            fs.mask_to_func(
                mask = self.fmap['anat'][self.subj]['misc']['fs_brainmask'],
                func_mean = self.fmap['func'][self.subj][self.ses]['misc']['mean_vol'],
                mask_out = self.fmap['func'][self.subj][self.ses]['misc']['fs_func_mask'],
                mat_lta = self.fmap['func'][self.subj][self.ses]['misc']['surfregister_mat']
            )
            # run tedana workflow
            tedana_prefix = os.path.basename(
                self.fmap['func'][self.subj][self.ses]['tedana']
            )
            tedana_denoise(
                fps_in = self.fmap['func'][self.subj][self.ses]['motion'],
                echo_times = self.func_params['echos'],
                mask = self.fmap['func'][self.subj][self.ses]['misc']['fs_func_mask'],
                out_dir = self.fmap['func'][self.subj][self.ses]['tedana'],
                out_prefix = tedana_prefix
            )
            fp_vol2surf = self.fmap['func'][self.subj][self.ses]['misc']['tedana_denoised']
        else:
            fp_vol2surf = self.fmap['func'][self.subj][self.ses]['motion']

        # from here on, we process separate hemispheres (lh, rh)
        for hemi in ['lh', 'rh']:
            # from volume to native (subject) surface with Freesurfer
            # mri_vol2surf
            fs.vol2surf(
                hemi = hemi,
                smooth_fwhm = self.func_params['smooth_fwhm'],
                fp_in = fp_vol2surf,
                fp_out =self.fmap['func'][self.subj][self.ses]['surface_smooth'][hemi],
                subj=self.subj,
                subjects_dir=self.fmap['anat'][self.subj]['freesurfer'],
                fp_reg_mat = self.fmap['func'][self.subj][self.ses]['misc']['surfregister_mat']
            )
            # freesurfer native surface to fs_LR surface with
            # workbench metric-resample
            wb.fs2fslr(
                hemi = hemi,
                fp_in = self.fmap['func'][self.subj][self.ses]['surface_smooth'][hemi],
                fp_out =self.fmap['func'][self.subj][self.ses]['surface_lr'][hemi],
                fs_mid = self.fmap['anat'][self.subj]['misc'][f'fs_mid_{hemi}'],
                lr_mid = self.fmap['anat'][self.subj]['misc'][f'fslr_mid_{hemi}'],
                fs_sphere = self.fmap['anat'][self.subj]['misc'][f'fs_sphere_{hemi}'],
                fs_subj_dir = self.fmap['anat'][self.subj]['freesurfer_sdir']

            )


def _par_execute_func(params: str) -> None:
    # execution of functional pipeline for run_parallel()
    func_proc = FunctionalPipeline(**params)
    func_proc.run()


def _par_execute_anat(params: str) -> None:
    # execution of anatomical pipeline for run_parallel()
    anat_proc = AnatomicalPipeline(**params)
    anat_proc.run()











