"""
Module for loading data (func, eeg, etc.), handling data file paths
and iterating subject lists.
"""

import json
from typing import Literal

import pandas as pd


class Participant:
    """
    Class for loading and iterating
    through participant lists.

    Attributes
    ----------
    dataset : str
        chosen dataset

    Methods
    -------
    subject():
        Iterate through subjects (ignore scans)

    subject_by_session():
        Iterate through subjects with individual sessions returned
        as nested list

    filepaths(data, basedir, echo):
        Iterate through filepaths

    to_file(data, subject, session, echo=None):
        Get filepath based on parameters

    """

    # define available participant list fields
    avail_fields = ['subject', 'session', 'echo']

    def __init__(self, dataset: Literal['vanderbilt']):
        # load subject list for chosen dataset
        df = pd.read_csv(
            f'scan/meta/{dataset}_participant.csv',
            dtype=str
        )
        # subject is necessary field and should be first column
        if 'subject' not in df.columns:
            raise ValueError('"subject" column missing from participant list')

        # get location of subject column
        self.subject_loc = df.columns.get_loc('subject')

        # check participant fields
        self.fields = []
        for c in df.columns:
            if c not in self.avail_fields:
                raise ValueError(
                    f"{dataset} subject list column '{c}' not recognized"
                )
            self.fields.append(c)

        self.dataset = dataset
        # get values for fields from every row in nested list
        self.values = list(zip(*[df[c] for c in self.fields]))
        self.n_scans = len(self.values)
        # get data formatting
        with open('scan/meta/params.json', 'rb') as f:
            params = json.load(f)[dataset]
        self.file_format = params['file_format']
        # check if multiecho dataset
        self.multiecho = params['multiecho']
        # determine number of echos, if applicable
        if self.multiecho:
            echos = params['func']['echos']
            self.n_echos = len(echos)
        else:
            self.n_echos = None

    def __iter__(self):
        """
        iterate through rows of
        subject list in order
        """
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        if self.n < self.n_scans:
            return self.values[self.n]
        raise StopIteration

    def subject(self):
        """
        iterate through subjects in order
        """
        # does not reorder
        subjects = list(
            dict.fromkeys([s[self.subject_loc] for s in self.values])
        )
        n_subj = len(subjects)
        n = 0
        while n < n_subj:
            yield subjects[n]
            n += 1

    def subject_by_session(self):
        """
        Iterate through subjects and return
        (possible) multiple sessions as a nested
        list
        """
        if 'session' not in self.fields:
            raise ValueError(f'{self.dataset} has no "session" field')
        session_loc = self.fields.index('session')
        subjects = list(self.subject())
        n_subj = len(subjects)
        n = 0
        while n < n_subj:
            subj = subjects[n]
            subj_ses = list(dict.fromkeys(
                [s[session_loc] for s in self.values
                if s[self.subject_loc] == subj]
            ))
            yield (subj, subj_ses)
            n += 1

    def filepath(
        self,
        data: Literal['func', 'anat', 'eeg', 'physio'],
        basedir: str = None,
        echo: bool = False,
    ):
        """
        iterate through rows and return file path to
        data object (eeg, anat, func or physio). If specified,
        prepend with a directory path with basedir. If multiecho,
        return for all echo fps as a list by setting echo = True.
        """
        # initialize counter
        n = 0
        while n < self.n_scans:
            # get values for each field per row
            ss = dict(zip(self.fields, self.values[n]))
            # if specified, loop through echos
            if (data == 'func') & echo & self.multiecho:
                fp_echos = []
                for n_e in range(self.n_echos):
                    # assuming echo label always start at 0
                    # may not always be the case
                    ss['echo'] = n_e + 1
                    # path to file, with echo
                    fp = self.to_file(data, basedir=basedir, **ss)
                    fp_echos.append(fp)
                yield fp_echos
                n += 1
            # if echo not specified
            else:
                fp = self.to_file(data, basedir=basedir, **ss)
                yield fp
                n += 1

    def to_file(
        self,
        data: Literal['func', 'anat', 'eeg', 'physio'],
        subject: str,
        func_ext: Literal['nii', 'gii'] = 'nii',
        hemi: Literal['nii', 'gii'] = None,
        session: str = None,
        echo: int = None,
        basedir: str = None
    ) -> str:
        """
        take fields (e.g. subject, session, echo) and return file path.
        If echo is provided, return file path to specific echo. If
        specified, prepend with a directory path with basedir.

        Parameters
        ----------
        data: Literal['func', 'anat', 'eeg', 'physio']
            Choice of data modality
        subject: str
            subject label
        func_ext:  Literal['nii', 'gii']
            whether you want a .nii or .gii extension for the functional
            file path. Ignored if data modality not 'func' (default: nii)
        hemi: Literal[rh, lh]
            left or right hemisphere. Ignored if data modality not 'func'
        session: str
            session label (optional)
        echo: int
            echo label (optional)
        basedir: str
            prepend filepath to directory (optional)
        """
        if basedir is not None:
            basepath = basedir + '/'
        else:
            basepath = ''
        if (data == 'func') & self.multiecho:
            if echo is not None:
                fp = self.file_format[data]['echo'].format(
                    subject=subject, session=session, echo=echo
                )
            else:
                if func_ext == 'nii':
                    fp = self.file_format[data]['nii'].format(
                        subject=subject, session=session
                    )
                elif func_ext == 'gii':
                    fp = self.file_format[data]['gii'].format(
                        subject=subject, session=session,
                        hemi=hemi
                    )
                else:
                    raise ValueError("nii or gii are the only options for hemi")
        else:
            fp = self.file_format[data].format(
                subject=subject, session=session
            )
        return f'{basepath}{fp}'




