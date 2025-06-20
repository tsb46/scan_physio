"""
Utilites for preprocessing data
"""
from scan.preprocess import fsl
from scan.preprocess import freesurfer
from scan.preprocess import dataset
from scan.preprocess import physio
from scan.preprocess import workbench
from scan.preprocess import custom
from scan.preprocess.pipeline import Pipeline


__all__ = [
    'fsl',
    'freesurfer',
    'dataset',
    'physio',
    'workbench',
    'custom',
    'Pipeline'
]
