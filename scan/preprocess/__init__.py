"""
Utilites for preprocessing data
"""
from scan.preprocess import fsl
from scan.preprocess import freesurfer
from scan.preprocess.pipeline import Pipeline

__all__ = [
    'fsl',
    'freesurfer',
    'Pipeline'
]
