"""
Module containing regression models for modeling the relationship between
physio signals and functional MRI signals.
"""

from scan.model import corr

from scan.model.corr import DistributedLagModel
from scan.model.corr import MultivariateDistributedLagModel
from scan.model.complex_pca import ComplexPCA

__all__ = [
    'corr',
    'DistributedLagModel',
    'MultivariateDistributedLagModel',
    'ComplexPCA'
]
