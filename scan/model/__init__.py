"""
Module containing regression models for modeling the relationship between
physio signals and functional MRI signals.
"""

from scan.model import corr
from scan.model import cluster

from scan.model.corr import DistributedLagModel
from scan.model.corr import DistributedLagNonLinearModel

__all__ = [
    'corr',
    'cluster',
    'DistributedLagModel',
    'DistributedLagNonLinearModel'
]
