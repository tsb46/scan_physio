"""
Module for estimating seed-based functional connectivity modulation
by a continuous signal interaction.
"""

import os
import pickle

from typing import List

import numpy as np

from sklearn.linear_model import LinearRegression

from scan.io.load import Gifti
from scan.io.write import write_func_gii

DEFAULT_MODULATOR_PERCENTILES = np.array([5, 25, 50, 75, 95], dtype=float)


class FCInteractionResults:
    """
    Class for storing results of seed-by-modulator interaction analysis.

    Attributes
    ----------
    coefficient_maps: List[np.ndarray]
        Model coefficient maps for the seed, modulator, and interaction terms.
    coefficient_labels: List[str]
        Labels for the coefficient maps.
    percentile_fc_maps: List[np.ndarray]
        Seed-based FC maps evaluated at selected modulator percentiles.
    percentile_labels: List[str]
        Labels for the percentile-specific FC maps.
    modulator_percentiles: np.ndarray
        Percentiles used for the FC summaries.
    modulator_percentile_values: np.ndarray
        Raw modulator values at the requested percentiles.
    modulator_percentile_centered_values: np.ndarray
        Mean-centered modulator values at the requested percentiles.
    modulator_mean: float
        Mean used to center the modulator.
    """

    def __init__(
        self,
        coefficient_maps: List[np.ndarray],
        coefficient_labels: List[str],
        percentile_fc_maps: List[np.ndarray],
        percentile_labels: List[str],
        modulator_percentiles: np.ndarray,
        modulator_percentile_values: np.ndarray,
        modulator_percentile_centered_values: np.ndarray,
        modulator_mean: float,
    ):
        self.coefficient_maps = coefficient_maps
        self.coefficient_labels = coefficient_labels
        self.percentile_fc_maps = percentile_fc_maps
        self.percentile_labels = percentile_labels
        self.modulator_percentiles = modulator_percentiles
        self.modulator_percentile_values = modulator_percentile_values
        self.modulator_percentile_centered_values = modulator_percentile_centered_values
        self.modulator_mean = modulator_mean

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str | None = None,
        out_dir: str | None = None,
    ) -> None:
        """
        Write out interaction results to func.gii files.
        """
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f"{out_dir}/{file_prefix}"

        metadata = {
            "modulator_percentiles": self.modulator_percentiles,
            "modulator_percentile_values": self.modulator_percentile_values,
            "modulator_percentile_centered_values": self.modulator_percentile_centered_values,
            "modulator_mean": self.modulator_mean,
            "coefficient_labels": self.coefficient_labels,
            "percentile_labels": self.percentile_labels,
        }
        with open(f"{out_prefix}.pkl", "wb") as f:
            pickle.dump(metadata, f)

        for label, coef in zip(self.coefficient_labels, self.coefficient_maps):
            write_func_gii(coef[np.newaxis, :], gii_params, f"{out_prefix}_{label}")

        for label, fc in zip(self.percentile_labels, self.percentile_fc_maps):
            write_func_gii(fc[np.newaxis, :], gii_params, f"{out_prefix}_{label}")


class FCMapResults:
    """
    Class for storing results of simple seed-based functional connectivity.

    Attributes
    ----------
    fc_map: np.ndarray
        Seed-based functional connectivity map.
    model_params: dict
        Parameters used to fit the model.
    """

    def __init__(self, fc_map: np.ndarray, model_params: dict):
        self.fc_map = fc_map
        self.model_params = model_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str | None = None,
        out_dir: str | None = None,
    ) -> None:
        """
        Write out a simple seed-based functional connectivity map to func.gii.
        """
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f"{out_dir}/{file_prefix}"

        with open(f"{out_prefix}.pkl", "wb") as f:
            pickle.dump(self.model_params, f)

        write_func_gii(self.fc_map[np.newaxis, :], gii_params, out_prefix)


class FCMapModel:
    """
    Class for estimating simple seed-based functional connectivity.

    Attributes
    ----------
    model : LinearRegression
        The fitted linear regression model.
    """

    def fit(self, seed_ts: np.ndarray, func_data: np.ndarray) -> FCMapResults:
        """
        Fit a simple seed-based FC model.

        Parameters
        ----------
        seed_ts: np.ndarray
            1d time series data for a seed region of interest (ROI).
        func_data : np.ndarray
            Functional MRI data: a 2D array where rows are time points and columns
            are vertices or regions.

        Returns
        -------
        FCMapResults
            Object containing the seed-based FC map.
        """
        seed_ts = np.asarray(seed_ts)
        func_data = np.asarray(func_data)

        if func_data.ndim != 2:
            raise ValueError("func_data must be a 2d array.")

        if seed_ts.ndim == 1:
            seed_ts = seed_ts[:, np.newaxis]
        elif seed_ts.ndim == 2 and seed_ts.shape[1] == 1:
            pass
        else:
            raise ValueError("seed_ts must be a 1d array or a single-column 2d array.")

        if seed_ts.shape[0] != func_data.shape[0]:
            raise ValueError(
                "seed_ts and func_data must have the same number of time points."
            )

        self.model = LinearRegression()
        self.model.fit(seed_ts, func_data)

        fc_map = np.asarray(self.model.coef_).squeeze()

        return FCMapResults(
            fc_map=fc_map,
            model_params={
                "model": "linear_regression",
                "n_timepoints": int(func_data.shape[0]),
                "n_vertices": int(func_data.shape[1]),
            },
        )


class FCInteractionModel:
    """
    Class for estimating seed-based functional connectivity modulation
    by a continuous modulator signal.

    Attributes
    ----------
    modulator_percentiles : np.ndarray
        Percentiles used for evaluating simple slopes of seed-based FC.
    """

    def __init__(
        self,
        modulator_percentiles: np.ndarray | None = None,
    ):
        if modulator_percentiles is None:
            modulator_percentiles = DEFAULT_MODULATOR_PERCENTILES

        modulator_percentiles = np.asarray(modulator_percentiles, dtype=float).squeeze()
        if modulator_percentiles.ndim != 1:
            raise ValueError("modulator_percentiles must be a 1d array.")

        self.modulator_percentiles = modulator_percentiles
        self.modulator = None
        self.modulator_mean = None
        self.modulator_centered = None
        self.modulator_percentile_values = None
        self.modulator_percentile_centered_values = None

    @staticmethod
    def _format_percentile_label(percentile: float) -> str:
        percentile_int = int(round(percentile))
        return f"mod_p{percentile_int:02d}"

    def fit(
        self,
        seed_ts: np.ndarray,
        func_data: np.ndarray,
        modulator: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> FCInteractionResults:
        """
        Estimate seed-based functional connectivity modulation using a linear
        interaction model.

        Parameters
        ----------
        func_data : np.ndarray
            Functional MRI data: a 2D array where rows are time points and columns
            are vertices or regions.
        seed_ts: np.ndarray
            1d time series data for a seed region of interest (ROI).
        modulator: np.ndarray
            1d time series used as the continuous moderator.
        weights: np.ndarray, optional
            1d array of weights for each time point, used in weighted regression.
             If None, ordinary least squares regression is performed.

        Returns
        -------
        FCInteractionResults
            Object containing coefficient maps and percentile-specific FC maps.
        """
        seed_ts = np.asarray(seed_ts)
        func_data = np.asarray(func_data)

        if func_data.ndim != 2:
            raise ValueError("func_data must be a 2d array.")

        if seed_ts.ndim == 1:
            seed_ts = seed_ts[:, np.newaxis]
        elif seed_ts.ndim == 2 and seed_ts.shape[1] == 1:
            pass
        else:
            raise ValueError("seed_ts must be a 1d array or a single-column 2d array.")

        if seed_ts.shape[0] != func_data.shape[0]:
            raise ValueError(
                "seed_ts and func_data must have the same number of time points."
            )

        modulator = np.asarray(modulator).squeeze()
        if modulator.ndim != 1:
            raise ValueError("modulator must be a 1d array.")

        if modulator.shape[0] != func_data.shape[0]:
            raise ValueError(
                "modulator must have the same number of time points as func_data."
            )

        self.modulator = modulator
        self.modulator_mean = float(np.mean(modulator))
        self.modulator_centered = modulator - self.modulator_mean
        self.modulator_percentile_values = np.percentile(
            self.modulator, self.modulator_percentiles
        )
        self.modulator_percentile_centered_values = (
            self.modulator_percentile_values - self.modulator_mean
        )

        modulator_term = self.modulator_centered[:, np.newaxis]
        interaction_term = seed_ts * modulator_term
        design_matrix = np.hstack([seed_ts, modulator_term, interaction_term])

        self.model = LinearRegression()
        self.model.fit(design_matrix, func_data, sample_weight=weights)

        coefficients = np.asarray(self.model.coef_)
        seed_beta = coefficients[:, 0]
        modulator_beta = coefficients[:, 1]
        interaction_beta = coefficients[:, 2]

        percentile_fc_maps = [
            seed_beta + interaction_beta * centered_value
            for centered_value in self.modulator_percentile_centered_values
        ]
        percentile_labels = [
            self._format_percentile_label(percentile)
            for percentile in self.modulator_percentiles
        ]

        return FCInteractionResults(
            coefficient_maps=[seed_beta, modulator_beta, interaction_beta],
            coefficient_labels=["seed_beta", "modulator_beta", "interaction_beta"],
            percentile_fc_maps=percentile_fc_maps,
            percentile_labels=percentile_labels,
            modulator_percentiles=self.modulator_percentiles,
            modulator_percentile_values=self.modulator_percentile_values,
            modulator_percentile_centered_values=self.modulator_percentile_centered_values,
            modulator_mean=self.modulator_mean,
        )
