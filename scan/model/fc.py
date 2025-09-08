"""
Module for estimating the modulation of seed-based functional 
connectivity by sleep stages.
"""

import numpy as np

from sklearn.linear_model import LinearRegression

from scan.io.write import FCStageResults

# define sleep stages
SLEEP_STAGES = {
    1: "alert",
    0: "intermediate",
    -1: "drowsy",
}

class FCModulation:
    """
    Class for estimating the modulation of seed-based 
    functional connectivity by sleep stages.

    Attributes
    ----------
    sleep_stages : np.ndarray
        1d array of sleep stages
        Sleep stages represented as integers:
        - 1 = VIGALL alert
        - 0 = VIGALL intermediate
        - -1 = VIGALL drowsy AND (U-Sleep W or N1)
        - -2 = VIGALL drowsy AND (U-Sleep N2, N3, or REM)
        Each time point is assigned to one of these stages, and should be equal to
        the length of the functional MRI data.
    """

    def __init__(
        self, 
        sleep_stages: np.ndarray
    ):
        # convert sleep stage -2 to -1
        sleep_stages = np.where(sleep_stages == -2, -1, sleep_stages)
        self.sleep_stages = sleep_stages
        # get unique sleep stages
        unique_sleep_stages = np.unique(sleep_stages)
        # get count of sleep stage occurrences
        self.sleep_stage_counts = np.unique(sleep_stages, return_counts=True)[1]
        # check that each sleep stage is represented in the data
        for stage in SLEEP_STAGES:
            if stage not in unique_sleep_stages:
                raise ValueError(f"Sleep stage {SLEEP_STAGES[stage]} is not represented in the data.")


    def estimate_fc_modulation(self, seed_ts: np.ndarray, func_data: np.ndarray) -> FCStageResults:
        """
        Estimate the modulation of functional connectivity by sleep stages.

        Parameters
        ----------
        func_data : np.ndarray
            Functional MRI data: a 2D array where rows are time points and columns are
            time points.
        seed_ts: np.ndarray
            1d time series data for a seed region of interest (ROI).

        Returns
        -------
        FCModulationResults
            An object containing the results of the functional connectivity modulation analysis.
        """
        # check that seed_ts and func_data have the same number of time points
        if len(seed_ts) != func_data.shape[0]:
            raise ValueError("seed_ts and func_data must have the same number of time points.")
        # if seed_ts is 1d, convert to 2d array with one column
        if seed_ts.ndim == 1:
            seed_ts = seed_ts[:, np.newaxis]
        # check that sleep_stages has the same number of time points as func_data
        if len(self.sleep_stages) != func_data.shape[0]:
            raise ValueError("sleep_stages must have the same number of time points as func_data.")

        # Loop through sleep stages and compute seed-based functional connectivity
        fc_stages = []
        stage_labels = []
        for stage in SLEEP_STAGES:
            # create mask for current sleep stage
            stage_mask = self.sleep_stages == stage
            # fit linear regression model
            self.model = LinearRegression()
            self.model.fit(seed_ts[stage_mask], func_data[stage_mask])
            fc_stages.append(self.model.coef_)
            stage_labels.append(SLEEP_STAGES[stage])

        # return results
        return FCStageResults(
            fc_stages=fc_stages,
            stage_labels=stage_labels
        )

