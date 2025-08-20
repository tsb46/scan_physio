"""
Module for estimating the modulation of seed-based functional 
connectivity by sleep stages.
"""

import numpy as np

from sklearn.linear_model import LinearRegression

# define sleep stages
SLEEP_STAGES = {
    1: "alert",
    0: "intermediate",
    -1: "drowsy_w_n1",
    -2: "drowsy_sleep"
}

class FCModulation:
    """
    Class for estimating the modulation of seed-based 
    functional connectivity by sleep stages.

    Attributes
    ----------
    sleep_stages : list[int]
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
        sleep_stages: list[int],
        reference_stage: int = 0
    ):
        self.sleep_stages = sleep_stages
        # get unique sleep states
        self.unique_sleep_stages = np.unique(sleep_stages)
        # get count of sleep stage occurrences
        self.sleep_stage_counts = np.unique(sleep_stages, return_counts=True)[1]
        # check that each sleep stage is represented in the data
        for stage in SLEEP_STAGES:
            if stage not in self.unique_sleep_stages:
                raise ValueError(f"Sleep stage {SLEEP_STAGES[stage]} is not represented in the data.")
        
        # check that reference stage is allowed
        if reference_stage not in SLEEP_STAGES:
            raise ValueError(f"Reference stage {reference_stage} is not a valid sleep stage.")

        # create dummy matrix of sleep stage vectors
        self.sleep_stage_matrix = np.zeros((len(self.sleep_stages), len(SLEEP_STAGES)))
        self.sleep_stage_matrix_cols = []
        col_i = 0
        for stage in self.sleep_stages:
            # skip reference stage
            if stage != reference_stage:
                # create mask of sleep stages
                stage_indx = self.sleep_stages == stage
                # set the corresponding column to 1 for the current stage
                self.sleep_stage_matrix[stage_indx, col_i] = 1
                col_i += 1

    def fit(self, seed_ts: np.ndarray, func_data: np.ndarray) -> None:
        """
        Estimate the modulation of functional connectivity by sleep stages.

        func_data : np.ndarray
            Functional MRI data: a 2D array where rows are time points and columns are
            time points.
        seed_ts: np.ndarray
            1d time series data for a seed region of interest (ROI).

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

        # create interaction matrix of seed and sleep stage vectors
        interaction_matrix = np.zeros(self.sleep_stage_matrix.shape)
        for i in range(self.sleep_stage_matrix.shape[1]):
            interaction_matrix[:, i] = seed_ts * self.sleep_stage_matrix[:, i]
        
        # create design matrix from seed_ts, sleep stage matrix and interaction matrix
        self.design_matrix = np.hstack(
            [seed_ts, self.sleep_stage_matrix, interaction_matrix]
        )
        # get indices of interaction terms of design matrix
        self.interaction_indices = np.arange(
            seed_ts.shape[1] + self.sleep_stage_matrix.shape[1], self.design_matrix.shape[1]
        )
        # fit linear regression model
        self.model = LinearRegression()
        self.model.fit(self.design_matrix, func_data)
