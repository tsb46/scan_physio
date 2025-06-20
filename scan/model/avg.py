"""
Module for calculating windowed averages of functional MRI signals
around physiological events
"""
from typing import List

import numpy as np

from scan.io.write import WindowAverageResults


class WindowAverage:
    """
    Calculate windowed averages of functional MRI signals
    around physiological events
    """

    def __init__(
        self, 
        left_edge: int = 10,
        right_edge: int = 10,
    ):
        """
        Parameters
        ----------
        left_edge: int
            Number of timepoints to include to the left of the physiological event
        right_edge: int
            Number of timepoints to include to the right of the physiological event
        """
        if left_edge < 0 or right_edge < 0:
            raise ValueError("left_edge and right_edge must be non-negative")

        self.left_edge = left_edge
        self.right_edge = right_edge

    def calculate_avg(
        self, 
        func_data: np.ndarray, 
        physio_events: List[int]
    ) -> WindowAverageResults:
        """
        Calculate windowed averages of functional MRI signals
        around physiological events
        """
         # index windows for each marker
        windows = []
        for center in physio_events:
            windows.append(
                extract_range(func_data, center, -self.left_edge, self.right_edge)
            )
        # convert to 3d array
        windows = np.stack(windows, axis=0)

        # average all windows
        w_avg = np.nanmean(windows, axis=0)

        window_average_params = {
            'left_edge': self.left_edge,
            'right_edge': self.right_edge,
            'physio_events': physio_events,
        }

        return WindowAverageResults(w_avg, window_average_params)


def extract_range(
    array: np.ndarray, 
    center: int, 
    left_edge: int, 
    right_edge: int
) -> np.ndarray:
    """
    Extract a range of rows from an array with NaN padding for out-of-bound indices

    Parameters
    ----------
    array : np.ndarray
        array to extract range from
    center : int
        center of the range
    left_edge : int
        left edge of the range
    right_edge : int
        right edge of the range

    Returns
    -------
    padded_range : np.ndarray
        array with NaN padding for out-of-bound indices
    """
    num_rows, num_cols = array.shape
    range_size = right_edge - left_edge + 1
    # Create a NaN-filled placeholder for the range
    padded_range = np.full((range_size, num_cols), np.nan)

    # Calculate valid row indices
    start = max(0, center + left_edge)  # Valid start index in the array
    end = min(num_rows, center + right_edge + 1)  # Valid end index in the array
    insert_start = max(0, -1 * (center + left_edge))  # Where to insert in padded_range
    insert_end = insert_start + (end - start)  # End position in padded_range

    # Insert valid rows into the padded range
    padded_range[insert_start:insert_end, :] = array[start:end, :]
    return padded_range