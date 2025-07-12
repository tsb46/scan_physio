"""
Module for calculating windowed averages of functional MRI signals
around physiological events
"""
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
from scipy import signal

from scan.io.write import WindowAverageResults


@dataclass
class EventFinderResults:
    """
    Results from event finding in physiological signals
    
    Parameters
    ----------
    events : Dict[str, List[int]]
        Dictionary mapping signal names to lists of event indices
    co_localization_stats : Dict[str, Dict]
        Statistics about co-localized events between signals
    event_finder_params : Dict
        Parameters used for event finding
    """
    events: Dict[str, List[int]]
    co_localization_stats: Dict[str, Dict]
    event_finder_params: Dict


class EventFinder:
    """
    Detect physiological events using peak detection algorithms.
    
    This class provides methods to find peaks in physiological signals
    that exceed given thresholds, with optional parameters to tune
    the peak-finding algorithm. It can handle multiple physiological
    signals and provides statistics for co-localized peaks between signals.
    """

    def __init__(
        self,
        threshold: Union[float, Dict[str, float]] = None,
        height: Union[float, Dict[str, float]] = None,
        distance: int = 1,
        prominence: Optional[Union[float, Dict[str, float]]] = None,
        width: Optional[Union[int, Dict[str, int]]] = None,
        rel_height: float = 0.5,
        plateau_size: Optional[int] = None,
        co_localization_tolerance: int = None,
        mask: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ):
        """
        Initialize EventFinder with peak detection parameters.
        
        Parameters
        ----------
        threshold : float or Dict[str, float], optional
            Required height of peaks. If dict, maps signal names to thresholds.
            If None, no threshold filtering is applied.
        height : float or Dict[str, float], optional
            Required height of peaks. If dict, maps signal names to heights.
            If None, no height filtering is applied.
        distance : int, optional
            Required minimal horizontal distance (>= 1) in samples between
            neighboring peaks. Smaller peaks are removed first until the
            condition is fulfilled for all remaining peaks.
        prominence : float or Dict[str, float], optional
            Required prominence of peaks. If dict, maps signal names to prominence.
            If None, no prominence filtering is applied.
        width : int or Dict[str, int], optional
            Required width of peaks in samples. If dict, maps signal names to widths.
            If None, no width filtering is applied.
        rel_height : float, optional
            Used for calculation of the peaks width, thus it is only used if
            width is given. See signal.peak_widths for a full description of
            its effects.
        plateau_size : int, optional
            A peak's maximum value must be higher than its neighbors by
            plateau_size samples on both sides. If None, no plateau filtering
            is applied.
        co_localization_tolerance : int, optional
            Tolerance window in samples for determining co-localized events
            between different signals. If None, uses the distance parameter.
        mask : np.ndarray or Dict[str, np.ndarray], optional
            Binary mask (1s and 0s) to exclude peaks at timepoints with 0 values.
            If dict, maps signal names to specific masks. Must match signal length.
            If None, no masking is applied.
        """
        self.threshold = threshold
        self.height = height
        self.distance = distance
        self.prominence = prominence
        self.width = width
        self.rel_height = rel_height
        self.plateau_size = plateau_size
        self.co_localization_tolerance = co_localization_tolerance
        self.mask = mask

    def find_events(
        self, 
        signals: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> EventFinderResults:
        """
        Find events in physiological signals using peak detection.
        
        Parameters
        ----------
        signals : np.ndarray or Dict[str, np.ndarray]
            Single signal array or dictionary mapping signal names to arrays.
            Each array should be 1D with timepoints in rows.
            
        Returns
        -------
        EventFinderResults
            Object containing detected events and co-localization statistics
        """
        if isinstance(signals, np.ndarray):
            # Single signal case
            signals = {'signal': signals}
        
        events = {}
        
        # Find peaks for each signal
        for signal_name, signal_data in signals.items():
            if signal_data.ndim != 1:
                raise ValueError(f"Signal {signal_name} must be 1D array")
            
            # Get parameters for this signal
            threshold = self._get_param_for_signal(self.threshold, signal_name)
            height = self._get_param_for_signal(self.height, signal_name)
            prominence = self._get_param_for_signal(self.prominence, signal_name)
            width = self._get_param_for_signal(self.width, signal_name)
            mask = self._get_param_for_signal(self.mask, signal_name)
            
            # Find peaks using scipy.signal.find_peaks
            peak_kwargs = {
                'distance': self.distance,
                'rel_height': self.rel_height,
            }
            
            if threshold is not None:
                peak_kwargs['threshold'] = threshold
            if height is not None:
                peak_kwargs['height'] = height
            if prominence is not None:
                peak_kwargs['prominence'] = prominence
            if width is not None:
                peak_kwargs['width'] = width
            if self.plateau_size is not None:
                peak_kwargs['plateau_size'] = self.plateau_size
            
            peaks, properties = signal.find_peaks(signal_data, **peak_kwargs)
            
            # Apply mask if provided
            if mask is not None:
                # Validate mask
                if len(mask) != len(signal_data):
                    raise ValueError(f"Mask length ({len(mask)}) must match signal length ({len(signal_data)}) for signal {signal_name}")
                
                # Filter peaks to only include those where mask is 1
                if len(peaks) > 0:
                    valid_peaks = []
                    for peak in peaks:
                        if peak < len(mask) and mask[peak] == 1:
                            valid_peaks.append(peak)
                    peaks = np.array(valid_peaks)
                else:
                    peaks = np.array([])
            
            events[signal_name] = peaks.tolist()
        
        # Calculate co-localization statistics if multiple signals
        co_localization_stats = {}
        if len(signals) > 1:
            co_localization_stats = self._calculate_co_localization_stats(events)
        
        # Prepare parameters for results
        event_finder_params = {
            'threshold': self.threshold,
            'height': self.height,
            'distance': self.distance,
            'prominence': self.prominence,
            'width': self.width,
            'rel_height': self.rel_height,
            'plateau_size': self.plateau_size,
            'co_localization_tolerance': self.co_localization_tolerance,
            'mask': self.mask,
        }
        
        return EventFinderResults(events, co_localization_stats, event_finder_params)

    def _get_param_for_signal(
        self, 
        param: Union[float, Dict[str, float], None], 
        signal_name: str
    ) -> Optional[float]:
        """Helper method to get parameter value for a specific signal."""
        if param is None:
            return None
        elif isinstance(param, dict):
            return param.get(signal_name, None)
        else:
            return param

    def _calculate_co_localization_stats(
        self, 
        events: Dict[str, List[int]]
    ) -> Dict[str, Dict]:
        """
        Calculate statistics about co-localized events between signals.
        
        Parameters
        ----------
        events : Dict[str, List[int]]
            Dictionary mapping signal names to lists of event indices
            
        Returns
        -------
        Dict[str, Dict]
            Statistics about co-localized events
        """
        signal_names = list(events.keys())
        co_localization_stats = {}
        
        # Calculate pairwise co-localization statistics
        for i, signal1 in enumerate(signal_names):
            for j, signal2 in enumerate(signal_names[i+1:], i+1):
                pair_name = f"{signal1}_vs_{signal2}"
                
                # Find co-localized events within a tolerance window
                tolerance = self.distance  # Use distance parameter as tolerance
                co_localized = self._find_co_localized_events(
                    events[signal1], events[signal2], tolerance
                )
                
                # Calculate statistics
                total_events_1 = len(events[signal1])
                total_events_2 = len(events[signal2])
                co_localized_count = len(co_localized)
                
                stats = {
                    'total_events_signal1': total_events_1,
                    'total_events_signal2': total_events_2,
                    'co_localized_events': co_localized_count,
                    'co_localized_indices': co_localized,
                    'co_localization_rate_signal1': co_localized_count / total_events_1 if total_events_1 > 0 else 0,
                    'co_localization_rate_signal2': co_localized_count / total_events_2 if total_events_2 > 0 else 0,
                    'tolerance_window': tolerance,
                }
                
                co_localization_stats[pair_name] = stats
        
        return co_localization_stats

    def _find_co_localized_events(
        self, 
        events1: List[int], 
        events2: List[int], 
        tolerance: int
    ) -> List[Tuple[int, int]]:
        """
        Find events that are co-localized within a tolerance window.
        
        Parameters
        ----------
        events1 : List[int]
            Event indices from first signal
        events2 : List[int]
            Event indices from second signal
        tolerance : int
            Tolerance window in samples
            
        Returns
        -------
        List[Tuple[int, int]]
            List of co-localized event pairs (event1_idx, event2_idx)
        """
        co_localized = []
        
        for event1 in events1:
            for event2 in events2:
                if abs(event1 - event2) <= tolerance:
                    co_localized.append((event1, event2))
        
        return co_localized

    def get_lone_events(
        self,
        signal_name: str,
        events: Dict[str, List[int]],
        tolerance: Optional[int] = None,
    ) -> List[int]:
        """
        Return events for a signal that occur in isolation (no other signal has a peak within tolerance).

        Parameters
        ----------
        signal_name : str
            Name of the signal to extract lone events from
        events : Dict[str, List[int]]
            Dictionary mapping signal names to lists of event indices
        tolerance : int, optional
            Distance criterion for isolation (default: co_localization_tolerance or distance)

        Returns
        -------
        List[int]
            List of event indices in signal_name that are lone events
        """
        if signal_name not in events:
            raise ValueError(f"Signal '{signal_name}' not found in events")
        
        all_other_signals = [k for k in events if k != signal_name]
        lone_events = []
        tol = tolerance if tolerance is not None else (
            self.co_localization_tolerance if self.co_localization_tolerance is not None else self.distance
        )
        for event in events[signal_name]:
            is_lone = True
            for other in all_other_signals:
                for other_event in events[other]:
                    if abs(event - other_event) <= tol:
                        is_lone = False
                        break
                if not is_lone:
                    break
            if is_lone:
                lone_events.append(event)
        return lone_events

    def get_joint_events(
        self,
        signal_names: List[str],
        events: Dict[str, List[int]],
        tolerance: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """
        Return tuples of events that are co-localized (joint) across all specified signals.

        Parameters
        ----------
        signal_names : List[str]
            List of signal names to find joint events for
        events : Dict[str, List[int]]
            Dictionary mapping signal names to lists of event indices
        tolerance : int, optional
            Distance criterion for co-localization (default: co_localization_tolerance or distance)

        Returns
        -------
        List[Tuple[int, ...]]
            List of tuples, each containing one event index from each signal, all within tolerance
        """
        for name in signal_names:
            if name not in events:
                raise ValueError(f"Signal '{name}' not found in events")
        tol = tolerance if tolerance is not None else (
            self.co_localization_tolerance if self.co_localization_tolerance is not None else self.distance
        )
        # Start with all events from the first signal
        base_events = events[signal_names[0]]
        joint_events = []
        for base_event in base_events:
            candidate = [base_event]
            valid = True
            # For each other signal, find a peak within tolerance
            for other_name in signal_names[1:]:
                found = False
                for other_event in events[other_name]:
                    if abs(base_event - other_event) <= tol:
                        candidate.append(other_event)
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                joint_events.append(tuple(candidate))
        return joint_events


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
        
        Parameters
        ----------
        func_data : np.ndarray
            Functional MRI data with shape (timepoints, vertices)
        physio_events : List[int]
            List of event indices around which to calculate windows
            
        Returns
        -------
        WindowAverageResults
            Object containing the windowed average and parameters
        """
        # Input validation
        if func_data.ndim != 2:
            raise ValueError(f"func_data must be 2D array, got shape {func_data.shape}")
        
        if not isinstance(physio_events, list):
            raise ValueError("physio_events must be a list")
        
        # Validate event indices
        if physio_events:
            max_event = max(physio_events)
            min_event = min(physio_events)
            if min_event < 0:
                raise ValueError(f"Event indices must be non-negative, got {min_event}")
            if max_event >= func_data.shape[0]:
                raise ValueError(f"Event index {max_event} exceeds data length {func_data.shape[0]}")
        
        if not physio_events:
            # Handle empty events list
            window_size = self.left_edge + self.right_edge + 1
            w_avg = np.full((window_size, func_data.shape[1]), np.nan)
        else:
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
            'num_events': len(physio_events),
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
    # Input validation
    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {array.shape}")
    
    if not isinstance(center, int):
        raise ValueError(f"center must be an integer, got {type(center)}")
    
    if not isinstance(left_edge, int) or not isinstance(right_edge, int):
        raise ValueError("left_edge and right_edge must be integers")
    
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
    if end > start:  # Only insert if there are valid rows
        padded_range[insert_start:insert_end, :] = array[start:end, :]
    
    return padded_range