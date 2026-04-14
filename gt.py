"""
GroundTruthReader Module
---------------------------------------
Parses ground truth data from the UBFC-rPPG dataset and synchronizes 
the medical sensor's Heart Rate with the current video frame timestamp.
"""

import numpy as np
import logging

class GroundTruthReader:
    def __init__(self, filepath: str):
        """
        Loads the UBFC ground truth file.
        The file must contain 3 rows: Signal, HR (BPM), and Timestamps.
        """
        try:
            # np.loadtxt automatically handles the spaces and builds a 2D array
            data = np.loadtxt(filepath)
            
            # UBFC-rPPG Standard Format
            self.gt_signal = data[0, :]
            self.gt_hr = data[1, :]
            self.timestamps = data[2, :]
            
            logging.info(f"Loaded Ground Truth data: {len(self.timestamps)} frames found.")
            
        except Exception as e:
            logging.error(f"Failed to load Ground Truth file: {e}")
            self.timestamps = []
            self.gt_hr = []

    def get_hr_at_time(self, current_time: float) -> float:
        """
        Finds the closest ground truth Heart Rate for a given video timestamp.
        """
        if len(self.timestamps) == 0:
            return None
            
        # np.searchsorted is a blazing-fast binary search to find the closest timestamp
        idx = np.searchsorted(self.timestamps, current_time)
        
        # Handle edge cases (beginning or end of video)
        if idx == 0:
            return self.gt_hr[0]
        if idx == len(self.timestamps):
            return self.gt_hr[-1]
            
        # Check which timestamp is closer: the one before or the one after
        left_time = self.timestamps[idx - 1]
        right_time = self.timestamps[idx]
        
        if (current_time - left_time) < (right_time - current_time):
            return self.gt_hr[idx - 1]
        else:
            return self.gt_hr[idx]