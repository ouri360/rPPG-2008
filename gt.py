"""
GroundTruthReader Module
---------------------------------------
Parses ground truth data from various datasets (UBFC-rPPG .txt, UBFC-Phys .csv, and UBFC .xmp)
and synchronizes the medical sensor's Heart Rate with the video timeline.
"""

import numpy as np
import logging
import os

class GroundTruthReader:
    def __init__(self, filepath: str):
        self.timestamps = []
        self.gt_hr = []
        self.gt_signal = []

        if not os.path.exists(filepath):
            logging.error(f"Ground truth file not found: {filepath}")
            return

        # ==========================================
        # DSP UPGRADE: Multi-Dataset Auto-Parser
        # Automatically routes the correct parsing logic based on the extension!
        # ==========================================
        if filepath.endswith('.txt'):
            self._parse_ubfc_rppg(filepath)
        elif filepath.endswith('.csv'):
            self._parse_empatica_csv(filepath)
        elif filepath.endswith('.xmp'):
            self._parse_ubfc_xmp(filepath)
        else:
            logging.error(f"Unsupported file format: {filepath}")

    def _parse_ubfc_rppg(self, filepath: str):
        """Parses the 3-row .txt file from the UBFC-rPPG dataset."""
        try:
            data = np.loadtxt(filepath)
            self.gt_signal = data[0, :]
            self.gt_hr = data[1, :]
            self.timestamps = data[2, :]
            logging.info(f"Loaded UBFC-rPPG .txt Ground Truth: {len(self.timestamps)} frames found.")
        except Exception as e:
            logging.error(f"Failed to load UBFC-rPPG .txt file: {e}")

    def _parse_ubfc_xmp(self, filepath: str):
        """Parses the hidden CSV format inside the UBFC .xmp files."""
        try:
            # Data format: Time(ms), HR(bpm), SpO2(%), PPG_Signal
            data = np.loadtxt(filepath, delimiter=',')
            
            # Convert timestamps from milliseconds to seconds
            self.timestamps = data[:, 0] / 1000.0
            self.gt_hr = data[:, 1]
            self.gt_signal = data[:, 3]
            
            logging.info(f"Loaded UBFC .xmp Ground Truth: {len(self.timestamps)} medical sensor readings found.")
        except Exception as e:
            logging.error(f"Failed to load UBFC .xmp file: {e}")

    def _parse_empatica_csv(self, filepath: str):
        """Parses the Empatica E4 HR.csv file from the UBFC-Phys dataset."""
        try:
            data = np.loadtxt(filepath, delimiter=',')
            start_time = data[0]
            hz = data[1]
            hr_data = data[2:]
            
            self.timestamps = np.arange(len(hr_data)) / hz
            self.gt_hr = hr_data
            
            logging.info(f"Loaded UBFC-Phys (Empatica) Ground Truth: {len(self.timestamps)} readings at {hz} Hz.")
        except Exception as e:
            logging.error(f"Failed to load Empatica .csv file: {e}")

    def get_hr_at_time(self, current_time: float) -> float:
        """Finds the closest ground truth Heart Rate for a given video timestamp."""
        if len(self.timestamps) == 0 or len(self.gt_hr) == 0:
            return None
            
        # Blazing-fast binary search
        idx = np.searchsorted(self.timestamps, current_time)
        
        # Handle edges
        if idx == 0:
            return self.gt_hr[0]
        if idx >= len(self.timestamps):
            return self.gt_hr[-1]
            
        # Interpolate closest value
        left_time = self.timestamps[idx - 1]
        right_time = self.timestamps[idx]
        
        if (current_time - left_time) < (right_time - current_time):
            return self.gt_hr[idx - 1]
        else:
            return self.gt_hr[idx]