"""
GroundTruthReader Module
---------------------------------------
Parses ground truth data from various datasets (UBFC-rPPG, UBFC-Phys)
and synchronizes the medical sensor's Heart Rate with the video timeline.
"""

import numpy as np
import logging
import os
from scipy.signal import butter, filtfilt, find_peaks

class GroundTruthReader:
    def __init__(self, filepath: str):
        self.timestamps = []
        self.gt_hr = []
        self.gt_signal = []

        if not os.path.exists(filepath):
            logging.error(f"Ground truth file not found: {filepath}")
            return

        filename_lower = os.path.basename(filepath).lower()

        # ==========================================
        # DSP UPGRADE: Multi-Dataset Auto-Parser
        # Routes the parsing logic based on the file type and sensor type!
        # ==========================================
        if filepath.endswith('.txt'):
            self._parse_ubfc_rppg(filepath)
        elif filepath.endswith('.xmp'):
            self._parse_ubfc_xmp(filepath)
        elif filepath.endswith('.csv'):
            if 'bvp' in filename_lower:
                self._parse_empatica_bvp(filepath)
            elif 'eda' in filename_lower:
                logging.error("CRITICAL: eda.csv measures sweat (Electrodermal Activity), not Heart Rate! Please point to bvp.csv.")
            else:
                self._parse_empatica_csv(filepath) # Fallback if HR.csv exists
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
            data = np.loadtxt(filepath, delimiter=',')
            self.timestamps = data[:, 0] / 1000.0
            self.gt_hr = data[:, 1]
            self.gt_signal = data[:, 3]
            logging.info(f"Loaded UBFC .xmp Ground Truth: {len(self.timestamps)} medical sensor readings found.")
        except Exception as e:
            logging.error(f"Failed to load UBFC .xmp file: {e}")

    def _parse_empatica_bvp(self, filepath: str):
        """
        Parses raw Blood Volume Pulse (bvp.csv) from Empatica E4 
        and dynamically calculates the true BPM using Peak Detection.
        """
        try:
            # Empatica BVP format: Row 0 = Start Time, Row 1 = Hz, Row 2+ = Raw PPG Signal
            data = np.loadtxt(filepath, delimiter=',')
            hz = data[1]
            bvp_signal = data[2:]
            
            # 1. Clean the raw wrist BVP signal (Bandpass 0.7 - 3.0 Hz)
            b, a = butter(2, [0.7, 3.0], btype='bandpass', fs=hz)
            clean_bvp = filtfilt(b, a, bvp_signal)
            
            # 2. Sliding window to calculate BPM (8-second window, 1-second stride)
            window_sec = 8
            stride_sec = 1
            window_pts = int(window_sec * hz)
            stride_pts = int(stride_sec * hz)
            
            hr_list = []
            ts_list = []
            
            for i in range(0, len(clean_bvp) - window_pts, stride_pts):
                window = clean_bvp[i : i + window_pts]
                
                # Minimum distance between peaks: 0.33 seconds (max 180 BPM)
                peaks, _ = find_peaks(window, distance=hz*0.33) 
                
                if len(peaks) >= 2:
                    ibis = np.diff(peaks) / hz # Calculate Inter-Beat Intervals
                    bpm = 60.0 / np.mean(ibis)
                else:
                    bpm = hr_list[-1] if len(hr_list) > 0 else 70.0 # Fallback
                    
                hr_list.append(bpm)
                # Assign the calculated HR to the exact middle of the window
                ts_list.append((i + (window_pts / 2)) / hz)
                
            self.timestamps = np.array(ts_list)
            self.gt_hr = np.array(hr_list)
            
            logging.info(f"Loaded UBFC-Phys BVP: Dynamically calculated {len(self.gt_hr)} True HR points from raw wrist optical data at {hz} Hz.")
        except Exception as e:
            logging.error(f"Failed to process BVP file: {e}")

    def _parse_empatica_csv(self, filepath: str):
        """Parses the Empatica E4 HR.csv file (If available)."""
        try:
            data = np.loadtxt(filepath, delimiter=',')
            hz = data[1]
            hr_data = data[2:]
            self.timestamps = np.arange(len(hr_data)) / hz
            self.gt_hr = hr_data
            logging.info(f"Loaded UBFC-Phys pre-calculated HR: {len(self.timestamps)} readings at {hz} Hz.")
        except Exception as e:
            logging.error(f"Failed to load Empatica .csv file: {e}")

    def get_hr_at_time(self, current_time: float) -> float:
        """Finds the closest ground truth Heart Rate for a given video timestamp."""
        if len(self.timestamps) == 0 or len(self.gt_hr) == 0:
            return None
            
        idx = np.searchsorted(self.timestamps, current_time)
        
        if idx == 0:
            return self.gt_hr[0]
        if idx >= len(self.timestamps):
            return self.gt_hr[-1]
            
        left_time = self.timestamps[idx - 1]
        right_time = self.timestamps[idx]
        
        if (current_time - left_time) < (right_time - current_time):
            return self.gt_hr[idx - 1]
        else:
            return self.gt_hr[idx]