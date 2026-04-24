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
        self.timestamps = []     # High-res timeline for the DL waveform
        self.gt_signal = []      # The actual BVP/PPG waveform
        
        self.hr_timestamps = []  # Low-res timeline for the HUD
        self.gt_hr = []          # The calculated BPM values

        if not os.path.exists(filepath):
            logging.error(f"Ground truth file not found: {filepath}")
            return

        filename_lower = os.path.basename(filepath).lower()

        # Route the parsing logic based on the file type and sensor type
        if filepath.endswith('.txt'):
            self._parse_ubfc_rppg(filepath)
        elif filepath.endswith('.xmp'):
            self._parse_ubfc_xmp(filepath)
        elif filepath.endswith('.csv'):
            if 'bvp' in filename_lower:
                self._parse_empatica_bvp(filepath)
            elif 'eda' in filename_lower:
                logging.error("CRITICAL: eda.csv measures sweat, not Heart Rate! Point to bvp.csv.")
            else:
                self._parse_empatica_csv(filepath) 
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
        Parses raw Blood Volume Pulse (bvp.csv) from Empatica E4.
        Extracts the exact waveform for DL training and calculates true BPM for HUD.
        """
        try:
            bvp_signal = np.loadtxt(filepath, delimiter=',')
            hz = 64.0  # Hardcoded Empatica E4 sensor frequency
            
            # 1. Clean the raw wrist BVP signal (Bandpass 0.7 - 3.0 Hz)
            b, a = butter(2, [0.7, 3.0], btype='bandpass', fs=hz)
            clean_bvp = filtfilt(b, a, bvp_signal)
            
            # EXPORT THE WAVEFORM FOR DEEP LEARNING (Negative Pearson Loss)
            self.gt_signal = clean_bvp
            self.timestamps = np.arange(len(clean_bvp)) / hz
            
            # 2. Sliding window to calculate BPM (For the live HUD)
            window_sec = 8
            stride_sec = 1
            window_pts = int(window_sec * hz)
            stride_pts = int(stride_sec * hz)
            
            hr_list = []
            hr_times = []
            
            for i in range(0, len(clean_bvp) - window_pts, stride_pts):
                window = clean_bvp[i : i + window_pts]
                peaks, _ = find_peaks(window, distance=hz*0.33) 
                
                if len(peaks) >= 2:
                    ibis = np.diff(peaks) / hz
                    bpm = 60.0 / np.mean(ibis)
                else:
                    bpm = hr_list[-1] if len(hr_list) > 0 else 70.0 
                    
                hr_list.append(bpm)
                # Timestamp for the HUD is the center of the 8-second window
                hr_times.append((i + (window_pts / 2)) / hz)
                
            self.gt_hr = np.array(hr_list)
            self.hr_timestamps = np.array(hr_times)
            
            logging.info(f"Loaded UBFC-Phys BVP: Extracted {len(self.gt_signal)} waveform points at {hz} Hz.")
        except Exception as e:
            logging.error(f"Failed to process BVP file: {e}")

    def _parse_empatica_csv(self, filepath: str):
        """Parses the Empatica E4 HR.csv file (If available)."""
        try:
            data = np.loadtxt(filepath, delimiter=',')
            hz = data[1]
            hr_data = data[2:]
            
            self.hr_timestamps = np.arange(len(hr_data)) / hz
            self.gt_hr = hr_data
            
            # Fallback for DL if no waveform is present (flatline)
            self.timestamps = self.hr_timestamps
            self.gt_signal = hr_data 
            
            logging.info(f"Loaded UBFC-Phys pre-calculated HR: {len(self.hr_timestamps)} readings at {hz} Hz.")
        except Exception as e:
            logging.error(f"Failed to load Empatica .csv file: {e}")
        
    def get_hr_at_time(self, current_time: float) -> float:
        """Finds the closest ground truth Heart Rate for a given video timestamp."""
        
        # FIX: We now search the dedicated hr_timestamps array, not the waveform array!
        time_axis = self.hr_timestamps if len(self.hr_timestamps) > 0 else self.timestamps
        
        if len(time_axis) == 0 or len(self.gt_hr) == 0:
            return None
            
        idx = np.searchsorted(time_axis, current_time)
        
        if idx == 0:
            return float(self.gt_hr[0])
        if idx >= len(time_axis):
            return float(self.gt_hr[-1])
            
        left_time = time_axis[idx - 1]
        right_time = time_axis[idx]
        
        if abs(current_time - left_time) < abs(current_time - right_time):
            return float(self.gt_hr[idx - 1])
        else:
            return float(self.gt_hr[idx])