"""
SignalProcessor Module for rPPG Signal Extraction
---------------------------------------
This module handles the extraction, filtering, and buffering of the raw rPPG signal from video frames.
It crops the detected face region, isolates the green channel, calculates the spatial average,
and maintains a rolling buffer of the signal values and their corresponding timestamps for real-time processing.
It also includes methods for applying a bandpass filter to the raw signal, 
and estimating the heart rate using frequency analysis.
"""

import numpy as np
import time
import logging
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, filtfilt, welch, detrend

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 3 
BUFFER_AVERAGE = 15     # Number of BPM estimates to average for smoothing
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 3.0        # Corresponds to ~180 BPM
ORDER = 2               # Filter order (2nd order Butterworth is a common choice for rPPG)
NFFT = 2048             # Number of points for FFT (zero-padding for better frequency resolution)

class SignalProcessor:
    """
    Handles the extraction, filtering, buffering, and frequency analysis of the rPPG signal.
    """

    def __init__(self, buffer_seconds: int = 10, target_fps: int = 30):
        """
        Initializes rolling buffers for the signal and timestamps.
        
        Args:
            buffer_seconds (int): How many seconds of data to hold in memory.
            target_fps (int): Expected framerate of the camera.
        """
        self.target_fps = target_fps
        # Calculate maximum buffer size
        self.max_length = buffer_seconds * target_fps
        
        # deques automatically pop the oldest item when maxlen is reached
        self.raw_signal = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 15 estimates)
        self.bpm_buffer = deque(maxlen=BUFFER_AVERAGE) 
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
        """
        Crops the ROI, isolates the green channel, calculates the spatial average, 
        and appends it to the time-series buffer.

        Args:
            frame (np.ndarray): The full BGR frame.
            roi (Tuple[int, int, int, int]): The (x, y, w, h) bounding box.

        Returns:
            float: The spatial average of the green channel for this frame.
        """
        x, y, w, h = roi
        
        # 1. NumPy Slicing: Extract the region of interest
        # Array shape is (rows, cols, channels), which translates to (y, x, c)
        cropped_roi = frame[y:y+h, x:x+w]

        # 2. Channel Isolation: Isolate the Green channel (Index 1 in BGR)
        # The syntax [:, :, 1] means "take all rows, all cols, but only channel 1 (green)"
        green_channel = cropped_roi[:, :, 1]

        # 3. Spatial Averaging: Calculate the mean pixel value
        spatial_average = float(np.mean(green_channel))

        # 4. Buffering: Store the value and the exact time it was captured
        self.raw_signal.append(spatial_average)
        self.timestamps.append(time.time())

        return spatial_average

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict) -> float:
        """
        Calculates a weighted spatial average from multiple ROIs.
        """
        # Define the weights (must sum to 1.0)
        weights = {
            'forehead': 0.60,
            'left_cheek': 0.20,
            'right_cheek': 0.20
        }
        
        weighted_sum = 0.0
        
        for region_name, box in rois.items():
            x, y, w, h = box
            
            # Extract ROI and isolate Green channel
            cropped_roi = frame[y:y+h, x:x+w]
            green_channel = cropped_roi[:, :, 1]
            
            # Calculate mean and apply the specific weight for this region
            region_mean = float(np.mean(green_channel))
            weighted_sum += region_mean * weights[region_name]

        # Buffer the final weighted value
        self.raw_signal.append(weighted_sum)
        self.timestamps.append(time.time())

        return weighted_sum

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the current buffers as NumPy arrays for filtering/FFT.
        """
        return np.array(self.raw_signal), np.array(self.timestamps)
    
    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """
        Applies a zero-phase Butterworth bandpass filter to the raw signal.
        """
        # We need a minimum amount of data to filter properly (e.g. 3 seconds)
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        signal = np.array(self.raw_signal)

        # Remove linear baseline drift before filtering (useful for changes in lighting or subject movement)
        signal = detrend(signal)
        
        # 1. Design the filter (e.g. 0.7 Hz to 3.0 Hz covers ~42 to 180 BPM)
        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # Calculate coefficients
        b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps)
        
        # 2. Apply the filter forward and backward (zero phase)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculates the PSD and returns the BPM along with the frequency data for plotting.
        Returns: (smoothed_bpm, frequencies_array, psd_array)
        """
        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None
            
        # Zero-padding with nfft for better frequency resolution in the FFT.
        n_fft = NFFT
        
        # Welch's method computes the PSD
        # nperseg is the length of each segment. Using the full length gives max frequency resolution.
        frequencies, psd = welch(
            filtered_signal, 
            fs=self.target_fps, 
            nperseg=len(filtered_signal), 
            nfft=n_fft
        )

        # We create a mask to focus only on the frequencies that correspond to plausible heart rates (0.7 Hz to 3.0 Hz)
        valid_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        
        valid_frequencies = frequencies[valid_indices]
        valid_psd = psd[valid_indices]
        
        # Find the index of the highest power peak
        peak_index = np.argmax(valid_psd)
        # Get the corresponding frequency in Hz
        dominant_freq = valid_frequencies[peak_index]
        
        # Convert Hz to Beats Per Minute (BPM)
        raw_bpm = dominant_freq * 60.0
        
        # Simple moving average to smooth the BPM estimates over time
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        # We return the frequency and PSD arrays alongside the BPM
        return smoothed_bpm, valid_frequencies, valid_psd