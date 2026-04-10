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
from scipy.signal import butter, filtfilt, welch

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 3 
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 3.0        # Corresponds to ~180 BPM
ORDER = 2               # Filter order (2nd order Butterworth is a common choice for rPPG)

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
        
        # 1. Design the filter (e.g. 0.7 Hz to 3.0 Hz covers ~42 to 180 BPM)
        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # Calculate coefficients
        b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps)
        
        # 2. Apply the filter forward and backward (zero phase)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal

    def estimate_heart_rate(self) -> Optional[float]:
        """
        Calculates the Power Spectral Density (PSD) using Welch's method 
        to find the dominant heart rate frequency.
        """
        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None
            
        # Welch's method computes the PSD
        # nperseg is the length of each segment. Using the full length gives max frequency resolution.
        frequencies, psd = welch(filtered_signal, fs=self.target_fps, nperseg=len(filtered_signal))
        
        # Find the index of the highest power peak
        peak_index = np.argmax(psd)
        
        # Get the corresponding frequency in Hz
        dominant_freq = frequencies[peak_index]
        
        # Convert Hz to Beats Per Minute (BPM)
        bpm = dominant_freq * 60.0
        
        return bpm