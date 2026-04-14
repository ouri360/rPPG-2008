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
from scipy.signal import butter, filtfilt, detrend

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 3 
BUFFER_AVERAGE = 15     # Number of BPM estimates to average for smoothing
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 2.5        # Corresponds to ~150 BPM
ORDER = 4               # Filter order (2nd order Butterworth is a common choice for rPPG)
NFFT = 8192             # Number of points for FFT (zero-padding for better frequency resolution)

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
        self.max_length = int(buffer_seconds * target_fps)
        
        # deques automatically pop the oldest item when maxlen is reached
        self.raw_signal = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 15 estimates)
        self.bpm_buffer = deque(maxlen=BUFFER_AVERAGE) 
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

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
        Calculates the Standard FFT and returns the BPM along with frequency data for plotting.
        Returns: (smoothed_bpm, frequencies_array, magnitude_array)
        """
        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None
            
        n_fft = NFFT # Zero-padding to maintain decimal precision for BPM
        
        # ==========================================
        # DSP UPGRADE: Apply a Hanning Window to stop Spectral Leakage
        # ==========================================
        window = np.hanning(len(filtered_signal))
        windowed_signal = filtered_signal * window
        
        # 1. Compute the Standard 1D Discrete Fourier Transform
        fft_complex = np.fft.rfft(windowed_signal, n=n_fft)
        # ==========================================
        
        # 2. Extract the magnitude (absolute value) from the complex numbers
        fft_magnitude = np.abs(fft_complex)
        
        # 3. Generate the corresponding frequency bins
        # d is the sample spacing (inverse of framerate)
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)
        
        # 4. Mask the arrays to strictly look at human heart rates (0.7 to 3.0 Hz)
        valid_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        
        valid_frequencies = frequencies[valid_indices]
        valid_magnitude = fft_magnitude[valid_indices]
        
        # 5. Find the peak frequency
        peak_index = np.argmax(valid_magnitude)
        dominant_freq = valid_frequencies[peak_index]
        
        raw_bpm = dominant_freq * 60.0
        
        # Apply the rolling average smoothing
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        return smoothed_bpm, valid_frequencies, valid_magnitude