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
LOWCUT_HZ = 0.9         # Corresponds to ~54 BPM
HIGHCUT_HZ = 2          # Corresponds to ~120 BPM
ORDER = 4               # Filter order
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
            break # ONLY THE FOREHEAD

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
        
        # 1. Design the filter
        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # Calculate coefficients
        b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps)
        
        # 2. Apply the filter forward and backward (zero phase)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None, None

        # 1. Get both signals
        raw_detrended_signal = detrend(np.array(self.raw_signal))
        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None, None
            
        n_fft = NFFT
        
        # 2. Apply the Hanning Window
        window = np.hanning(len(filtered_signal))
        windowed_raw = raw_detrended_signal * window
        windowed_filt = filtered_signal * window
        
        # 3. Compute the FFTs
        fft_raw_complex = np.fft.rfft(windowed_raw, n=n_fft)
        raw_magnitude = np.abs(fft_raw_complex)
        
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=n_fft)
        filtered_magnitude = np.abs(fft_filtered_complex)
        
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)

        # ==========================================
        # VISUALIZATION MASK (0.0 to 3.0 Hz)
        # We send this wide view to the Matplotlib dashboard
        # ==========================================
        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 3.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_raw_mag = raw_magnitude[plot_indices]
        plot_filt_mag = filtered_magnitude[plot_indices]
        
        # ==========================================
        # BPM CALCULATION MASK (0.7 to 2.5 Hz)
        # We strictly calculate the Heart Rate in the human passband
        # ==========================================
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_magnitude[bpm_indices]
        
        # 4. Calculate BPM using the strictly masked array
        peak_index = np.argmax(bpm_filt_mag)
        dominant_freq = bpm_freqs[peak_index]
        
        raw_bpm = dominant_freq * 60.0
        
        # Apply the rolling average smoothing
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        # Return the WIDE arrays for the dashboard, and the smoothed BPM
        return smoothed_bpm, plot_freqs, plot_raw_mag, plot_filt_mag