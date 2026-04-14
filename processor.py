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
from scipy.signal import butter, sosfiltfilt, detrend

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 3 
BUFFER_AVERAGE = 30     # Number of BPM estimates to average for smoothing
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.9         # Corresponds to ~54 BPM
HIGHCUT_HZ = 2          # Corresponds to ~120 BPM
ORDER = 4               # Filter order
NFFT = 8192             # Number of points for FFT (zero-padding for better frequency resolution)

class SignalProcessor:
    """
    Handles the extraction, filtering, buffering, and frequency analysis of the rPPG signal.
    """

    def __init__(self, buffer_seconds: int = 10, target_fps: float = 30.0):
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

        # Buffer for smoothing BPM estimates over time (average over the last 30 estimates)
        smoothing_frames = int(self.target_fps * 5)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 

        # Tracker for Outlier Rejection
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> float:
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
        self.timestamps.append(timestamp)

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
        ts = np.array(self.timestamps)
        
        if len(ts) < 2:
            return None

        # ==========================================
        # UPGRADE 1: Uniform Resampling
        # Fixes camera stutter by creating a perfect time grid
        # ==========================================
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        signal_uniform = np.interp(t_uniform, ts, signal)
        
        # Detrend the newly uniform signal
        signal_uniform = detrend(signal_uniform)
        
        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # Use our SOS high-precision filter instead of b, a
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, signal_uniform)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None, None

        # 1. Resample and detrend the raw signal to match the uniform filtered signal
        ts = np.array(self.timestamps)
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        raw_uniform = np.interp(t_uniform, ts, np.array(self.raw_signal))
        raw_detrended_signal = detrend(raw_uniform)

        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None, None
            
        n_fft = NFFT
        
        # 2. Apply the Hanning Window
        window = np.hanning(len(filtered_signal))
        windowed_raw = raw_detrended_signal * window
        windowed_filt = filtered_signal * window
        
        # 3. Compute the Power Spectra (Magnitude Squared)
        fft_raw_complex = np.fft.rfft(windowed_raw, n=n_fft)
        raw_power = (np.abs(fft_raw_complex)**2) 
        
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=n_fft)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)

        # Matplotlib Dashboard Mask (0.0 to 3.0 Hz)
        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 3.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_raw_mag = raw_power[plot_indices]
        plot_filt_mag = filtered_power[plot_indices]
        
        # BPM Calculation Mask
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        # 4. Find the Peak
        peak_index = np.argmax(bpm_filt_mag)
        dominant_freq = bpm_freqs[peak_index]
        
        # ==========================================
        # UPGRADE 2: Parabolic Sub-bin Interpolation
        # Calculates the true vertex between frequency bins
        # ==========================================
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: # Avoid division by zero
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df
        # ==========================================

        raw_bpm = dominant_freq * 60.0
        
        # ==========================================
        # FIX 2: Medical Outlier Rejection
        # A human heart cannot jump by 10+ BPM instantly. 
        # If it does, we know it's a traveling Hanning artifact.
        # ==========================================
        if self.last_valid_bpm is not None:
            # If the jump is larger than 10 BPM in a fraction of a second...
            if abs(raw_bpm - self.last_valid_bpm) > 10.0:
                # Reject the noise! Feed the last known stable BPM into the buffer instead.
                raw_bpm = self.last_valid_bpm
                
        # Update the tracker with the verified BPM
        self.last_valid_bpm = raw_bpm
        # ==========================================

        # Add to our new, longer 5-second smoothing buffer
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        return smoothed_bpm, plot_freqs, plot_raw_mag, plot_filt_mag
    
    def get_current_fps(self) -> float:
        """Returns actual measured FPS from the last ~1 second of timestamps."""
        if len(self.timestamps) < 2:
            return 0.0
        # Use last 30 samples (≈1 second) for a stable reading
        recent_ts = np.array(self.timestamps)[-30:]
        if len(recent_ts) < 2:
            return 0.0
        return 1.0 / np.mean(np.diff(recent_ts))