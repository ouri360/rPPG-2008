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
import logging
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, sosfiltfilt, detrend, medfilt

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 3 
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

        # Buffer for smoothing BPM estimates over time (average over the last 5 seconds)
        smoothing_frames = int(self.target_fps * 1)
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
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        signal = np.array(self.raw_signal)
        ts = np.array(self.timestamps)
        
        if len(ts) < 2:
            return None

        # 1. Uniform Resampling
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        signal_uniform = np.interp(t_uniform, ts, signal)
        
        # 2. Detrend the newly uniform signal
        signal_uniform = detrend(signal_uniform)
        
        # ==========================================
        # THE TRUE FIX: Z-Score Artifact Clamping
        # Clamps massive movement spikes without distorting the pulse wave
        # ==========================================
        std_val = np.std(signal_uniform)
        if std_val > 0:
            # 3.0 Standard Deviations leaves the heartbeat untouched, but kills blinks.
            signal_uniform = np.clip(signal_uniform, -3.0 * std_val, 3.0 * std_val)
        # ==========================================

        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # 3. Apply the High-Precision Butterworth
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, signal_uniform)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None, None

        # 1. Resample and detrend the raw signal
        ts = np.array(self.timestamps)
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        raw_uniform = np.interp(t_uniform, ts, np.array(self.raw_signal))
        raw_detrended_signal = detrend(raw_uniform)

        # Apply the exact same Median Filter here so the Raw FFT graph matches reality
        kernel_size = int(self.target_fps * 0.25)
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        raw_detrended_signal = medfilt(raw_detrended_signal, kernel_size)

        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None, None
            
        n_fft = NFFT
        
        # 2. Apply the Hanning Window
        window = np.hanning(len(filtered_signal))
        windowed_raw = raw_detrended_signal * window
        windowed_filt = filtered_signal * window
        
        # 3. Compute the Power Spectra
        fft_raw_complex = np.fft.rfft(windowed_raw, n=n_fft)
        raw_power = (np.abs(fft_raw_complex)**2) 
        
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=n_fft)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)

        # Dashbaord arrays
        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 3.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_raw_mag = raw_power[plot_indices]
        plot_filt_mag = filtered_power[plot_indices]
        
        # BPM Arrays
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        # 4. Find the Peak
        peak_index = np.argmax(bpm_filt_mag)
        dominant_freq = bpm_freqs[peak_index]
        
        # Parabolic Sub-bin Interpolation
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: 
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df

        raw_bpm = dominant_freq * 60.0
        
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        return smoothed_bpm, plot_freqs, plot_raw_mag, plot_filt_mag
    
    def get_current_fps(self) -> float:
        """Returns actual measured FPS from the last ~1 second of timestamps."""
        if len(self.timestamps) < 2:
            return 0.0
            
        recent_ts = np.array(self.timestamps)[-30:]
        if len(recent_ts) < 2:
            return 0.0
            
        # Protect against division by zero if timestamps ever get completely stuck
        mean_diff = float(np.mean(np.diff(recent_ts)))
        if mean_diff <= 0.0:
            return self.target_fps # Fallback to the target FPS safely
            
        return 1.0 / mean_diff