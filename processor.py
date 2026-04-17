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
from scipy.signal import butter, sosfiltfilt, detrend
import cv2

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 4 
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 3.0          # Corresponds to ~180 BPM
ORDER = 3               # Filter order (x2 with sosfiltfilt for zero phase distortion)
NFFT = 8192             # Number of points for FFT (zero-padding for better frequency resolution)

class SignalProcessor:
    """
    Handles the extraction, filtering, buffering, and frequency analysis of the rPPG signal.
    """

    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        """
        Initializes rolling buffers for the signal and timestamps.
        
        Args:
            buffer_seconds (int): How many seconds of data to hold in memory.
            target_fps (int): Expected framerate of the camera.
        """
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        # Calculate maximum buffer size
        self.max_length = int(buffer_seconds * target_fps)
        
        # deques automatically pop the oldest item when maxlen is reached
        self.raw_g = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 1 seconds)
        smoothing_frames = int(self.target_fps * 1)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 

        # Tracker for Outlier Rejection
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        # Area-Weighted Super Mask
        master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for region_name, polygon in rois.items():
            cv2.fillPoly(master_mask, [polygon], 255)
            
        g_channel = frame[:, :, 1]
        g_pixels = g_channel[master_mask == 255]
        
        if len(g_pixels) > 0:
            self.raw_g.append(float(np.mean(g_pixels)))
        else:
            self.raw_g.append(0.0)
            
        self.timestamps.append(timestamp)

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        g = np.array(list(self.raw_g))
        ts = np.array(list(self.timestamps))
        
        # 1. Strict Physical Time Pruning
        time_limit = ts[-1] - self.buffer_seconds
        valid_indices = np.where(ts >= time_limit)[0]
        
        g = g[valid_indices]
        ts = ts[valid_indices]
        
        if len(ts) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        # 2. Temporal Resampling
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        g_u = np.interp(t_uniform, ts, g)
        
        # 3. Detrending and Variance Standardization
        g_norm = self.detrend_and_normalize(g_u)
        
        # 4. Butterworth Bandpass
        sos = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, g_norm)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        filtered_signal = self.get_filtered_signal()
        if filtered_signal is None:
            return None, None, None
            
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        fft_complex = np.fft.rfft(windowed_filt, n=NFFT)
        power = (np.abs(fft_complex)**2)
        freqs = np.fft.rfftfreq(NFFT, d=1.0/self.target_fps)
        
        bpm_indices = np.where((freqs >= LOWCUT_HZ) & (freqs <= HIGHCUT_HZ))[0]
        peak_index = np.argmax(power[bpm_indices])
        dominant_freq = freqs[bpm_indices][peak_index]

        raw_bpm = dominant_freq * 60.0
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        plot_idx = np.where((freqs >= 0.0) & (freqs <= 3.0))[0]
        return smoothed_bpm, freqs[plot_idx], power[plot_idx]
    
    def get_current_fps(self) -> float:
        if len(self.timestamps) < 2: return 0.0
        recent_ts = np.array(self.timestamps)[-30:]
        mean_diff = float(np.mean(np.diff(recent_ts))) if len(recent_ts) >= 2 else 0.0
        return 1.0 / mean_diff if mean_diff > 0.0 else self.target_fps
    
    @staticmethod
    def detrend_and_normalize(x: np.ndarray) -> np.ndarray:
        x = detrend(np.array(x, dtype=np.float64))
        std = np.std(x)
        return (x - np.mean(x)) / (std + 1e-8) if std > 0 else x