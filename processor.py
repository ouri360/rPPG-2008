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
import cv2
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, sosfiltfilt, detrend

MINIMUM_AMOUNT_OF_DATA = 4 
LOWCUT_HZ = 0.7         
HIGHCUT_HZ = 3.0        
ORDER = 2                
NFFT = 8192             

class SignalProcessor:
    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        self.max_length = int(buffer_seconds * target_fps)
        
        self.raw_g = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)
        self.bpm_buffer = deque(maxlen=int(self.target_fps * 0.5)) 

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for region_name, polygon in rois.items():
            cv2.fillPoly(master_mask, [polygon], 255)
            
        g_channel = frame[:, :, 1]
        skin_pixels = g_channel[master_mask == 255]
        
        if len(skin_pixels) == 0:
            self.raw_g.append(0.0)
            self.timestamps.append(timestamp)
            return
            
        # ==========================================
        # Asymmetric Pixel Sorter
        # Mandatory for single-channel methods to prevent 
        # specular glare from destroying the 1D mean.
        # ==========================================
        sorted_pixels = np.sort(skin_pixels)
        bottom_trim = int(len(sorted_pixels) * 0.05)
        top_trim = int(len(sorted_pixels) * 0.20)
        
        if bottom_trim > 0 and top_trim > 0:
            pure_skin = sorted_pixels[bottom_trim:-top_trim]
        else:
            pure_skin = sorted_pixels
            
        self.raw_g.append(float(np.mean(pure_skin)))
        self.timestamps.append(timestamp)

    def get_filtered_signal(self) -> Optional[np.ndarray]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        g = np.array(list(self.raw_g))
        ts = np.array(list(self.timestamps))
        
        time_limit = ts[-1] - self.buffer_seconds
        valid_indices = np.where(ts >= time_limit)[0]
        
        g = g[valid_indices]
        ts = ts[valid_indices]
        
        if len(ts) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        g_u = np.interp(t_uniform, ts, g)
        
        # ==========================================
        # Impulse Noise Removal
        # Acts as a shock absorber for the Butterworth filter.
        # ==========================================
        g_u = self.remove_impulse_noise(g_u)
        
        g_norm = self.detrend_and_normalize(g_u)
        
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
        if len(self.timestamps) < 2: 
            return 0.0
        recent_ts = np.array(self.timestamps)[-30:]
        mean_diff = float(np.mean(np.diff(recent_ts))) if len(recent_ts) >= 2 else 0.0
        return 1.0 / mean_diff if mean_diff > 0.0 else self.target_fps
    
    @staticmethod
    def detrend_and_normalize(x: np.ndarray) -> np.ndarray:
        x = detrend(np.array(x, dtype=np.float64))
        std = np.std(x)
        return (x - np.mean(x)) / (std + 1e-8) if std > 0 else x

    @staticmethod
    def remove_impulse_noise(x: np.ndarray) -> np.ndarray:
        if len(x) < 2:
            return x
        diffs = np.diff(x)
        std_diff = np.std(diffs)
        if std_diff > 0:
            clamped_diffs = np.clip(diffs, -3.0 * std_diff, 3.0 * std_diff)
            return np.concatenate(([x[0]], x[0] + np.cumsum(clamped_diffs)))
        return x