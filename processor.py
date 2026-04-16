"""
SignalProcessor Module for rPPG Signal Extraction (POS Upgraded)
---------------------------------------
This module handles the extraction, filtering, and buffering of the raw rPPG signal.
It utilizes the state-of-the-art POS (Plane-Orthogonal-to-Skin) algorithm 
to mathematically annihilate ambient lighting flicker and motion artifacts.
"""

import numpy as np
import logging
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.ndimage import uniform_filter1d
import cv2

MINIMUM_AMOUNT_OF_DATA = 4 
LOWCUT_HZ = 0.7         # Lowered to 42 BPM to safely capture resting heart rates
HIGHCUT_HZ = 3.0        # Raised to 180 BPM to capture elevated heart rates
ORDER = 6               # Standardized order for steep frequency cutoffs
NFFT = 8192             

class SignalProcessor:
    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.max_length = int(buffer_seconds * target_fps)
        
        # ==========================================
        # DSP UPGRADE: 3-Channel Architecture
        # POS requires independent tracking of Red, Green, and Blue.
        # ==========================================
        self.raw_r = deque(maxlen=self.max_length)
        self.raw_g = deque(maxlen=self.max_length)
        self.raw_b = deque(maxlen=self.max_length)
        
        # Alias for main.py Graph 1 backwards compatibility (shows Green channel)
        self.raw_signal = self.raw_g 
        
        self.timestamps = deque(maxlen=self.max_length)

        smoothing_frames = int(self.target_fps * 1)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor upgraded to POS RGB Architecture.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> float:
        weights = {
            'forehead': 0.60,
            'left_cheek': 0.20,
            'right_cheek': 0.20
        }
        
        weighted_r = 0.0
        weighted_g = 0.0
        weighted_b = 0.0
        
        # Extract all three color channels (OpenCV uses BGR order)
        b_channel, g_channel, r_channel = cv2.split(frame)
        
        for region_name, polygon in rois.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            r_pixels = r_channel[mask == 255]
            g_pixels = g_channel[mask == 255]
            b_pixels = b_channel[mask == 255]
            
            if len(g_pixels) == 0:
                continue
                
            # ==========================================
            # DSP UPGRADE: Synchronized 3-Channel Sorting
            # We use the Green channel's luminance to find the glare/shadows,
            # and apply those exact same drop indices to Red and Blue!
            # ==========================================
            sort_indices = np.argsort(g_pixels)
            
            bottom_trim = int(len(sort_indices) * 0.05)
            top_trim = int(len(sort_indices) * 0.20)
            
            if bottom_trim > 0 and top_trim > 0:
                valid_indices = sort_indices[bottom_trim:-top_trim]
            else:
                valid_indices = sort_indices
                
            r_val = float(np.mean(r_pixels[valid_indices]))
            g_val = float(np.mean(g_pixels[valid_indices]))
            b_val = float(np.mean(b_pixels[valid_indices]))
            # ==========================================
            
            weighted_r += r_val * weights[region_name]
            weighted_g += g_val * weights[region_name]
            weighted_b += b_val * weights[region_name]

        # Push to the RGB buffers
        self.raw_r.append(weighted_r)
        self.raw_g.append(weighted_g)
        self.raw_b.append(weighted_b)
        
        self.timestamps.append(timestamp)

        return weighted_g

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.raw_g), np.array(self.timestamps)
    
    def get_filtered_signal(self) -> Optional[np.ndarray]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        r = np.array(list(self.raw_r))
        g = np.array(list(self.raw_g))
        b = np.array(list(self.raw_b))
        ts = np.array(list(self.timestamps))
        
        if len(ts) < 2:
            return None

        # 1. Uniform Resampling for all channels
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        r_u = np.interp(t_uniform, ts, r)
        g_u = np.interp(t_uniform, ts, g)
        b_u = np.interp(t_uniform, ts, b)
        
        # ==========================================
        # DSP UPGRADE 1: Pre-POS Anti-Movement Clamping
        # We crush the physical movement spikes BEFORE they enter the POS math!
        # ==========================================
        r_u = self.remove_impulse_noise(r_u)
        g_u = self.remove_impulse_noise(g_u)
        b_u = self.remove_impulse_noise(b_u)
        
        # ==========================================
        # DSP UPGRADE 2: 2.0-Second POS Window
        # ==========================================
        window_size = int(self.target_fps * 2.0)
        
        # 1. Rolling Temporal Normalization
        r_mean = uniform_filter1d(r_u, size=window_size)
        g_mean = uniform_filter1d(g_u, size=window_size)
        b_mean = uniform_filter1d(b_u, size=window_size)
        
        r_n = r_u / (r_mean + 1e-8)
        g_n = g_u / (g_mean + 1e-8)
        b_n = b_u / (b_mean + 1e-8)

        # 2. POS Orthogonal Projections
        s1 = g_n - b_n
        s2 = -2.0 * r_n + g_n + b_n

        # 3. Dynamic Rolling Alpha
        def rolling_std(x, w):
            mean_x = uniform_filter1d(x, size=w)
            mean_x2 = uniform_filter1d(x**2, size=w)
            var = mean_x2 - mean_x**2
            var = np.maximum(var, 0) # Prevent negative floating point roots
            return np.sqrt(var)
            
        std_s1 = rolling_std(s1, window_size)
        std_s2 = rolling_std(s2, window_size)
        
        alpha = std_s1 / (std_s2 + 1e-8)
        
        # 4. Final POS Pulse Wave
        pos_signal = s1 + alpha * s2
        
        # Detrend and Normalize (Removes the gentle slope left by the clamp)
        pos_signal = self.detrend_and_normalize(pos_signal)

        # Apply High-Precision Butterworth Bandpass
        sos = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, pos_signal)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None

        filtered_signal = self.get_filtered_signal()
        if filtered_signal is None:
            return None, None, None
            
        n_fft = NFFT
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=n_fft)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)

        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 3.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_filt_mag = filtered_power[plot_indices]
        
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        peak_index = np.argmax(bpm_filt_mag)
        dominant_freq = bpm_freqs[peak_index]
        
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: 
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df

        raw_bpm = dominant_freq * 60.0
        
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        return smoothed_bpm, plot_freqs, plot_filt_mag
    
    def get_current_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        recent_ts = np.array(self.timestamps)[-30:]
        if len(recent_ts) < 2:
            return 0.0
        mean_diff = float(np.mean(np.diff(recent_ts)))
        if mean_diff <= 0.0:
            return self.target_fps 
        return 1.0 / mean_diff
    
    def detrend_and_normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float64)
        
        # Remove low-frequency baseline drift
        window_size = int(self.target_fps * 1.5)
        baseline = uniform_filter1d(x, size=window_size)
        x = x - baseline
        
        std = np.std(x)
        if std == 0:
            return x
        return (x - np.mean(x)) / (std + 1e-8)
    
    @staticmethod
    def remove_impulse_noise(x: np.ndarray) -> np.ndarray:
        """
        Uses Robust Percentile Velocity Clamping to decapitate sudden movement spikes 
        while perfectly preserving the gentle pixel speed of a normal human heartbeat.
        """
        if len(x) < 2:
            return x
            
        diffs = np.diff(x)
        abs_diffs = np.abs(diffs)
        
        # ==========================================
        # DSP UPGRADE: The Percentile Shield
        # 85% of the time, the subject is relatively still (only blood is pulsing).
        # We find this normal speed and enforce it relentlessly.
        # ==========================================
        normal_speed = float(np.percentile(abs_diffs, 85))
        
        # Strict clamp: No pixel is allowed to change faster than 1.5x the normal heartbeat speed
        max_velocity = normal_speed * 1.5
        
        # Epsilon fallback in case the subject is perfectly frozen like a statue
        if max_velocity == 0:
            max_velocity = 1e-5
            
        clamped_diffs = np.clip(diffs, -max_velocity, max_velocity)
        reconstructed = np.concatenate(([x[0]], x[0] + np.cumsum(clamped_diffs)))
        
        return reconstructed