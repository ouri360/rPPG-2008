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
from scipy.signal import butter, detrend, sosfiltfilt
import cv2

MINIMUM_AMOUNT_OF_DATA = 2 
LOWCUT_HZ = 0.7         # Lowered to 42 BPM to safely capture resting heart rates
HIGHCUT_HZ = 3.0        # Raised to 180 BPM to capture elevated heart rates
ORDER = 3               # Standardized order for steep frequency cutoffs
NFFT = 8192             

class SignalProcessor:
    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        self.max_length = int(buffer_seconds * target_fps)
        
        # ==========================================
        # 3-Channel Architecture
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
        
        logging.info("SignalProcessor using to POS RGB Architecture.")

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
                
            r_val = float(np.mean(r_pixels))
            g_val = float(np.mean(g_pixels))
            b_val = float(np.mean(b_pixels))
            
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
        
        # Timestamps array
        ts = np.array(list(self.timestamps))

        # ==========================================
        # 1. Temporal Resampling
        # POS strictly requires uniform matrices. We interpolate 
        # the erratic physical frames into a flawless 30Hz grid.
        # ==========================================
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        r_u = np.interp(t_uniform, ts, r)
        g_u = np.interp(t_uniform, ts, g)
        b_u = np.interp(t_uniform, ts, b)
        
        C = np.vstack([r_u, g_u, b_u])
        N = C.shape[1]
        
        L = int(self.target_fps * 1.6) 
        H = np.zeros(N)
        
        # ==========================================
        # 2. Strict POS Overlap-Add Loop
        # ==========================================
        for n in range(N - L + 1):
            C_window = C[:, n:n+L]
            
            mean_c = np.mean(C_window, axis=1, keepdims=True)
            Cn = C_window / (mean_c + 1e-8)
            
            S1 = Cn[1, :] - Cn[2, :]
            S2 = -2.0 * Cn[0, :] + Cn[1, :] + Cn[2, :]
            
            alpha = np.std(S1) / (np.std(S2) + 1e-8)
            h = S1 + alpha * S2
            
            H[n:n+L] += (h - np.mean(h))
            
        # ==========================================
        # 3. Direct Filtering
        # Detrend guarantees a zero-mean, flat DC line so the 
        # Butterworth filter doesn't panic on the Overlap-Add envelope!
        # ==========================================
        H = detrend(H)
        
        sos = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, H)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None

        filtered_signal = self.get_filtered_signal()
        if filtered_signal is None:
            return None, None, None
            
        # 1. Hanning Window
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        # 2. High-Resolution FFT
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=NFFT)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(NFFT, d=1.0/self.target_fps)

        # 3. Restrict Peak Detection to physiological band
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        # 4. Pure Argmax Peak Detection
        peak_index = np.argmax(bpm_filt_mag)
        raw_bpm = bpm_freqs[peak_index] * 60.0

        return raw_bpm, bpm_freqs, bpm_filt_mag
    
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
    
