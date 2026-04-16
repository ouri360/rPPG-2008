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
from scipy.signal import butter, sosfiltfilt
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
        # DSP UPGRADE: Overlap-Add Envelope Flattening
        # We must count exactly how many times each frame is overlapped
        # to flatten the trapezoidal shape!
        # ==========================================
        overlap_count = np.zeros(N)
        
        for n in range(N - L + 1):
            C_window = C[:, n:n+L]
            
            mean_c = np.mean(C_window, axis=1, keepdims=True)
            Cn = C_window / (mean_c + 1e-8)
            
            S1 = Cn[1, :] - Cn[2, :]
            S2 = -2.0 * Cn[0, :] + Cn[1, :] + Cn[2, :]
            
            alpha = np.std(S1) / (np.std(S2) + 1e-8)
            
            h = S1 + alpha * S2
            h_zero_mean = h - np.mean(h)
            
            H[n:n+L] += h_zero_mean
            overlap_count[n:n+L] += 1
            
        # 1. Flatten the mountain back into a flat line
        H = H / (overlap_count + 1e-8)
        
        # 2. Standardize the dynamic range before it hits the Bandpass filter
        H = (H - np.mean(H)) / (np.std(H) + 1e-8)
        
        sos = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, H)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None

        filtered_signal = self.get_filtered_signal()
        if filtered_signal is None:
            return None, None, None
            
        # 1. Hanning Window to prevent Spectral Leakage at the edges
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        # 2. High-Resolution FFT (Zero-padded to 8192 bins)
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=NFFT)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(NFFT, d=1.0/self.target_fps)

        # 3. Restrict Peak Detection strictly to the human physiological band
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        # 4. State-of-the-Art Peak Detection
        peak_index = np.argmax(bpm_filt_mag)
        dominant_freq = bpm_freqs[peak_index]
        
        # 5. Parabolic Interpolation for ultra-precision (fixes sub-bin errors)
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: 
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df

        # The true mathematically calculated heart rate
        raw_bpm = dominant_freq * 60.0
        
        # Plotting arrays (expanded to 4.0 Hz to show the noise floor being crushed)
        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 4.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_filt_mag = filtered_power[plot_indices]
        
        return raw_bpm, plot_freqs, plot_filt_mag
    
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
    
