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
import cv2
from collections import deque
from typing import Tuple, Optional
from scipy.signal import filtfilt, firwin, detrend
from sklearn.decomposition import FastICA

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 4 
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 3.0        # Corresponds to ~180 BPM
ORDER = 3               # Filter order
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
        # WE NOW NEED 3 BUFFERS FOR RGB!
        self.raw_r = deque(maxlen=self.max_length)
        self.raw_g = deque(maxlen=self.max_length)
        self.raw_b = deque(maxlen=self.max_length)

        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 1 second)
        smoothing_frames = int(self.target_fps * 1)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 

        # Tracker for Outlier Rejection
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        b_channel, g_channel, r_channel = cv2.split(frame)
        master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Area-Weighted Super Mask
        for region_name, polygon in rois.items():
            cv2.fillPoly(master_mask, [polygon], 255)
            
        r_pixels = r_channel[master_mask == 255]    
        g_pixels = g_channel[master_mask == 255]
        b_pixels = b_channel[master_mask == 255]
        
        if len(g_pixels) > 0:
            final_r = float(np.mean(r_pixels))
            final_g = float(np.mean(g_pixels))
            final_b = float(np.mean(b_pixels))
        else:
            final_r, final_g, final_b = 0.0, 0.0, 0.0

        self.raw_r.append(final_r)
        self.raw_g.append(final_g)
        self.raw_b.append(final_b)
        self.timestamps.append(timestamp)
    
    def get_ica_signal(self) -> Tuple[Optional[np.ndarray], Optional[list]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None

        ts = np.array(list(self.timestamps))
        r = np.array(list(self.raw_r))
        g = np.array(list(self.raw_g))
        b = np.array(list(self.raw_b))
        
        time_limit = ts[-1] - self.buffer_seconds
        valid_indices = np.where(ts >= time_limit)[0]
        
        r = r[valid_indices]
        g = g[valid_indices]
        b = b[valid_indices]
        ts = ts[valid_indices]
        
        if len(ts) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        r_uni = np.interp(t_uniform, ts, r)
        g_uni = np.interp(t_uniform, ts, g)
        b_uni = np.interp(t_uniform, ts, b)
        
        r_norm = self.detrend_and_normalize(r_uni)
        g_norm = self.detrend_and_normalize(g_uni)
        b_norm = self.detrend_and_normalize(b_uni)
        
        X = np.column_stack((r_norm, g_norm, b_norm))
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica = FastICA(n_components=3, random_state=42, max_iter=1000)
                S = ica.fit_transform(X) 
        except ValueError:
            return None, None
            
        taps = firwin(
            numtaps=71, 
            cutoff=[LOWCUT_HZ, HIGHCUT_HZ], 
            fs=self.target_fps, 
            pass_zero=False
        )
        safe_padlen = min(MINIMUM_AMOUNT_OF_DATA * len(taps), len(g_uni) - 1)
        
        all_filtered_components = []
        max_power = -1.0
        best_component_idx = 0
        
        # ==========================================
        # Dynamic Component Selection
        # We find the component with the strongest physiological heartbeat.
        # ==========================================
        for i in range(3):
            filtered_comp = filtfilt(taps, 1.0, S[:, i], padlen=safe_padlen)
            all_filtered_components.append(filtered_comp)
            
            # Mini-FFT to find the dominant physiological power
            window = np.hanning(len(filtered_comp))
            fft_res = np.fft.rfft(filtered_comp * window, n=NFFT)
            power = np.abs(fft_res)**2
            freqs = np.fft.rfftfreq(NFFT, d=1.0/self.target_fps)
            
            # Look only in the human heart rate band
            valid_band = np.where((freqs >= LOWCUT_HZ) & (freqs <= HIGHCUT_HZ))[0]
            peak_power_in_band = np.max(power[valid_band])
            
            # Keep track of the component with the highest peak power
            if peak_power_in_band > max_power:
                max_power = peak_power_in_band
                best_component_idx = i
                
        best_component = all_filtered_components[best_component_idx]
                
        return best_component, all_filtered_components
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None, None, None

        # Call the ICA method and accept its strictly selected component
        selected_component, all_components = self.get_ica_signal()
        
        # Safety check: Ensure ICA actually returned a valid component
        if selected_component is None:
            return None, None, None, None, None
        
        n_fft = NFFT
        
        # Apply the Hanning Window to our strictly selected component
        window = np.hanning(len(selected_component))
        windowed_filt = selected_component * window
        
        # Compute the Power Spectra
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
        
        # Parabolic interpolation for sub-bin frequency resolution
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: 
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df

        raw_bpm = dominant_freq * 60.0
        
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        # Return the selected_component so your matplotlib dashboard updates correctly
        return smoothed_bpm, plot_freqs, plot_filt_mag, selected_component, all_components
    
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
    
    @staticmethod
    def detrend_and_normalize(x: np.ndarray) -> np.ndarray:
        """Removes linear trend and standardizes variance to act as a dynamic range compressor."""
        x = np.array(x, dtype=np.float64)
        x = detrend(x)
        
        std = np.std(x)
        if std == 0:
            return x
            
        return (x - np.mean(x)) / (std + 1e-8)

