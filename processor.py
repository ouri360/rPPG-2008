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

try:
    import cupy as cp
    USE_GPU = True
    logging.info("🚀 CuPy loaded successfully. GPU Acceleration ENABLED.")
except ImportError as e:
    import numpy as cp # Use numpy as a drop-in replacement for CuPy
    USE_GPU = False
    logging.warning(f"⚠️ CuPy not available ({e}). Falling back to CPU (NumPy).")

# Minimum number of seconds required to perform filtering and FFT analysis
MINIMUM_AMOUNT_OF_DATA = 4 
# Filter parameters for bandpass filter (these can be tuned based on expected heart rate range)
LOWCUT_HZ = 0.7         # Corresponds to ~42 BPM
HIGHCUT_HZ = 3.0        # Corresponds to ~180 BPM
ORDER = 2               # Filter order (x2 with sosfiltfilt for zero phase distortion)
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
        # Calculate maximum buffer size
        self.max_length = int(buffer_seconds * target_fps)
        
        # deques automatically pop the oldest item when maxlen is reached
        self.raw_signal = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 1 seconds)
        smoothing_frames = int(self.target_fps * 1)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 

        # Tracker for Outlier Rejection
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> float:
        weights = {
            'forehead': 0.60,
            'left_cheek': 0.20,
            'right_cheek': 0.20
        }
        
        weighted_sum = 0.0
        
        # Extract the green channel for the whole frame once (faster computation)
        green_channel = frame[:, :, 1]
        
        for region_name, polygon in rois.items():
            # 1. Create a blank black canvas
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # 2. Draw the dynamic shape-shifting polygon in solid white
            cv2.fillPoly(mask, [polygon], 255)
            
            # 3. Extract ONLY the green pixels that fall inside the white polygon
            skin_pixels = green_channel[mask == 255]
            
            if len(skin_pixels) == 0:
                continue
                
            # ==========================================
            # Pixel Sorting (Trimmed Mean)
            # Destroys Specular Glare and Shadows before they enter the buffer
            # ==========================================
            # GPU OPTIMIZATION: Send array to Ampere GPU for fast sorting
            # ==========================================
            # cp becomes either CuPy (Jetson) or NumPy (PC) dynamically
            skin_pixels_array = cp.asarray(skin_pixels)
            sorted_pixels = cp.sort(skin_pixels_array)
            
            total_pixels = len(sorted_pixels)
            bottom_trim = int(total_pixels * 0.05)
            top_trim = int(total_pixels * 0.20)
            
            if bottom_trim > 0 and top_trim > 0:
                pure_skin = sorted_pixels[bottom_trim:-top_trim]
            else:
                pure_skin = sorted_pixels
                
            mean_val = cp.mean(pure_skin)
            
            # CuPy arrays need .get() to become floats, NumPy floats don't
            if USE_GPU:
                region_val = float(mean_val.get())
            else:
                region_val = float(mean_val)
            # ==========================================
            
            weighted_sum += region_val * weights[region_name]

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

        signal = np.array(list(self.raw_signal))
        ts = np.array(list(self.timestamps))
        
        if len(ts) < 2:
            return None

        # 1. Uniform Resampling
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        # ==========================================
        # SECURITY FIX: Ensure the time window is physically wide enough 
        # to produce enough data points for the SciPy filter (padlen ~ 15).
        # ==========================================
        if len(t_uniform) <= 30: 
            return None
            
        signal_uniform = np.interp(t_uniform, ts, signal)
        
        # 2. Prevent Filter Ringing
        signal_uniform = self.remove_impulse_noise(signal_uniform)
        
        # 3. Apply Dynamic Range Compressor
        signal_uniform = self.detrend_and_normalize(signal_uniform)

        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        order = ORDER
        
        # 4. Apply the High-Precision Butterworth
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, signal_uniform)
        
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_signal) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None

        filtered_signal = self.get_filtered_signal()
        
        if filtered_signal is None:
            return None, None, None
            
        n_fft = NFFT
        
        # Apply the Hanning Window only to the filtered signal
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        # Compute the Power Spectra
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=n_fft)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)

        # Dashboard arrays
        plot_indices = np.where((frequencies >= 0.0) & (frequencies <= 3.0))[0]
        plot_freqs = frequencies[plot_indices]
        plot_filt_mag = filtered_power[plot_indices]
        
        # BPM Arrays
        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        # Find the Peak
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
        
        # Return only the Smoothed BPM, Frequencies, and Filtered Magnitude
        return smoothed_bpm, plot_freqs, plot_filt_mag
    
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
    
    @staticmethod
    def remove_impulse_noise(x: np.ndarray) -> np.ndarray:
        """
        Uses Velocity Clamping (Derivative Limiting) to turn sharp, 
        instantaneous camera glitches into gentle slopes, preventing filter ringing.
        """
        if len(x) < 2:
            return x
        
        # 1. Calculate the frame-to-frame velocity
        diffs = np.diff(x)
        
        # 2. Find the normal maximum speed of the heartbeat
        std_diff = np.std(diffs)
        
        if std_diff > 0:
            # 3. Clamp the velocity. 3.0 std allows strong heartbeats to pass, 
            # but completely stops instantaneous vertical glitches.
            clamped_diffs = np.clip(diffs, -3.0 * std_diff, 3.0 * std_diff)
            
            # 4. Reconstruct the signal by integrating the clamped velocity
            reconstructed = np.concatenate(([x[0]], x[0] + np.cumsum(clamped_diffs)))
            return reconstructed
            
        return x