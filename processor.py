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
from sklearn.decomposition import FastICA

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
        # WE NOW NEED 3 BUFFERS FOR RGB!
        self.raw_r = deque(maxlen=self.max_length)
        self.raw_g = deque(maxlen=self.max_length)
        self.raw_b = deque(maxlen=self.max_length)

        self.timestamps = deque(maxlen=self.max_length)

        # Buffer for smoothing BPM estimates over time (average over the last 5 seconds)
        smoothing_frames = int(self.target_fps * 5)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 

        # Tracker for Outlier Rejection
        self.last_valid_bpm = None
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        """
        Calculates a weighted spatial average from multiple ROIs.
        """
        # Define the weights (must sum to 1.0)
        weights = {
            'forehead': 0.60,
            'left_cheek': 0.20,
            'right_cheek': 0.20
        }
        
        sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
        
        for region_name, box in rois.items():
            x, y, w, h = box
            if y+h > frame.shape[0] or x+w > frame.shape[1]:
                continue

            cropped_roi = frame[y:y+h, x:x+w]
            
            # OpenCV is BGR (Blue=0, Green=1, Red=2)
            b_mean = float(np.mean(cropped_roi[:, :, 0]))
            g_mean = float(np.mean(cropped_roi[:, :, 1]))
            r_mean = float(np.mean(cropped_roi[:, :, 2]))
            
            sum_b += b_mean * weights[region_name]
            sum_g += g_mean * weights[region_name]
            sum_r += r_mean * weights[region_name]

        # Append to our 3 separate color buffers
        self.raw_b.append(sum_b)
        self.raw_g.append(sum_g)
        self.raw_r.append(sum_r)
        
        self.timestamps.append(timestamp)

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the current buffers as NumPy arrays for filtering/FFT.
        """
        return np.array(self.raw_signal), np.array(self.timestamps)
    
    def get_ica_signal(self) -> Tuple[Optional[np.ndarray], Optional[list]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None

        ts = np.array(list(self.timestamps))
        r = np.array(list(self.raw_r))
        g = np.array(list(self.raw_g))
        b = np.array(list(self.raw_b))
        
        if len(ts) < 2: return None, None

        # 1. Uniform Resampling
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        r_uni = np.interp(t_uniform, ts, r)
        g_uni = np.interp(t_uniform, ts, g)
        b_uni = np.interp(t_uniform, ts, b)
        
        # 2. Detrend and Normalize (Crucial for Poh 2010)
        r_norm = self.detrend_and_normalize(r_uni)
        g_norm = self.detrend_and_normalize(g_uni)
        b_norm = self.detrend_and_normalize(b_uni)
        
        # 3. Stack into a matrix
        X = np.column_stack((r_norm, g_norm, b_norm))
        
        # 4. Apply FastICA
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica = FastICA(n_components=3, random_state=42)
                S = ica.fit_transform(X) 
        except ValueError:
            return None, None
            
        # ==========================================
        # POH 2010 EXACT IMPLEMENTATION: Spectral Peak Selection
        # Instead of variance, we find the component with the sharpest 
        # periodic frequency peak in the human heart rate band.
        # ==========================================
        lowcut = LOWCUT_HZ
        highcut = HIGHCUT_HZ
        sos = butter(ORDER, [lowcut, highcut], btype='bandpass', fs=self.target_fps, output='sos')
        
        best_component = None
        max_snr = -1
        all_filtered_components = []
        
        n_fft = NFFT
        frequencies = np.fft.rfftfreq(n_fft, d=1.0/self.target_fps)
        
        # Only look for peaks within the human heart rate band
        valid_idx = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        
        for i in range(3):
            # 1. Apply the bandpass filter
            filtered_comp = sosfiltfilt(sos, S[:, i])
            all_filtered_components.append(filtered_comp)
            
            # 2. Run FFT to analyze the frequency purity
            window = np.hanning(len(filtered_comp))
            power_spectrum = np.abs(np.fft.rfft(filtered_comp * window, n=n_fft))**2
            
            passband_power = power_spectrum[valid_idx]
            
            # 3. Find the exact index of the tallest peak
            peak_idx = np.argmax(passband_power)
            
            # 4. Calculate SNR (Signal vs Background Noise)
            # We sum the peak and its immediate neighbors to capture the full heartbeat energy
            peak_region = passband_power[max(0, peak_idx-1) : min(len(passband_power), peak_idx+2)]
            peak_power = np.sum(peak_region)
            
            total_power = np.sum(passband_power)
            background_power = total_power - peak_power
            
            snr = peak_power / (background_power + 1e-8)
            
            # 5. Select the component with the highest SNR (the cleanest wave)
            if snr > max_snr:
                max_snr = snr
                best_component = filtered_comp
                
        return best_component, all_filtered_components
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None, None, None

        # Call our new ICA method
        best_component, all_components = self.get_ica_signal()
        
        if best_component is None:
            return None, None, None, None, None
            
        n_fft = NFFT
        
        # Apply the Hanning Window to the BEST component
        window = np.hanning(len(best_component))
        windowed_filt = best_component * window
        
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
        
        if 0 < peak_index < len(bpm_filt_mag) - 1:
            y0, y1, y2 = bpm_filt_mag[peak_index-1 : peak_index+2]
            if (y0 - 2*y1 + y2) != 0: 
                x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                df = bpm_freqs[1] - bpm_freqs[0]
                dominant_freq += x * df

        raw_bpm = dominant_freq * 60.0
        
        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        
        # Notice we are now returning 5 items!
        return smoothed_bpm, plot_freqs, plot_filt_mag, best_component, all_components
    
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