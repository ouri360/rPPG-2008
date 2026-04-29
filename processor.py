"""
SignalProcessor Module for rPPG Signal Extraction
---------------------------------------
This module handles the extraction, filtering, and buffering of the raw rPPG signal.
It utilizes a purely statistical, hierarchical POS algorithm to mathematically 
reduce ambient lighting flicker and motion artifacts without Deep Learning.
"""

import numpy as np
import logging
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, detrend, sosfiltfilt

MINIMUM_AMOUNT_OF_DATA = 2 
LOWCUT_HZ = 0.7         
HIGHCUT_HZ = 3.0        
ORDER = 3               
NFFT = 8192             


class BiologicalHRTracker:
    """A simple tracker to stabilize heart rate estimates over time, preventing unrealistic jumps."""
    def __init__(self, max_jump: float = 15.0):
        self.max_jump = max_jump 
        self.last_bpm = None

    def update(self, valid_freqs_hz: np.ndarray, valid_power: np.ndarray) -> float:
        if len(valid_freqs_hz) == 0:
            return self.last_bpm if self.last_bpm else 75.0
        if self.last_bpm is None:
            best_idx = np.argmax(valid_power)
            self.last_bpm = valid_freqs_hz[best_idx] * 60.0
            return self.last_bpm
        norm_power = valid_power / (np.max(valid_power) + 1e-8)
        freq_bpm = valid_freqs_hz * 60.0
        distance_penalty = np.abs(freq_bpm - self.last_bpm) / self.max_jump
        scores = norm_power - (distance_penalty * 0.5) 
        best_idx = np.argmax(scores)
        raw_bpm = valid_freqs_hz[best_idx] * 60.0
        self.last_bpm = (0.7 * self.last_bpm) + (0.3 * raw_bpm)
        return self.last_bpm

class SignalProcessor:
    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        self.max_length = int(buffer_seconds * target_fps)
        
        self.raw_r = deque(maxlen=self.max_length)
        self.raw_g = deque(maxlen=self.max_length)
        self.raw_b = deque(maxlen=self.max_length)
        
        self.roi_keys = [
            'forehead_1', 'forehead_2', 'forehead_3',
            'left_cheek_1', 'left_cheek_2', 'left_cheek_3',
            'right_cheek_1', 'right_cheek_2', 'right_cheek_3'
        ]

        self.rois_history = {
            key: {'r': deque(maxlen=self.max_length), 
                  'g': deque(maxlen=self.max_length), 
                  'b': deque(maxlen=self.max_length)}
            for key in self.roi_keys
        }

        # Mathematical State Trackers
        self.latest_sub_alphas = {k: 1.0 for k in self.roi_keys}
        self.latest_regional_alphas = {'forehead': 1.0, 'left_cheek': 1.0, 'right_cheek': 1.0}
        self.latest_global_alpha = 1.0
        self.current_backend = "NumPy (CPU)"

        logging.info(f"SignalProcessor running securely on: {self.current_backend}")

        self.raw_signal = self.raw_g 
        self.timestamps = deque(maxlen=self.max_length)
        self.hr_tracker = BiologicalHRTracker()

    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        global_g = [] 
        for region_name in self.roi_keys:
            color_bgr = rois.get(region_name)
            if color_bgr is not None:
                b, g, r = color_bgr[0], color_bgr[1], color_bgr[2]
                self.rois_history[region_name]['b'].append(float(b))
                self.rois_history[region_name]['g'].append(float(g))
                self.rois_history[region_name]['r'].append(float(r))
                global_g.append(float(g))
            else:
                self.rois_history[region_name]['r'].append(0.0)
                self.rois_history[region_name]['g'].append(0.0)
                self.rois_history[region_name]['b'].append(0.0)

        if len(global_g) > 0:
            self.raw_g.append(float(np.mean(global_g)))
        else:
            self.raw_g.append(0.0)
        self.timestamps.append(timestamp)

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.raw_g), np.array(self.timestamps)
    
    def get_filtered_signal(self) -> Optional[np.ndarray]:
        if len(self.timestamps) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        ts = np.array(list(self.timestamps))
        time_limit = ts[-1] - self.buffer_seconds
        valid_indices = np.where(ts >= time_limit)[0]
        ts = ts[valid_indices]
        
        if len(ts) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        dt = 1.0 / self.target_fps
        t_uniform = np.arange(ts[0], ts[-1] + dt/2, dt)
        
        C_rois = {}
        for region_name in self.rois_history.keys():
            r = np.array(list(self.rois_history[region_name]['r']))[valid_indices]
            g = np.array(list(self.rois_history[region_name]['g']))[valid_indices]
            b = np.array(list(self.rois_history[region_name]['b']))[valid_indices]
            
            r_u = np.interp(t_uniform, ts, r)
            g_u = np.interp(t_uniform, ts, g)
            b_u = np.interp(t_uniform, ts, b)
            C_rois[region_name] = np.vstack([r_u, g_u, b_u])
            
        N = C_rois[self.roi_keys[0]].shape[1]
        L = int(self.target_fps * 1.6) 
        H = np.zeros(N)
        
        num_windows = N - L + 1

        for n in range(num_windows):
            alphas = {}
            S1_all = []
            S2_all = []
            noises = {}
            
            # ========================================================================
            # PHASE 1: NOISE ESTIMATION
            # ========================================================================
            for region_name in self.roi_keys:
                C_window = C_rois[region_name][:, n:n+L] 
                mean_c = np.mean(C_window, axis=1, keepdims=True)
                Cn = C_window / (mean_c + 1e-8)
                
                S1 = Cn[1, :] - Cn[2, :]
                S2 = -2.0 * Cn[0, :] + Cn[1, :] + Cn[2, :]
                
                S1_all.append(S1)
                S2_all.append(S2)
                
                # Standard Deviation (std) mathematically measures "Spread".
                # A clean, perfect heartbeat has a very low standard deviation.
                # A cheek covered in moving shadows or talking lips will have a violently 
                # high standard deviation because the pixels are rapidly changing color.
                std_s1 = np.std(S1)
                std_s2 = np.std(S2)
                
                # We calculate the POS Alpha for this specific sub-region.
                alphas[region_name] = std_s1 / (std_s2 + 1e-8)
                
                # We define the total "Noise" of this region as the sum of its chaos.
                noises[region_name] = std_s1 + std_s2

            # ========================================================================
            # PHASE 2: INVERSE-VARIANCE WEIGHTING
            # ========================================================================
            
            # THE LAW OF INVERSE VARIANCE: 
            # If a region is highly noisy (e.g., Noise = 100), we want to ignore it. 
            # If we invert it (1 / 100), the weight becomes 0.01 (Very small).
            # If a region is perfectly clean (e.g., Noise = 2), we want to trust it.
            # If we invert it (1 / 2), the weight becomes 0.50 (Very large).
            # The + 1e-8 prevents a mathematical "Divide by Zero" crash if the video is completely black.
            inv_variance = {k: 1.0 / (v + 1e-8) for k, v in noises.items()}
            
            # NORMALIZATION :
            # Right now, our inverse weights are random raw decimals. 
            # We need them to act like percentages that add up to exactly 100% (or 1.0).
            total_weight = sum(inv_variance.values())
            
            # By dividing each raw weight by the total pool of weights, we force them 
            # into a perfect 0.0 to 1.0 distribution. 
            # Example: Forehead might get 0.60 (60%), Left Cheek 0.35 (35%), Right Cheek in shadow 0.05 (5%).
            weights = {k: v / total_weight for k, v in inv_variance.items()}

            # ========================================================================
            # PHASE 3: APPLY THE WEIGHTS (FUSION)
            # ========================================================================
            
            # Now we calculate the 3 Regional Alphas. 
            # If 'forehead_1' is clean and 'forehead_2' is noisy, 'forehead_1' dominates this equation.
            def get_weighted_reg_alpha(r1, r2, r3):
                w_sum = weights[r1] + weights[r2] + weights[r3] + 1e-8
                return (weights[r1]*alphas[r1] + weights[r2]*alphas[r2] + weights[r3]*alphas[r3]) / w_sum

            alpha_forehead = get_weighted_reg_alpha('forehead_1', 'forehead_2', 'forehead_3')
            alpha_left = get_weighted_reg_alpha('left_cheek_1', 'left_cheek_2', 'left_cheek_3')
            alpha_right = get_weighted_reg_alpha('right_cheek_1', 'right_cheek_2', 'right_cheek_3')
            
            # The Global Alpha is simply the sum of every individual Alpha multiplied by its trust percentage.
            global_alpha = sum(weights[k] * alphas[k] for k in self.roi_keys)
            
            # Save telemetry for the OpenCV visualizer
            self.latest_sub_alphas = alphas.copy()
            self.latest_regional_alphas = {
                'forehead': alpha_forehead,
                'left_cheek': alpha_left,
                'right_cheek': alpha_right
            }
            self.latest_global_alpha = global_alpha
            self.latest_math_weights = weights.copy()
            
            # Finally, we physically shrink the noisy waves and boost the clean waves 
            # before we do the final POS subtraction math.
            S1_weighted = np.sum([S1_all[i] * weights[self.roi_keys[i]] for i in range(9)], axis=0)
            S2_weighted = np.sum([S2_all[i] * weights[self.roi_keys[i]] for i in range(9)], axis=0)
            
            # The final, mathematically purified heartbeat signal for this window (1.6 seconds)
            h = S1_weighted + (global_alpha * S2_weighted)
                
            H[n:n+L] += (h - np.mean(h))
            
        H_flat = H[L-1 : -(L-1)]
        if len(H_flat) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None
        
        H_detrended = detrend(H_flat) 
        sos = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        filtered_signal = sosfiltfilt(sos, H_detrended)
        return filtered_signal
    
    def estimate_heart_rate(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None
        filtered_signal = self.get_filtered_signal()
        if filtered_signal is None:
            return None, None, None
            
        window = np.hanning(len(filtered_signal))
        windowed_filt = filtered_signal * window
        
        fft_filtered_complex = np.fft.rfft(windowed_filt, n=NFFT)
        filtered_power = (np.abs(fft_filtered_complex)**2)
        frequencies = np.fft.rfftfreq(NFFT, d=1.0/self.target_fps)

        bpm_indices = np.where((frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ))[0]
        bpm_freqs = frequencies[bpm_indices]
        bpm_filt_mag = filtered_power[bpm_indices]
        
        raw_bpm = self.hr_tracker.update(bpm_freqs, bpm_filt_mag)
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
    
    def get_latest_weights(self) -> dict:
        """Returns the mathematical Inverse-Variance weights for the UI."""
        if hasattr(self, 'latest_math_weights'):
            return self.latest_math_weights
        return {k: 1.0 / len(self.roi_keys) for k in self.roi_keys}
        
    def get_alpha_telemetry(self):
        """Returns all 13 mathematically derived Alphas for the UI."""
        return self.latest_sub_alphas, self.latest_regional_alphas, self.latest_global_alpha
    
    def get_backend_name(self) -> str:
        return self.current_backend