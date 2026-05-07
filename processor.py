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
        
        snr_peak = np.max(valid_power) / (np.mean(valid_power) + 1e-8)
        
        # Si le signal est pur, on supprime la pénalité de distance 
        # pour autoriser le BPM à fuir le piège des basses fréquences
        penalty_weight = 0.1 if snr_peak > 4.0 else 0.6
        
        distance_penalty = np.abs(freq_bpm - self.last_bpm) / self.max_jump
        scores = norm_power - (distance_penalty * penalty_weight) 

        best_idx = np.argmax(scores)

        # --- INTERPOLATION PARABOLIQUE ---
        if 0 < best_idx < len(valid_freqs_hz) - 1:
            alpha = scores[best_idx - 1]
            beta = scores[best_idx]
            gamma = scores[best_idx + 1]
            shift = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-8)
            exact_freq = valid_freqs_hz[best_idx] + shift * (valid_freqs_hz[1] - valid_freqs_hz[0])
        else:
            exact_freq = valid_freqs_hz[best_idx]
            
        raw_bpm = exact_freq * 60.0

        # Si la vérité est trouvée, on met à jour très vite. Sinon on lisse fortement.
        if snr_peak > 4.0:
            self.last_bpm = (0.3 * self.last_bpm) + (0.7 * raw_bpm)
        else:
            self.last_bpm = (0.85 * self.last_bpm) + (0.15 * raw_bpm)
            
        return self.last_bpm

class SignalProcessor:
    def __init__(self, buffer_seconds: int = 30, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds
        self.max_length = int(buffer_seconds * target_fps)

        self.last_confident_hz = None
        
        self.raw_r = deque(maxlen=self.max_length)
        self.raw_g = deque(maxlen=self.max_length)
        self.raw_b = deque(maxlen=self.max_length)
        
        self.roi_keys = [
            'forehead_1', 'forehead_2', 'forehead_3',
            'left_cheek',
            'right_cheek'
        ]

        self.rois_history = {
            key: {'r': deque(maxlen=self.max_length), 
                  'g': deque(maxlen=self.max_length), 
                  'b': deque(maxlen=self.max_length)}
            for key in self.roi_keys
        }

        # Mathematical State Trackers
        self.smoothed_weights = {k: 1.0 / len(self.roi_keys) for k in self.roi_keys}
        self.latest_sub_alphas = {k: 1.0 for k in self.roi_keys}
        self.latest_regional_alphas = {'forehead': 1.0, 'left_cheek': 1.0, 'right_cheek': 1.0}
        self.latest_global_alpha = 1.0
        self.current_backend = "NumPy (CPU)"

        logging.info(f"SignalProcessor running on: {self.current_backend}")

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
        H_counts = np.zeros(N)
        
        num_windows = N - L + 1
        step = 4

        local_smoothed_weights = {k: 1.0 / len(self.roi_keys) for k in self.roi_keys}

        for n in range(0, num_windows, step):            
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
                
                std_s1 = np.std(S1)
                std_s2 = np.std(S2)
                
                alphas[region_name] = std_s1 / (std_s2 + 1e-8)
                noises[region_name] = std_s1 + std_s2

            # ========================================================================
            # PHASE 2: INVERSE-VARIANCE WEIGHTING
            # ========================================================================
            inv_variance = {k: 1.0 / (v + 1e-8) for k, v in noises.items()}
            total_weight = sum(inv_variance.values())
            raw_weights = {k: v / total_weight for k, v in inv_variance.items()}

            for k in self.roi_keys:
                local_smoothed_weights[k] = (0.8 * local_smoothed_weights[k]) + (0.2 * raw_weights[k])  
            
            weights = local_smoothed_weights

            # ========================================================================
            # PHASE 3: APPLY THE WEIGHTS (TRUE REGIONAL FUSION)
            # ========================================================================
            h_final = np.zeros(L)

            for i, region_name in enumerate(self.roi_keys):
                w = weights[region_name]
                a = alphas[region_name]
                
                # Onde POS pure pour cette zone spécifique
                h_region = S1_all[i] + (a * S2_all[i])
                
                # Ajout pondéré au signal final
                h_final += (w * h_region)

            H[n:n+L] += (h_final - np.mean(h_final))
            H_counts[n:n+L] += 1 

        # ========================================================================
        # CORRECTION DE L'AMPLITUDE (OVERLAP-ADD NORMALIZATION)
        # ========================================================================
        H_counts[H_counts == 0] = 1 
        H = H / H_counts 

        # ========================================================================
        # MISE À JOUR DE L'INTERFACE 
        # ========================================================================
        # --- NOUVEAU : Recalcul des valeurs globales spécifiquement pour le Dashboard ---
        def get_weighted_reg_alpha(r1, r2, r3):
            w_sum = weights[r1] + weights[r2] + weights[r3] + 1e-8
            return (weights[r1]*alphas[r1] + weights[r2]*alphas[r2] + weights[r3]*alphas[r3]) / w_sum

        alpha_forehead = get_weighted_reg_alpha('forehead_1', 'forehead_2', 'forehead_3')
        alpha_left = alphas['left_cheek']
        alpha_right = alphas['right_cheek']
        # -------------------------------------------------------------------------------

        self.latest_sub_alphas = alphas.copy()
        self.latest_regional_alphas = {
            'forehead': alpha_forehead,
            'left_cheek': alpha_left,
            'right_cheek': alpha_right
        }
        
        global_alpha = sum(weights[k] * alphas[k] for k in self.roi_keys)
        self.latest_global_alpha = global_alpha
        self.latest_math_weights = weights.copy()

        H_flat = H[L-1 : -(L-1)]
        if len(H_flat) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None
        
        H_detrended = detrend(H_flat) 
        
        # --- VOIE 1 : (Pour la FFT) ---
        # La FFT DOIT voir tout le spectre (0.7 à 3.0 Hz)
        sos_math = butter(ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=self.target_fps, output='sos')
        signal_math = sosfiltfilt(sos_math, H_detrended)
        
        # --- VOIE 2 : (Pour le Dashboard ECG) ---
        # On nettoie l'onde 
        if self.last_confident_hz is not None:
            low_bound = max(LOWCUT_HZ, self.last_confident_hz - 0.35)
            high_bound = min(HIGHCUT_HZ, self.last_confident_hz + 0.35)
            sos_visuel = butter(ORDER, [low_bound, high_bound], btype='bandpass', fs=self.target_fps, output='sos')
            signal_visuel = sosfiltfilt(sos_visuel, H_detrended)
        else:
            signal_visuel = signal_math
        
        # On renvoie les deux signaux 
        return signal_math, signal_visuel
    
    def estimate_heart_rate(self, filtered_signal: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.raw_g) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None, None, None
            
        if filtered_signal is None:
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

        # On calcule le rapport signal/bruit du pic actuel
        peak_power = np.max(bpm_filt_mag)
        mean_power = np.mean(bpm_filt_mag)
        snr = peak_power / (mean_power + 1e-8)

        # On enregistre la fréquence (Hz) pour le prochain passage du filtre
        if raw_bpm is not None:
            if snr > 3.5:
                # Signal clair : on verrouille le filtre autour de la valeur
                self.last_confident_hz = raw_bpm / 60.0
            elif snr < 2.0:
                # Signal détruit (Mouvement brusque) : on CASSE le verrou !
                self.last_confident_hz = None
            
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