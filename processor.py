"""
SignalProcessor Module for rPPG Signal Extraction (POS Upgraded)
---------------------------------------
This module handles the extraction, filtering, and buffering of the raw rPPG signal.
It utilizes the POS (Plane-Orthogonal-to-Skin) algorithm 
to mathematically reduce ambient lighting flicker and motion artifacts.
"""

import numpy as np
import logging
from collections import deque
from typing import Tuple, Optional
from scipy.signal import butter, detrend, sosfiltfilt
import cv2

from model import POSNet
import torch

MINIMUM_AMOUNT_OF_DATA = 2 
LOWCUT_HZ = 0.7         # Lowered to 42 BPM to safely capture resting heart rates
HIGHCUT_HZ = 3.0        # Raised to 180 BPM to capture elevated heart rates
ORDER = 3               # Standardized order for steep frequency cutoffs
NFFT = 8192             


class BiologicalHRTracker:
    """
    State-of-the-Art post-processing filter for rPPG.
    Replaces naive argmax with biological momentum tracking to ignore motion spikes.
    """
    def __init__(self, max_jump: float = 15.0):
        self.max_jump = max_jump # Maximum realistic BPM change per second
        self.last_bpm = None

    def update(self, valid_freqs_hz: np.ndarray, valid_power: np.ndarray) -> float:
        """Scores peaks based on power and biological momentum."""
        if len(valid_freqs_hz) == 0:
            return self.last_bpm if self.last_bpm else 75.0

        # If first frame, trust the absolute highest peak
        if self.last_bpm is None:
            best_idx = np.argmax(valid_power)
            self.last_bpm = valid_freqs_hz[best_idx] * 60.0
            return self.last_bpm

        # Normalize the power spectrum so the highest peak is exactly 1.0
        norm_power = valid_power / (np.max(valid_power) + 1e-8)
        
        # Calculate the Biological Penalty (Distance from previous BPM)
        freq_bpm = valid_freqs_hz * 60.0
        distance_penalty = np.abs(freq_bpm - self.last_bpm) / self.max_jump
        
        # Score = Power - (Penalty * Weight)
        scores = norm_power - (distance_penalty * 0.5) 
        
        # Select the biologically logical peak
        best_idx = np.argmax(scores)
        raw_bpm = valid_freqs_hz[best_idx] * 60.0
        
        # Apply an Exponential Moving Average (EMA) to smooth out micro-jitters
        self.last_bpm = (0.7 * self.last_bpm) + (0.3 * raw_bpm)
        
        return self.last_bpm


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
        
        # The 9 mathematically sliced regions
        self.roi_keys = [
            'forehead_1', 'forehead_2', 'forehead_3',
            'left_cheek_1', 'left_cheek_2', 'left_cheek_3',
            'right_cheek_1', 'right_cheek_2', 'right_cheek_3'
        ]

        # Dynamically create the history deque for all 9 regions
        self.rois_history = {
            key: {'r': deque(maxlen=self.max_length), 
                  'g': deque(maxlen=self.max_length), 
                  'b': deque(maxlen=self.max_length)}
            for key in self.roi_keys
        }

        # ==========================================
        # Load the trained POSNet
        # ==========================================
        # 1. Détection automatique du GPU (pour la Jetson)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Instanciation ET transfert du modèle vers la puce graphique (POSNet is a lightweight CNN)
        self.pos_net = POSNet(num_rois=9).to(self.device).half()  # Use half precision for faster inference on compatible GPUs
        
        # 3. Chargement sécurisé avec `map_location` pour aligner la mémoire
        weights_path = 'pos_net_weights.pt'
        try:
            self.pos_net.load_state_dict(torch.load(weights_path, map_location=self.device))
            logging.info(f"Successfully loaded POSNet weights from {weights_path} onto {self.device}")
        except FileNotFoundError:
            logging.error(f"CRITICAL: Could not find {weights_path}! You must run train.py first.")
            
        # 4. Verrouillage du modèle en mode Inférence
        self.pos_net.eval()

        # Alias for main.py Graph 1 backwards compatibility (shows Green channel)
        self.raw_signal = self.raw_g 
        
        self.timestamps = deque(maxlen=self.max_length)

        smoothing_frames = int(self.target_fps * 1)
        self.bpm_buffer = deque(maxlen=smoothing_frames) 
        self.last_valid_bpm = None
        
        # Initialize the Biological Tracker
        self.hr_tracker = BiologicalHRTracker()

        logging.info("SignalProcessor using to POS RGB Architecture.")
    
    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        # We no longer need cv2.split or cv2.fillPoly here!
        # 'rois' now contains the exact (B, G, R) color values directly from the detector.
        
        global_g = [] # For legacy Graph 1 support
        
        for region_name in self.roi_keys:
            color_bgr = rois.get(region_name)
            
            if color_bgr is not None:
                # Extracted colors are in BGR format from OpenCV
                b, g, r = color_bgr[0], color_bgr[1], color_bgr[2]
                
                self.rois_history[region_name]['b'].append(float(b))
                self.rois_history[region_name]['g'].append(float(g))
                self.rois_history[region_name]['r'].append(float(r))
                global_g.append(float(g))
            else:
                self.rois_history[region_name]['r'].append(0.0)
                self.rois_history[region_name]['g'].append(0.0)
                self.rois_history[region_name]['b'].append(0.0)

        # Maintain raw_g for overall framerate checks and legacy systems
        if len(global_g) > 0:
            self.raw_g.append(float(np.mean(global_g)))
        else:
            self.raw_g.append(0.0)

        self.timestamps.append(timestamp)

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.raw_g), np.array(self.timestamps)
    
    def get_filtered_signal(self) -> Optional[np.ndarray]:
        # On vérifie sur les timestamps puisque c'est la source de vérité
        if len(self.timestamps) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        ts = np.array(list(self.timestamps))
        time_limit = ts[-1] - self.buffer_seconds
        valid_indices = np.where(ts >= time_limit)[0]
        ts = ts[valid_indices]
        
        if len(ts) < self.target_fps * MINIMUM_AMOUNT_OF_DATA:
            return None

        # 1. Resampling Temporel pour CHAQUE région
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
            
        # Dynamically grabs the length of the first available region (e.g., 'forehead_1')
        N = C_rois[self.roi_keys[0]].shape[1]
        L = int(self.target_fps * 1.6) 
        H = np.zeros(N)
        
        roi_keys = self.roi_keys
        num_windows = N - L + 1

        # 1. Prepare a single MASSIVE batch tensor on the CPU for all windows
        # Shape: (Batch_Size=num_windows, Channels=2, SeqLen=L, ROIs=9)
        batch_input = np.zeros((num_windows, 2, L, 9), dtype=np.float32)

        for n in range(num_windows):
            for roi_idx, region_name in enumerate(roi_keys):
                
                C_window = C_rois[region_name][:, n:n+L] 
                
                mean_c = np.mean(C_window, axis=1, keepdims=True)
                Cn = C_window / (mean_c + 1e-8)
                
                S1 = Cn[1, :] - Cn[2, :]
                S2 = -2.0 * Cn[0, :] + Cn[1, :] + Cn[2, :]
                
                batch_input[n, 0, :, roi_idx] = S1
                batch_input[n, 1, :, roi_idx] = S2
                
        # 2. SPEEDUP: Send the entire batch to the GPU ONCE
        with torch.no_grad():
            x_tensor = torch.from_numpy(batch_input)
            
            # Send to GPU and convert to half precision for faster inference
            model_device = next(self.pos_net.parameters()).device
            x_tensor = x_tensor.to(model_device).half()  # Use half precision for faster inference
            
            # The GPU processes all 800+ windows simultaneously
            h_preds = self.pos_net(x_tensor) 
            
            # Bring all 800+ predictions back to the CPU ONCE
            h_preds_numpy = h_preds.cpu().numpy() # Shape will be (num_windows, L)
            
        # 3. Apply the Overlap-Add using the predicted array
        for n in range(num_windows):
            h = h_preds_numpy[n]
            H[n:n+L] += (h - np.mean(h))
            
            
        # ==========================================
        # 3. Detrending + slicing edges
        # Detrend guarantees a zero-mean, flat DC line so the 
        # Butterworth filter doesn't panic on the Overlap-Add envelope!
        # ==========================================
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
        
        # 4. UPGRADE: Biological Peak Tracking (Replaces Argmax)
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
        """Returns the most recent spatial attention weights calculated by the neural network."""
        # If the model hasn't calculated anything yet, return equal baseline weights
        if not hasattr(self.pos_net, 'latest_weights') or self.pos_net.latest_weights is None:
            return {k: 0.11 for k in self.roi_keys}
            
        # Get the weights from the very last sliding window evaluated
        last_window_weights = self.pos_net.latest_weights[-1]
        
        # Map them back to the 9 region keys
        return {key: float(last_window_weights[i]) for i, key in enumerate(self.roi_keys)}
    
    def get_alpha_telemetry(self) -> Tuple[float, float]:
        """Returns (Math_Alpha, AI_Alpha) for the UI dashboard."""
        ai_alpha = 1.0
        math_alpha = 1.0
        
        # 1. Get the AI's Alpha
        if hasattr(self.pos_net, 'latest_alpha') and self.pos_net.latest_alpha is not None:
            # Grab the alpha from the very last sliding window
            ai_alpha = float(self.pos_net.latest_alpha[-1][0])
            
        # 2. Calculate the traditional Math Alpha
        # Math Alpha = Standard Deviation of S1 / Standard Deviation of S2
        if len(self.raw_g) > int(self.target_fps * 1.6):
            try:
                # Grab the most recent forehead colors to do a quick math check
                r = np.array(list(self.rois_history['forehead_2']['r'])[-50:])
                g = np.array(list(self.rois_history['forehead_2']['g'])[-50:])
                b = np.array(list(self.rois_history['forehead_2']['b'])[-50:])
                
                # Normalize
                rn = r / (np.mean(r) + 1e-8)
                gn = g / (np.mean(g) + 1e-8)
                bn = b / (np.mean(b) + 1e-8)
                
                s1 = gn - bn
                s2 = -2.0 * rn + gn + bn
                
                # The classic 2016 POS equation
                math_alpha = float(np.std(s1) / (np.std(s2) + 1e-8))
            except Exception:
                pass

        return math_alpha, ai_alpha
