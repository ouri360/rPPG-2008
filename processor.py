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
        
        # In order to separately track the ROIs, we maintain a history of pixel values for each region.
        self.rois_history = {
            'forehead': {'r': deque(maxlen=self.max_length), 'g': deque(maxlen=self.max_length), 'b': deque(maxlen=self.max_length)},
            'left_cheek': {'r': deque(maxlen=self.max_length), 'g': deque(maxlen=self.max_length), 'b': deque(maxlen=self.max_length)},
            'right_cheek': {'r': deque(maxlen=self.max_length), 'g': deque(maxlen=self.max_length), 'b': deque(maxlen=self.max_length)}
        }

        
        # ==========================================
        # Load the trained POSNet
        # ==========================================
        # 1. Détection automatique du GPU (pour la Jetson)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Instanciation ET transfert du modèle vers la puce graphique (POSNet is a lightweight CNN)
        self.pos_net = POSNet(num_rois=3).to(self.device)
        
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
        
        logging.info("SignalProcessor using to POS RGB Architecture.")
    
    def extract_and_buffer_multi(self, frame: np.ndarray, rois: dict, timestamp: float) -> None:
        b_channel, g_channel, r_channel = cv2.split(frame)
        
        global_g = [] # Pour maintenir la compatibilité avec self.raw_g
        
        for region_name, polygon in rois.items():
            # Il faut créer un masque spécifique à la région courante
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon], 255)
            
            # Extraction des pixels de CETTE région uniquement
            r_pixels = r_channel[roi_mask == 255]
            g_pixels = g_channel[roi_mask == 255]
            b_pixels = b_channel[roi_mask == 255]
            
            if len(g_pixels) > 0:
                self.rois_history[region_name]['r'].append(float(np.mean(r_pixels)))
                self.rois_history[region_name]['g'].append(float(np.mean(g_pixels)))
                self.rois_history[region_name]['b'].append(float(np.mean(b_pixels)))
                global_g.extend(g_pixels)
            else:
                self.rois_history[region_name]['r'].append(0.0)
                self.rois_history[region_name]['g'].append(0.0)
                self.rois_history[region_name]['b'].append(0.0)

        # Maintien de la variable raw_g pour le Graph 1 et les checks de longueur
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
            
        N = C_rois['forehead'].shape[1]
        L = int(self.target_fps * 1.6) 
        H = np.zeros(N)
        
        roi_keys = ['forehead', 'left_cheek', 'right_cheek']
        num_windows = N - L + 1

        # 1. Prepare a single MASSIVE batch tensor on the CPU for all windows
        # Shape: (Batch_Size=num_windows, Channels=2, SeqLen=L, ROIs=3)
        batch_input = np.zeros((num_windows, 2, L, 3), dtype=np.float32)

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
            
            # Send to GPU
            model_device = next(self.pos_net.parameters()).device
            x_tensor = x_tensor.to(model_device)
            
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
        
        # 4. Argmax Peak Detection
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
    
