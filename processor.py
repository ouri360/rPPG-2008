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
import os

from model import POSNet
import torch

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logging.warning("onnxruntime not installed. TensorRT hardware optimization disabled.")

MINIMUM_AMOUNT_OF_DATA = 2 
LOWCUT_HZ = 0.7         
HIGHCUT_HZ = 3.0        
ORDER = 3               
NFFT = 8192             

# --- PASTE YOUR BIOLOGICAL HR TRACKER HERE ---
class BiologicalHRTracker:
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

        # Telemetry State Trackers
        self.latest_ai_weights = {k: 0.11 for k in self.roi_keys}
        self.latest_ai_alpha = 1.0
        self.use_onnx = False

        # ==========================================
        # THE HYBRID ENGINE (TensorRT -> PyTorch)
        # ==========================================
        onnx_path = 'pos_net.onnx'
        pt_path = 'pos_net_weights.pt'

        if ORT_AVAILABLE and os.path.exists(onnx_path):
            logging.info(f"Found {onnx_path}! Initializing TensorRT Hardware Engine...")
            os.makedirs('./trt_cache', exist_ok=True)
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache',
                    # THE FIX: Tell TensorRT the absolute limits of the dynamic axes!
                    # Format: 'input_name:BatchxChannelsxSeqLenxROIs'
                    'trt_profile_min_shapes': 'input:1x2x10x9',
                    'trt_profile_opt_shapes': 'input:850x2x48x9',
                    'trt_profile_max_shapes': 'input:3000x2x200x9'
                }),
                ('CUDAExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider',
            ]
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.use_onnx = True
        else:
            logging.info("Loading PyTorch Engine (Fallback Mode)...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pos_net = POSNet(num_rois=9).to(self.device).half()
            if os.path.exists(pt_path):
                self.pos_net.load_state_dict(torch.load(pt_path, map_location=self.device))
            self.pos_net.eval()

        # ==========================================
        # Determine the Active Backend for the HUD
        # ==========================================
        self.current_backend = "Unknown"
        if self.use_onnx:
            # ONNXRuntime returns a list of active providers. The first is the primary engine.
            active_provider = self.ort_session.get_providers()[0]
            if "Tensorrt" in active_provider:
                self.current_backend = "TensorRT"
            elif "CUDA" in active_provider:
                self.current_backend = "ONNX (CUDA)"
            else:
                self.current_backend = "ONNX (CPU)"
        else:
            self.current_backend = f"PyTorch ({self.device.type.upper()})"

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
        
        roi_keys = self.roi_keys
        num_windows = N - L + 1
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
                
        # ==========================================
        # HYBRID INFERENCE EXECUTION
        # ==========================================
        if self.use_onnx:
            # 1. TENSORRT INFERENCE (BLAZING FAST C++)
            ort_inputs = {self.ort_input_name: batch_input}
            pulse_out, weights_out, alpha_out = self.ort_session.run(None, ort_inputs)
            
            # Extract Telemetry from the final window
            last_w = weights_out[-1]
            last_a = alpha_out[-1][0]
            
            self.latest_ai_weights = {key: float(last_w[i]) for i, key in enumerate(self.roi_keys)}
            self.latest_ai_alpha = float(last_a)
            h_preds_numpy = pulse_out
            
        else:
            # 2. PYTORCH INFERENCE
            with torch.no_grad():
                x_tensor = torch.from_numpy(batch_input)
                model_device = next(self.pos_net.parameters()).device
                x_tensor = x_tensor.to(model_device).half() 
                
                h_preds = self.pos_net(x_tensor) 
                h_preds_numpy = h_preds.cpu().numpy()
                
            # Extract Telemetry
            if hasattr(self.pos_net, 'latest_weights') and self.pos_net.latest_weights is not None:
                last_w = self.pos_net.latest_weights[-1]
                self.latest_ai_weights = {key: float(last_w[i]) for i, key in enumerate(self.roi_keys)}
            if hasattr(self.pos_net, 'latest_alpha') and self.pos_net.latest_alpha is not None:
                self.latest_ai_alpha = float(self.pos_net.latest_alpha[-1][0])

        for n in range(num_windows):
            h = h_preds_numpy[n]
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
        return self.latest_ai_weights
        
    def get_alpha_telemetry(self) -> Tuple[float, float]:
        ai_alpha = self.latest_ai_alpha
        math_alpha = 1.0
        
        if len(self.raw_g) > int(self.target_fps * 1.6):
            try:
                r = np.array(list(self.rois_history['forehead_2']['r'])[-50:])
                g = np.array(list(self.rois_history['forehead_2']['g'])[-50:])
                b = np.array(list(self.rois_history['forehead_2']['b'])[-50:])
                
                rn = r / (np.mean(r) + 1e-8)
                gn = g / (np.mean(g) + 1e-8)
                bn = b / (np.mean(b) + 1e-8)
                
                s1 = gn - bn
                s2 = -2.0 * rn + gn + bn
                
                math_alpha = float(np.std(s1) / (np.std(s2) + 1e-8))
            except Exception:
                pass

        return math_alpha, ai_alpha
    
    def get_backend_name(self) -> str:
        return self.current_backend