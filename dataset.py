"""
UBFC-Phys Dataset Loader for POSNet
---------------------------------------
Pre-processes rPPG video files and ground truth medical data into PyTorch tensors.
Aligns video frames and medical sensor data to a strict temporal grid, extracts
the POS S1/S2 signals, and segments them into training windows.
"""

import cv2
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
from typing import List, Tuple
import os                           
import concurrent.futures

from detector import FaceDetector
from gt import GroundTruthReader

from tqdm import tqdm

class UBFCPhysDataset(Dataset):
    """
    PyTorch Dataset for rPPG models. 
    Processes videos to extract spatial ROI signals and aligns them with Ground Truth waveforms.
    """

    def __init__(self, video_paths: List[str], gt_paths: List[str]) -> None:
        """
        Initializes the dataset by auto-detecting parameters and processing videos.
        
        Args:
            video_paths (List[str]): List of file paths to the video files.
            gt_paths (List[str]): List of file paths to the corresponding ground truth files.
        """
        if not video_paths or len(video_paths) != len(gt_paths):
            raise ValueError("Video paths and ground truth paths must be matched and non-empty.")

        # 1. Auto-Detect Hardware/Dataset Parameters
        self.target_fps = self._auto_detect_fps(video_paths[0])
        
        # 1.6 seconds is the ideal window to capture at least one cardiac cycle 
        # for a broad pulse-rate range [40, 240] BPM.
        self.seq_len = int(self.target_fps * 3)  # 3 seconds windows for more stable training (can be adjusted)
        
        logging.info(f"Auto-detected FPS: {self.target_fps:.1f}. Calculated SeqLen: {self.seq_len} frames.")

        self.x_data: List[Tensor] = []
        self.y_data: List[Tensor] = []
        self.roi_keys = [
            'forehead_1', 'forehead_2', 'forehead_3',
            'left_cheek_1', 'left_cheek_2', 'left_cheek_3',
            'right_cheek_1', 'right_cheek_2', 'right_cheek_3'
        ]
        
        # Determine how many CPU cores you have (Cap at 8 to prevent RAM overflow)
        max_threads = min(os.cpu_count() or 4, 8)
        logging.info(f"Accelerating extraction using {max_threads} parallel threads...")

        # 2. Process multiple videos simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all videos to the thread pool
            futures = [executor.submit(self._process_video, v, g) for v, g in zip(video_paths, gt_paths)]
            
            # Create ONE clean progress bar for the overall video count
            with tqdm(total=len(video_paths), desc="Processing Videos", unit="vid") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result() # Catch any errors that happen inside the thread
                    except Exception as e:
                        logging.error(f"Thread crashed: {e}")
                    pbar.update(1)
            
        if not self.x_data:
            raise RuntimeError("No valid data could be extracted from the provided files.")
            
        self.x_data_tensor = torch.stack(self.x_data)
        self.y_data_tensor = torch.stack(self.y_data)
        
        logging.info(f"Dataset loaded: {len(self.x_data_tensor)} windows generated.")

    def _auto_detect_fps(self, video_path: str) -> float:
        """Reads the native frames per second from the video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path} to detect FPS.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Fallback to 30.0 if metadata is corrupted
        return fps if (fps > 0 and not np.isnan(fps)) else 30.0

    def _process_video(self, video_path: str, gt_path: str) -> None:
        """
        Extracts frames, calculates RGB means for ROIs, aligns GT, and segments into windows.
        """
        detector = FaceDetector()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = self.target_fps

        # Get total frames for the progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read Ground Truth
        gt_reader = GroundTruthReader(gt_path)
        if len(gt_reader.timestamps) == 0 or len(gt_reader.gt_signal) == 0:
            logging.error(f"No valid ground truth waveform found in: {gt_path}")
            cap.release()
            return

        rois_history = {key: {'r': [], 'g': [], 'b': []} for key in self.roi_keys}
        video_timestamps = []
        
        frame_idx = 0
        
        logging.info(f"Processing: {video_path} ({total_frames} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            rois = detector.get_face_mesh_rois(frame)
            
            if rois:
                b_channel, g_channel, r_channel = cv2.split(frame)
                
                for region_name in self.roi_keys:
                    polygon = rois.get(region_name)
                    if polygon is not None:
                        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(roi_mask, [polygon], 255)
                        
                        r_pixels = r_channel[roi_mask == 255]
                        g_pixels = g_channel[roi_mask == 255]
                        b_pixels = b_channel[roi_mask == 255]
                        
                        if len(g_pixels) > 0:
                            rois_history[region_name]['r'].append(float(np.mean(r_pixels)))
                            rois_history[region_name]['g'].append(float(np.mean(g_pixels)))
                            rois_history[region_name]['b'].append(float(np.mean(b_pixels)))
                        else:
                            rois_history[region_name]['r'].append(0.0)
                            rois_history[region_name]['g'].append(0.0)
                            rois_history[region_name]['b'].append(0.0)
                    else:
                        rois_history[region_name]['r'].append(0.0)
                        rois_history[region_name]['g'].append(0.0)
                        rois_history[region_name]['b'].append(0.0)
                        
                video_timestamps.append(timestamp)
                
            frame_idx += 1
                
        cap.release()

        if len(video_timestamps) < self.seq_len:
            logging.warning(f"Video too short to process: {video_path}")
            return

        # Interpolate everything to the strict target_fps grid
        v_ts = np.array(video_timestamps)
        dt = 1.0 / self.target_fps
        t_uniform = np.arange(v_ts[0], v_ts[-1], dt)
        
        c_rois = {}
        for region in self.roi_keys:
            r = np.array(rois_history[region]['r'])
            g = np.array(rois_history[region]['g'])
            b = np.array(rois_history[region]['b'])
            
            r_u = np.interp(t_uniform, v_ts, r)
            g_u = np.interp(t_uniform, v_ts, g)
            b_u = np.interp(t_uniform, v_ts, b)
            c_rois[region] = np.vstack([r_u, g_u, b_u])  # Shape: (3, TotalFrames)

        # Interpolate the ground truth signal to the exact same time axis
        gt_signal_uniform = np.interp(t_uniform, gt_reader.timestamps, gt_reader.gt_signal)

        # Slice into overlapping or sequential windows
        total_frames = len(t_uniform)
        step_size = self.seq_len // 2  # 50% overlap for more training data
        
        for n in range(0, total_frames - self.seq_len + 1, step_size):
            # 1. Slice GT
            gt_window = gt_signal_uniform[n : n + self.seq_len]
            
            # Ensure GT has variation (avoid flatline segments where loss goes to NaN)
            if np.std(gt_window) < 1e-6:
                continue

            # 2. Slice and normalize video ROIs
            tensor_input = np.zeros((2, self.seq_len, len(self.roi_keys)), dtype=np.float32)
            
            for roi_idx, region in enumerate(self.roi_keys):
                c_window = c_rois[region][:, n : n + self.seq_len]
                
                # POS Temporal Normalization
                mean_c = np.mean(c_window, axis=1, keepdims=True)
                cn = c_window / (mean_c + 1e-8)
                
                # POS Projection Math
                s1 = cn[1, :] - cn[2, :]
                s2 = -2.0 * cn[0, :] + cn[1, :] + cn[2, :]
                
                tensor_input[0, :, roi_idx] = s1
                tensor_input[1, :, roi_idx] = s2
                
            self.x_data.append(torch.from_numpy(tensor_input))
            self.y_data.append(torch.from_numpy(gt_window.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.x_data_tensor)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x_data_tensor[idx], self.y_data_tensor[idx]