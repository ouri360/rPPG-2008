"""
Main application for rPPG signal extraction (Jetson Orin Optimized).
---------------------------------------
Integrates webcam streaming, ML face meshing, and signal processing.
Features an OpenCV, zero-latency medical dashboard. (No Matplotlib)
"""

import cv2
import logging
import numpy as np
import time

from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader


def main():
    """Main function to run the rPPG benchmark on a specified video and ground truth."""

    # ==========================================
    # USER CONFIGURATION
    # ==========================================
    #VIDEO_SOURCE = "dataset/UBFC-rPPG-Set2-Realistic/vid_subject4.avi"
    #GT_FILE = "dataset/UBFC-rPPG-Set2-Realistic/gt_subject4.txt"

    VIDEO_SOURCE = "dataset/UBFC-Phys-S1/vid_s1_T2.avi"       
    GT_FILE = "dataset/UBFC-Phys-S1/bvp_s1_T2.csv"    
    # ==========================================

    detector = FaceDetector()
    gt_reader = GroundTruthReader(GT_FILE) if GT_FILE else None
    is_live = isinstance(VIDEO_SOURCE, int)
    
    last_calculated_bpm = None
    last_freqs = None
    last_filt_mag = None
    
    with WebcamStream(source=VIDEO_SOURCE) as cam:
        frame_counter = 0
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)

        while True:
            success, frame = cam.read_frame()
            if not success: 
                break

            frame_counter += 1
            timestamp = time.time() if is_live else (frame_counter / cam.fps)

            # Draw ROIs equally since there is no AI attention weight
            current_weights = processor.get_latest_weights()
            multi_rois = detector.get_face_mesh_rois(frame, draw=True, ai_weights=current_weights)

            if multi_rois:
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)

            if frame_counter % 15 == 0:
                bpm, freqs, filt_mag = processor.estimate_heart_rate()
                if bpm is not None:
                    last_calculated_bpm = bpm
                    last_freqs = freqs
                    last_filt_mag = filt_mag

            # ==================================================
            # 3. OPENCV DASHBOARD (No latency)
            # ==================================================
            h, w = frame.shape[:2]
            
            # --- Sleeker, Adjusted Telemetry Box ---
            box_y1, box_y2 = h - 135, h - 10  # Shrunk the height to remove the title
            box_x1, box_x2 = 10, 420
            
            ecg_w, ecg_h = 400, 100
            ecg_x1, ecg_y1 = w - ecg_w - 10, h - ecg_h - 10
            ecg_x2, ecg_y2 = w - 10, h - 10
            
            bar_x1, bar_y1 = w - 40, ecg_y1 - 160
            bar_x2, bar_y2 = w - 10, ecg_y1 - 10

            # --------------------------------------------------
            # LAYER 1: Semi-Transparent Backgrounds
            # --------------------------------------------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1) 
            cv2.rectangle(overlay, (ecg_x1, ecg_y1), (ecg_x2, ecg_y2), (0, 0, 0), -1) 
            cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1) 
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # --------------------------------------------------
            # LAYER 2: 100% Opaque Text and Graphics
            # --------------------------------------------------
            fps = processor.get_current_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if last_calculated_bpm is not None:
                cv2.putText(frame, f"Est BPM: {last_calculated_bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if gt_reader is not None:
                gt_hr = gt_reader.get_hr_at_time(timestamp)
                if gt_hr is not None:
                    cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- B. Hierarchical Math Telemetry Grid (Shifted Up) ---
            sub_a, reg_a, global_a = processor.get_alpha_telemetry()
            
            # Forehead Row
            fh_str = f"FH : [{sub_a['forehead_1']:.2f}, {sub_a['forehead_2']:.2f}, {sub_a['forehead_3']:.2f}] -> {reg_a['forehead']:.2f}"
            cv2.putText(frame, fh_str, (box_x1 + 10, box_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Left Cheek Row
            lc_str = f"LC : [{sub_a['left_cheek_1']:.2f}, {sub_a['left_cheek_2']:.2f}, {sub_a['left_cheek_3']:.2f}] -> {reg_a['left_cheek']:.2f}"
            cv2.putText(frame, lc_str, (box_x1 + 10, box_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Right Cheek Row
            rc_str = f"RC : [{sub_a['right_cheek_1']:.2f}, {sub_a['right_cheek_2']:.2f}, {sub_a['right_cheek_3']:.2f}] -> {reg_a['right_cheek']:.2f}"
            cv2.putText(frame, rc_str, (box_x1 + 10, box_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Global Alpha
            cv2.putText(frame, f"GLOBAL ALPHA : {global_a:.3f}", (box_x1 + 10, box_y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Backend
            backend_name = processor.get_backend_name()
            cv2.putText(frame, f"[{backend_name}]", (box_x1 + 10, box_y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)

            # --- C. The ECG Waveform ---
            filtered_signal = processor.get_filtered_signal()
            if filtered_signal is not None and len(filtered_signal) > 30:
                display_pts = int(cam.fps * 3)
                wave_slice = filtered_signal[-display_pts:]
                min_val, max_val = np.min(wave_slice), np.max(wave_slice)
                normalized_wave = (wave_slice - min_val) / (max_val - min_val + 1e-8)
                pts_y = ecg_y2 - (normalized_wave * ecg_h * 0.8) - (ecg_h * 0.1) 
                pts_x = np.linspace(ecg_x1, ecg_x2, len(pts_y))
                pts = np.vstack((pts_x, pts_y)).T.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # --- D. The SNR Confidence Bar ---
            if last_filt_mag is not None:
                snr = np.max(last_filt_mag) / (np.mean(last_filt_mag) + 1e-8)
                fill_pct = np.clip((snr - 2.0) / 4.0, 0.0, 1.0)
                fill_h = int(fill_pct * (bar_y2 - bar_y1))
                if fill_pct < 0.4:
                    bar_color = (0, 0, 255)
                elif fill_pct < 0.7:
                    bar_color = (0, 255, 255)
                else:
                    bar_color = (0, 255, 0)
                cv2.rectangle(frame, (bar_x1, bar_y2 - fill_h), (bar_x2, bar_y2), bar_color, -1)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
                cv2.putText(frame, "SNR", (bar_x1 - 5, bar_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Math rPPG Dashboard", frame)

            delay = int(1000 / cam.fps) if isinstance(VIDEO_SOURCE, str) else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break


if __name__ == "__main__":
    """
    Entry point for the rPPG benchmarking script. Runs the main loop that processes the video,
    collects BPM estimates and ground truth, and calculates evaluation metrics.
    """
    main()