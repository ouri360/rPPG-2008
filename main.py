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
    detector = FaceDetector()

    VIDEO_SOURCE = "dataset/UBFC-rPPG-Set2-Realistic/vid_subject4.avi"
    GT_FILE = "dataset/UBFC-rPPG-Set2-Realistic/gt_subject4.txt"
    
    gt_reader = GroundTruthReader(GT_FILE)
    is_live = isinstance(VIDEO_SOURCE, int)
    
    # Caching Variables
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

            # 1. AI Spatial Attention
            current_ai_weights = processor.get_latest_weights()
            multi_rois = detector.get_face_mesh_rois(frame, draw=True, ai_weights=current_ai_weights)

            if multi_rois:
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)

            # 2. Asynchronous Heavy Math (Runs every 15 frames)
            if frame_counter % 15 == 0:
                bpm, freqs, filt_mag = processor.estimate_heart_rate()
                if bpm is not None:
                    last_calculated_bpm = bpm
                    last_freqs = freqs
                    last_filt_mag = filt_mag

            # ==================================================
            # 3. PURE OPENCV ZERO-LATENCY DASHBOARD
            # ==================================================
            h, w = frame.shape[:2]
            overlay = frame.copy()

            # --- A. Basic Text Info ---
            fps = processor.get_current_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if last_calculated_bpm is not None:
                cv2.putText(frame, f"Est BPM: {last_calculated_bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            gt_hr = gt_reader.get_hr_at_time(timestamp)
            if gt_hr is not None:
                cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- B. The Engine Telemetry Box (Bottom Left) ---
            box_y1, box_y2 = h - 90, h - 10
            box_x1, box_x2 = 10, 235
            
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            math_alpha, ai_alpha = processor.get_alpha_telemetry()
            
            cv2.putText(overlay, "POS ENGINE TELEMETRY", (box_x1 + 10, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, f"Math Alpha : {math_alpha:.3f}", (box_x1 + 10, box_y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"AI Alpha   : {ai_alpha:.3f}", (box_x1 + 10, box_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- C. The ECG Waveform (Bottom Right) ---
            ecg_w, ecg_h = 400, 100
            ecg_x1, ecg_y1 = w - ecg_w - 10, h - ecg_h - 10
            ecg_x2, ecg_y2 = w - 10, h - 10
            
            cv2.rectangle(overlay, (ecg_x1, ecg_y1), (ecg_x2, ecg_y2), (0, 0, 0), -1)
            
            filtered_signal = processor.get_filtered_signal()
            if filtered_signal is not None and len(filtered_signal) > 30:
                # Grab the last ~3 seconds of the wave for display
                display_pts = int(cam.fps * 3)
                wave_slice = filtered_signal[-display_pts:]
                
                # NumPy Broadcasting to map the wave to pixel coordinates instantly
                min_val, max_val = np.min(wave_slice), np.max(wave_slice)
                normalized_wave = (wave_slice - min_val) / (max_val - min_val + 1e-8)
                
                pts_y = ecg_y2 - (normalized_wave * ecg_h * 0.8) - (ecg_h * 0.1) # 10% padding
                pts_x = np.linspace(ecg_x1, ecg_x2, len(pts_y))
                
                # Format for cv2.polylines
                pts = np.vstack((pts_x, pts_y)).T.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # --- D. The SNR Confidence Bar (Right Edge) ---
            if last_filt_mag is not None:
                bar_x1, bar_y1 = w - 40, ecg_y1 - 160
                bar_x2, bar_y2 = w - 10, ecg_y1 - 10
                
                # Calculate Signal-to-Noise Ratio (Peak vs Average Noise)
                snr = np.max(last_filt_mag) / (np.mean(last_filt_mag) + 1e-8)
                
                # Map SNR (2.0 to 6.0) to a fill percentage (0% to 100%)
                fill_pct = np.clip((snr - 2.0) / 4.0, 0.0, 1.0)
                fill_h = int(fill_pct * (bar_y2 - bar_y1))
                
                # Dynamic Color: Red (Bad) -> Yellow (Okay) -> Green (Good)
                if fill_pct < 0.4:
                    bar_color = (0, 0, 255)
                elif fill_pct < 0.7:
                    bar_color = (0, 255, 255)
                else:
                    bar_color = (0, 255, 0)
                
                # Draw Background, Fill, and Border
                cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
                cv2.rectangle(overlay, (bar_x1, bar_y2 - fill_h), (bar_x2, bar_y2), bar_color, -1)
                cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
                cv2.putText(overlay, "SNR", (bar_x1 - 5, bar_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Apply the transparent blend once for all UI elements
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Show the final frame
            cv2.imshow("Jetson Orin - rPPG Medical Dashboard", frame)

            delay = int(1000 / cam.fps) if isinstance(VIDEO_SOURCE, str) else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break

if __name__ == "__main__":
    main()