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
import threading

from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader


def main():
    """Main function to run the rPPG benchmark on a specified video and ground truth."""

    # ==========================================
    # USER CONFIGURATION
    # ==========================================
    VIDEO_SOURCE = 0
    GT_FILE = None   
    # ==========================================

    detector = FaceDetector()
    gt_reader = GroundTruthReader(GT_FILE) if GT_FILE else None
    is_live = isinstance(VIDEO_SOURCE, int)
    
    last_calculated_bpm = None
    last_freqs = None
    last_filt_mag = None
    last_filtered_signal = None 
    last_ecg_normalized = None 
    last_snr = None
    is_calculating = False
    
    with WebcamStream(source=VIDEO_SOURCE) as cam:
        frame_counter = 0
        processor = SignalProcessor(buffer_seconds=10, target_fps=cam.fps)

        while True:
            success, frame = cam.read_frame()
            if not success: 
                break

            frame_counter += 1
            timestamp = time.time() if is_live else (frame_counter / cam.fps)

            # ==========================================
            DISPLAY_RATE = 3 # Afficher 1 frame sur 3
            should_draw = (frame_counter % DISPLAY_RATE == 0)

            current_weights = processor.get_latest_weights()
            
            # On passe should_draw au détecteur. Si False, le CPU ne calcule pas les couleurs Inferno ni les overlays
            multi_rois = detector.get_face_mesh_rois(frame, draw=should_draw, ai_weights=current_weights)

            if multi_rois:
                # L'extraction mathématique continue à chaque frame pour ne pas casser le signal
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)

            if frame_counter % 5 == 0 and not is_calculating:
                def math_worker():
                    # ATTENTION : Il faut bien ajouter last_ecg_normalized et last_snr ici !
                    nonlocal last_calculated_bpm, last_freqs, last_filt_mag, last_filtered_signal, last_ecg_normalized, last_snr, is_calculating
                    is_calculating = True 
                    
                    try:
                        # 1. On calcule le signal mathématique UNE SEULE FOIS
                        sig = processor.get_filtered_signal()
                        
                        if sig is not None and len(sig) > 30:
                            last_filtered_signal = sig
                            
                            # On normalise l'ECG pour qu'OpenCV puisse le dessiner
                            display_pts = int(cam.fps * 3)
                            wave_slice = sig[-display_pts:]
                            min_val, max_val = np.min(wave_slice), np.max(wave_slice)
                            last_ecg_normalized = (wave_slice - min_val) / (max_val - min_val + 1e-8)
                            # ---------------------------------------
                            
                            # 2. On passe 'sig' directement pour économiser le CPU
                            bpm, freqs, filt_mag = processor.estimate_heart_rate(filtered_signal=sig)
                            
                            if bpm is not None:
                                last_calculated_bpm = bpm
                                last_freqs = freqs
                                last_filt_mag = filt_mag
                                
                                # --- LE CALCUL DU SNR ---
                                last_snr = np.max(filt_mag) / (np.mean(filt_mag) + 1e-8)
                                # -----------------------------------------
                    finally:
                        is_calculating = False

                # Lancer la fonction sans bloquer la vidéo
                threading.Thread(target=math_worker, daemon=True).start()

            # ==================================================
            # 3. PURE OPENCV DASHBOARD (Exécuté seulement si should_draw est True)
            # ==================================================
            if should_draw:
                h, w = frame.shape[:2]
                
                # --- Expanded Telemetry Box ---
                box_y1, box_y2 = h - 155, h - 10  
                box_x1, box_x2 = 10, 480 
                
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

                # --- B. Hierarchical Math Telemetry Grid (With Weights!) ---
                sub_a, reg_a, global_a = processor.get_alpha_telemetry()
                
                w_fh = (current_weights['forehead_1'] + current_weights['forehead_2'] + current_weights['forehead_3']) * 100
                w_lc = (current_weights['left_cheek_1'] + current_weights['left_cheek_2'] + current_weights['left_cheek_3']) * 100
                w_rc = (current_weights['right_cheek_1'] + current_weights['right_cheek_2'] + current_weights['right_cheek_3']) * 100
                w_lp = current_weights['lips'] * 100
                
                fh_str = f"FH : [{sub_a['forehead_1']:.2f}, {sub_a['forehead_2']:.2f}, {sub_a['forehead_3']:.2f}] -> {reg_a['forehead']:.2f}  [W: {w_fh:.1f}%]"
                cv2.putText(frame, fh_str, (box_x1 + 10, box_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                lc_str = f"LC : [{sub_a['left_cheek_1']:.2f}, {sub_a['left_cheek_2']:.2f}, {sub_a['left_cheek_3']:.2f}] -> {reg_a['left_cheek']:.2f}  [W: {w_lc:.1f}%]"
                cv2.putText(frame, lc_str, (box_x1 + 10, box_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                rc_str = f"RC : [{sub_a['right_cheek_1']:.2f}, {sub_a['right_cheek_2']:.2f}, {sub_a['right_cheek_3']:.2f}] -> {reg_a['right_cheek']:.2f}  [W: {w_rc:.1f}%]"
                cv2.putText(frame, rc_str, (box_x1 + 10, box_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                lips_str = f"LP : [{sub_a['lips']:.2f}] -> {reg_a['lips']:.2f}  [W: {w_lp:.1f}%]"
                cv2.putText(frame, lips_str, (box_x1 + 10, box_y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                cv2.putText(frame, f"GLOBAL ALPHA : {global_a:.3f}", (box_x1 + 10, box_y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                backend_name = processor.get_backend_name()
                cv2.putText(frame, f"[{backend_name}]", (box_x1 + 10, box_y1 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)

                # --- C. The ECG Waveform ---
                if last_ecg_normalized is not None:
                    # On utilise directement l'onde en cache
                    pts_y = ecg_y2 - (last_ecg_normalized * ecg_h * 0.8) - (ecg_h * 0.1) 
                    pts_x = np.linspace(ecg_x1, ecg_x2, len(pts_y))
                    pts = np.vstack((pts_x, pts_y)).T.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

                # --- D. The SNR Confidence Bar ---
                if last_snr is not None:
                    fill_pct = np.clip((last_snr - 2.0) / 4.0, 0.0, 1.0)
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

            # Optimisation mineure du délai d'attente
            delay = 1 if is_live else int(1000 / cam.fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break


if __name__ == "__main__":
    """
    Entry point for the rPPG benchmarking script. Runs the main loop that processes the video,
    collects BPM estimates and ground truth, and calculates evaluation metrics.
    """
    main()