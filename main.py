"""
Main application for rPPG signal extraction.
---------------------------------------
Integrates webcam streaming, face detection, and signal processing.
Displays a 4-panel live plot showing Time and Frequency domains before and after filtering.
"""

import cv2
import logging
import matplotlib.pyplot as plt
from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader

DEBUG_MODE = True

def main():
    detector = FaceDetector()

    if DEBUG_MODE:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        fig.tight_layout(pad=4.0)

        # 1. Raw RGB Time Signals
        line_r, = ax1.plot([], [], 'r-', label='Red', alpha=0.7)
        line_g, = ax1.plot([], [], 'g-', label='Green', alpha=0.7)
        line_b, = ax1.plot([], [], 'b-', label='Blue', alpha=0.7)
        ax1.set_title("1. Raw RGB Signals")
        ax1.legend(loc='upper right')

        # 2. ICA Components
        line_c0, = ax2.plot([], [], 'c-', label='Comp 1', alpha=0.5)
        line_c1, = ax2.plot([], [], 'm-', label='Comp 2', alpha=0.5)
        line_c2, = ax2.plot([], [], 'y-', label='Comp 3', alpha=0.5)
        line_best, = ax2.plot([], [], 'k-', label='Selected Heartbeat', linewidth=2)
        ax2.set_title("2. Unmixed ICA Components")
        ax2.legend(loc='upper right')

        # 3. Filtered FFT (After Filter)
        line_fft, = ax3.plot([], [], 'm-') 
        ax3.set_title("3. Final FFT (Selected Component)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")

    VIDEO_SOURCE = "dataset/subject1.mp4" # Or set to "dataset/subject1.mp4"
    GT_FILE = "dataset/gt_subject1.txt"
    # Initialize the Ground Truth Reader
    gt_reader = GroundTruthReader(GT_FILE)

    with WebcamStream(source=VIDEO_SOURCE) as cam:
        
        processor = SignalProcessor(buffer_seconds=10, target_fps=cam.fps)
        logging.info("Démarrage de la boucle de traitement rPPG Multi-ROI...")
        
        frame_counter = 0

        while True:
            success, frame = cam.read_frame()

            if not success: 
                break

            frame_counter += 1

            # ==================================================
            # Completely prevents "Clock Beating" and 1Hz interpolation harmonics
            # ==================================================
            timestamp = frame_counter / cam.fps

            face_box = detector.detect_largest_face(frame)

            if face_box:
                multi_rois = detector.get_multi_rois(face_box)
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)
                
                for region_name, box in multi_rois.items():
                    rx, ry, rw, rh = box
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

                fx, fy, fw, fh = face_box

                # === NEW: Real FPS display on the OpenCV window ===
                current_fps = processor.get_current_fps()
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(frame, fps_text, (fx, fy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # ==================================================    

                # Unpack all 5 variables
                bpm, freqs, filt_mag, best_comp, all_comps = processor.estimate_heart_rate()
                
                if bpm is not None:
                    text = f"BPM: {bpm:.1f}"
                    cv2.putText(frame, text, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Calcul BPM...", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # ==================================================
                # Compare with UBFC Ground Truth
                # ==================================================
                gt_hr = gt_reader.get_hr_at_time(timestamp)
                if gt_hr is not None:
                    # Draw the Ground truth in Green, slightly below the bounding box
                    gt_text = f"True HR: {gt_hr:.1f}"
                    cv2.putText(frame, gt_text, (fx, fy + fh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # ==================================================

                # GUI Downsampling (Draw every 3 frames)
                if DEBUG_MODE and (frame_counter % 3 == 0):
                    # Fetch raw RGB lists
                    raw_r = list(processor.raw_r)
                    raw_g = list(processor.raw_g)
                    raw_b = list(processor.raw_b)

                    # Update Graph 1: RGB
                    if len(raw_g) > 10:
                        x_data = range(len(raw_g))
                        line_r.set_data(x_data, raw_r)
                        line_g.set_data(x_data, raw_g)
                        line_b.set_data(x_data, raw_b)
                        ax1.relim()
                        ax1.autoscale_view()

                    # Update Graph 2 & 3: ICA and FFT
                    if best_comp is not None:
                        x_data_comp = range(len(best_comp))
                        
                        # Plot the 3 background components
                        line_c0.set_data(x_data_comp, all_comps[0])
                        line_c1.set_data(x_data_comp, all_comps[1])
                        line_c2.set_data(x_data_comp, all_comps[2])
                        
                        # Plot the selected heartbeat thick and black over top
                        line_best.set_data(x_data_comp, best_comp)
                        ax2.relim()
                        ax2.autoscale_view()

                        # Update the FFT graph
                        line_fft.set_data(freqs, filt_mag)
                        ax3.relim()
                        ax3.autoscale_view()

                    fig.canvas.draw()
                    fig.canvas.flush_events()

            cv2.imshow("rPPG - Live Feed", frame)

            delay = int(1000 / cam.fps) if isinstance(VIDEO_SOURCE, str) else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logging.info("Fermeture demandée par l'utilisateur.")
                break
                
    if DEBUG_MODE:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()