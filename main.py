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

DEBUG_MODE = False

def main():
    detector = FaceDetector()

    if DEBUG_MODE:
        plt.ion()
        # Increased figsize to 12 tall to fit 4 graphs comfortably
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))
        fig.tight_layout(pad=4.0)

        # 1. Raw Time Signal
        line1, = ax1.plot([], [], 'g-')
        ax1.set_title("1. Raw Signal (Time Domain)")
        ax1.set_ylabel("Amplitude")

        # 2. Filtered Time Signal
        line2, = ax2.plot([], [], 'b-')
        ax2.set_title("2. Filtered Signal (Time Domain)")
        ax2.set_ylabel("Amplitude")

        # 3. Raw FFT (Before Filter)
        line3, = ax3.plot([], [], 'r-')
        ax3.set_title("3. Raw FFT (Before Filter)")
        ax3.set_ylabel("Power")

        # 4. Filtered FFT (After Filter)
        line4, = ax4.plot([], [], 'm-') # 'm-' is magenta line
        ax4.set_title("4. Filtered FFT (After Filter)")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Power")

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

            # === BUG FIX 1: Retrieve actual video/hardware timestamps ===
            # Try to get the hardware timestamp in seconds
            timestamp = cam.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Fallback: If live webcam returns 0.0, calculate perfect theoretical time
            if timestamp <= 0.0:
                timestamp = frame_counter / cam.fps
            # ============================================================

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

                # Unpack the 4 variables from the new method
                bpm, freqs, raw_mag, filt_mag = processor.estimate_heart_rate()
                
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
                    signal_data, _ = processor.get_signal_data()
                    filtered_data = processor.get_filtered_signal()

                    if len(signal_data) > 10:
                        line1.set_xdata(range(len(signal_data)))
                        line1.set_ydata(signal_data)
                        ax1.relim()
                        ax1.autoscale_view()

                    if filtered_data is not None:
                        line2.set_xdata(range(len(filtered_data)))
                        line2.set_ydata(filtered_data)
                        ax2.relim()
                        ax2.autoscale_view()

                    # Update both FFT graphs
                    if bpm is not None:
                        # Raw FFT
                        line3.set_xdata(freqs)
                        line3.set_ydata(raw_mag)
                        ax3.relim()
                        ax3.autoscale_view()
                        
                        # Filtered FFT
                        line4.set_xdata(freqs)
                        line4.set_ydata(filt_mag)
                        ax4.relim()
                        ax4.autoscale_view()

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