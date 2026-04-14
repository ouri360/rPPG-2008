"""
Main application for rPPG signal extraction.
---------------------------------------
This script integrates the webcam streaming, face detection, 
and signal processing modules to capture video frames, 
detect the face, extract the rPPG signal, and display it in real-time. 
It also includes a live plot of the raw signal values for visualization.

Press 'q' to quit the application safely.
"""

import cv2
import logging
import matplotlib.pyplot as plt
from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor

DEBUG_MODE = True

def main():
    detector = FaceDetector()

    if DEBUG_MODE:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        fig.tight_layout(pad=4.0)

        line1, = ax1.plot([], [], 'g-')
        ax1.set_title("1. Raw Signal (Weighted Multi-ROI)")
        ax1.set_ylabel("Amplitude")

        line2, = ax2.plot([], [], 'b-')
        ax2.set_title("2. Filtered Signal (Bandpass 0.7 - 3.0 Hz)")
        ax2.set_ylabel("Amplitude")

        line3, = ax3.plot([], [], 'r-')
        ax3.set_title("3. FFT (Frequency Domain)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude")

    # ==========================================
    # VIDEO SOURCE SELECTION
    # ==========================================
    #VIDEO_SOURCE = "dataset/subject1.mp4" # Or set to 0 for webcam
    VIDEO_SOURCE = 0


    with WebcamStream(source=VIDEO_SOURCE) as cam:
        
        processor = SignalProcessor(buffer_seconds=10, target_fps=cam.fps)
        logging.info("Démarrage de la boucle de traitement rPPG Multi-ROI...")
        
        frame_counter = 0

        while True:
            success, frame = cam.read_frame()
            if not success: 
                break

            frame_counter += 1

            # 1. Detect the main face
            face_box = detector.detect_largest_face(frame)

            if face_box:
                # 2. Extract the dictionary of 3 sub-regions
                multi_rois = detector.get_multi_rois(face_box)
                
                # 3. Process the weighted average
                processor.extract_and_buffer_multi(frame, multi_rois)
                
                # 4. Draw the 3 sub-regions on the frame
                for region_name, box in multi_rois.items():
                    rx, ry, rw, rh = box
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

                # Anchor the text to the top of the main face box
                fx, fy, fw, fh = face_box

                bpm, freqs, magnitude = processor.estimate_heart_rate()
                
                if bpm is not None:
                    text = f"BPM: {bpm:.1f}"
                    cv2.putText(frame, text, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Calcul BPM...", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if DEBUG_MODE and (frame_counter % 3 == 0):
                    signal_data, _ = processor.get_signal_data()
                    filtered_data = processor.get_filtered_signal()

                    # Update Line 1
                    if len(signal_data) > 10:
                        line1.set_xdata(range(len(signal_data)))
                        line1.set_ydata(signal_data)
                        ax1.relim()
                        ax1.autoscale_view()

                    # Update Line 2
                    if filtered_data is not None:
                        line2.set_xdata(range(len(filtered_data)))
                        line2.set_ydata(filtered_data)
                        ax2.relim()
                        ax2.autoscale_view()

                    # Update Graph 3
                    if bpm is not None:
                        line3.set_xdata(freqs)
                        line3.set_ydata(magnitude)
                        ax3.relim()
                        ax3.autoscale_view()
                        
                        ax3.relim()
                        ax3.autoscale_view()

                    # Execute the render
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