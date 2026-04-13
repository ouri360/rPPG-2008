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

# Set to False when deploying, True enables the 3-panel matplotlib dashboard for debugging and visualization
DEBUG_MODE = False

def main():
    detector = FaceDetector()
    processor = SignalProcessor(buffer_seconds=10, target_fps=30)

    if DEBUG_MODE:
        # ==========================================
        # 3-PANEL MATPLOTLIB DASHBOARD
        # ==========================================
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        fig.tight_layout(pad=4.0) # Adds spacing between graphs

        # 1. Raw Signal
        line1, = ax1.plot([], [], 'g-')
        ax1.set_title("1. Raw Signal (Time Domain)")
        ax1.set_ylabel("Amplitude")

        # 2. Filtered Signal
        line2, = ax2.plot([], [], 'b-')
        ax2.set_title("2. Filtered Signal (Bandpass 0.7 - 3.0 Hz)")
        ax2.set_ylabel("Amplitude")

        # 3. Frequency Spectrum (Welch)
        line3, = ax3.plot([], [], 'r-')
        ax3.set_title("3. Welch's PSD (Frequency Domain)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")
        # ==========================================

    with WebcamStream(camera_index=0) as cam:
        logging.info("Starting rPPG processing loop...")
        
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            face_box = detector.detect_largest_face(frame)

            if face_box:
                # Get the 3 regions of interests (ROIs)
                multi_rois = detector.get_multi_rois(face_box)
                
                # Extract using the weighted method
                processor.extract_and_buffer_multi(frame, multi_rois)
                
                # Draw the bounding boxes for visualization
                for name, box in multi_rois.items():
                    rx, ry, rw, rh = box
                    # Draw Forehead in Green, Cheeks in Blue to distinguish
                    color = (0, 255, 0) if name == 'forehead' else (255, 0, 0)
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

                # Extract specifically the forehead coordinates
                fx, fy, fw, fh = multi_rois['forehead']

                # Get the tuple from the estimate_heart_rate method
                bpm, freqs, psd = processor.estimate_heart_rate()

                if bpm is not None:
                    text = f"BPM: {bpm:.1f}"
                    cv2.putText(frame, text, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if DEBUG_MODE:
                        # Update Plot 3: Frequency Spectrum
                        line3.set_xdata(freqs)
                        line3.set_ydata(psd)
                        ax3.relim()
                        ax3.autoscale_view()
                else:
                    cv2.putText(frame, "Calcul BPM...", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if DEBUG_MODE:
                    # Update Plots 1 & 2 every frame
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

                    # Draw the updated graphs
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            cv2.imshow("rPPG - Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break
                
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()