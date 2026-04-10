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

def main():
    detector = FaceDetector()
    processor = SignalProcessor(buffer_seconds=10, target_fps=30)

    # Set up matplotlib for live plotting
    plt.ion() # Interactive mode on
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'g-')
    ax.set_title("Raw rPPG Signal (Green Channel Average)")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Amplitude")

    with WebcamStream(camera_index=0) as cam:
        logging.info("Starting rPPG processing loop...")
        
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            face_box = detector.detect_largest_face(frame)

            if face_box:
                roi_box = detector.get_rppg_roi(face_box)
                
                # Extract the signal!
                current_value = processor.extract_and_buffer(frame, roi_box)  # noqa: F841
                
                # Draw the bounding box
                rx, ry, rw, rh = roi_box
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

                # Estimate the heart rate and display it on the frame
                bpm = processor.estimate_heart_rate()
                if bpm is not None:
                    # If the buffer is full enough to estimate BPM, display it in red
                    text = f"BPM: {bpm:.1f}"
                    cv2.putText(frame, text, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # The buffer is not full enough to estimate BPM, display a warning in yellow
                    cv2.putText(frame, "Calcul BPM...", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Update the live plot dynamically
                signal_data, _ = processor.get_signal_data()
                
                if len(signal_data) > 10: # Wait for a few frames before plotting
                    line.set_xdata(range(len(signal_data)))
                    line.set_ydata(signal_data)
                    ax.relim()
                    ax.autoscale_view()
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