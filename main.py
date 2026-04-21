"""
Main application for rPPG signal extraction.
---------------------------------------
Integrates webcam streaming, ML face meshing, and signal processing.
Displays a 3-panel live plot showing Time and Frequency domains,
and overlays real-time BPM estimates and ground truth on the video feed.
"""

import cv2
import logging
import matplotlib.pyplot as plt
from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader
import time

DEBUG_MODE = True

def main():
    detector = FaceDetector()

    if DEBUG_MODE:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        fig.tight_layout(pad=4.0)

        # 1. Raw Time Signal
        line1, = ax1.plot([], [], 'g-')
        ax1.set_title("1. Raw Signal (Green Channel Average)")
        ax1.set_ylabel("Amplitude")

        # 2. Filtered Time Signal
        line2, = ax2.plot([], [], 'b-')
        ax2.set_title("2. Filtered Signal (Time Domain)")
        ax2.set_ylabel("Amplitude")

        # 3. Filtered FFT (After Filter)
        line3, = ax3.plot([], [], 'm-') 
        ax3.set_title("3. Filtered FFT (Power Spectrum)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")

    # Make sure this points to the right video and GT file
    VIDEO_SOURCE = "dataset/vid_subject8.avi" # or 0 for live webcam
    GT_FILE = "dataset/gt_subject8.xmp" 
    
    gt_reader = GroundTruthReader(GT_FILE)
    is_live = isinstance(VIDEO_SOURCE, int)
    
    with WebcamStream(source=VIDEO_SOURCE) as cam:
        frame_counter = 0
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)

        while True:
            success, frame = cam.read_frame()
            if not success: 
                break

            frame_counter += 1
            
            if is_live:
                timestamp = time.time() 
            else:
                # Use the exact time the camera physically took the picture
                if frame_counter - 1 < len(gt_reader.timestamps):
                    timestamp = float(gt_reader.timestamps[frame_counter - 1])
                else:
                    timestamp = frame_counter / cam.fps # Fallback if GT ends

            # ==================================================
            # 1. ML FACE MESHING & SIGNAL EXTRACTION
            # ==================================================
            rois = detector.get_face_mesh_rois(frame)

            if rois:
                # Send the polygons to the pixel sorter
                processor.extract_and_buffer_multi(frame, rois, timestamp)
                
                # Draw the dynamic ML polygons on the live feed
                for name, polygon in rois.items():
                    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # ==================================================
            # 2. HEADS UP DISPLAY (Static Screen Coordinates)
            # ==================================================
            # Draw FPS at the very top
            fps = processor.get_current_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calculate and Draw Estimated BPM (Red)
            bpm, freqs, filt_mag = processor.estimate_heart_rate()
            if bpm is not None:
                cv2.putText(frame, f"Est BPM: {bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Extract and Draw Ground Truth (Green)
            gt_hr = gt_reader.get_hr_at_time(timestamp)
            if gt_hr is not None:
                cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


            # ==================================================
            # 3. MATPLOTLIB DASHBOARD
            # ==================================================
            if DEBUG_MODE and (frame_counter % 3 == 0):
                signal_data = list(processor.raw_signal)
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

                if bpm is not None:
                    line3.set_xdata(freqs)
                    line3.set_ydata(filt_mag)
                    ax3.relim()
                    ax3.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()

            cv2.imshow("rPPG - Live Feed", frame)

            delay = int(1000 / cam.fps) if isinstance(VIDEO_SOURCE, str) else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break
                
    if DEBUG_MODE:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()