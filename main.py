"""
Main application for rPPG signal extraction.
---------------------------------------
Integrates webcam streaming, ML face meshing, and signal processing.
Displays a 3-panel live plot showing Time and Frequency domains.
"""

import logging
import cv2
import matplotlib.pyplot as plt
import time
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

        line1, = ax1.plot([], [], 'g-')
        ax1.set_title("1. Raw Signal (Green Channel Average)")
        ax1.set_ylabel("Amplitude")

        line2, = ax2.plot([], [], 'b-')
        ax2.set_title("2. Filtered Signal (Time Domain)")
        ax2.set_ylabel("Amplitude")

        line3, = ax3.plot([], [], 'm-') 
        ax3.set_title("3. Filtered FFT (Power Spectrum)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")

    VIDEO_SOURCE = 0 # or "dataset/subject1.mp4"
    GT_FILE = "dataset/gt_subject1.txt" 
    
    gt_reader = GroundTruthReader(GT_FILE)
    is_live = isinstance(VIDEO_SOURCE, int)
    
    last_bpm, last_freqs, last_mag = None, None, None

    with WebcamStream(source=VIDEO_SOURCE) as cam:
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)
        frame_counter = 0

        while True:
            success, frame = cam.read_frame()
            if not success:
                logging.warning("OpenCV FFMPEG reached End-of-File or dropped the stream early.")
                break
            frame_counter += 1

            if is_live:
                timestamp = time.time() 
            else:
                if frame_counter - 1 < len(gt_reader.timestamps):
                    timestamp = float(gt_reader.timestamps[frame_counter - 1])
                else:
                    timestamp = frame_counter / cam.fps 

            rois = detector.get_face_mesh_rois(frame)

            if rois:
                processor.extract_and_buffer_multi(frame, rois, timestamp)
                for name, polygon in rois.items():
                    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # ==================================================
                # DSP UPGRADE: Decoupled Heavy Math (Every 15 frames)
                # ==================================================
                if frame_counter % 15 == 0:
                    bpm, freqs, filt_mag = processor.estimate_heart_rate()
                    if bpm is not None:
                        last_bpm, last_freqs, last_mag = bpm, freqs, filt_mag

                fps = processor.get_current_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if last_bpm is not None:
                    cv2.putText(frame, f"Est BPM: {last_bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                gt_hr = gt_reader.get_hr_at_time(timestamp)
                if gt_hr is not None:
                    cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ==================================================
                # Throttled UI rendering
                # ==================================================
                if DEBUG_MODE and (frame_counter % 15 == 0):
                    raw_g = list(processor.raw_g)
                    filtered_data = processor.get_filtered_signal()

                    if len(raw_g) > 10:
                        line1.set_xdata(range(len(raw_g)))
                        line1.set_ydata(raw_g)
                        ax1.relim()
                        ax1.autoscale_view()

                    if filtered_data is not None:
                        line2.set_xdata(range(len(filtered_data)))
                        line2.set_ydata(filtered_data)
                        ax2.relim()
                        ax2.autoscale_view()

                    if last_bpm is not None:
                        line3.set_xdata(last_freqs)
                        line3.set_ydata(last_mag)
                        ax3.relim()
                        ax3.autoscale_view()

                    fig.canvas.draw()
                    fig.canvas.flush_events()

            cv2.imshow("rPPG - Live Feed", frame)

            delay = int(1000 / cam.fps) if isinstance(VIDEO_SOURCE, str) else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
    if DEBUG_MODE:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()