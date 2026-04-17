""""
Main application for real-time rPPG heart rate estimation using webcam or video file input.
---------------------------------------
Integrates face detection, signal processing, and ground truth comparison with a decoupled UI for visualization.
The application captures video frames, detects the face and relevant regions, extracts the raw RGB signals,
applies ICA for signal separation, performs frequency analysis to estimate heart rate, 
and compares it against ground truth data if available.
"""

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

        line_r, = ax1.plot([], [], 'r-', label='Red', alpha=0.7)
        line_g, = ax1.plot([], [], 'g-', label='Green', alpha=0.7)
        line_b, = ax1.plot([], [], 'b-', label='Blue', alpha=0.7)
        ax1.set_title("1. Raw RGB Signals")
        ax1.legend(loc='upper right')

        line_c0, = ax2.plot([], [], 'c-', label='Comp 1', alpha=0.5)
        line_c1, = ax2.plot([], [], 'm-', label='Comp 2', alpha=0.5)
        line_c2, = ax2.plot([], [], 'y-', label='Comp 3', alpha=0.5)
        line_best, = ax2.plot([], [], 'k-', label='Selected Heartbeat', linewidth=2)
        ax2.set_title("2. Unmixed ICA Components")
        ax2.legend(loc='upper right')
        ax2.set_ylim(-2, 2)

        line_fft, = ax3.plot([], [], 'm-') 
        ax3.set_title("3. Final FFT (Selected Component)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")

    VIDEO_SOURCE = 0 # Or set to "dataset/subject1.mp4"
    GT_FILE = "dataset/gt_subject1.txt"
    gt_reader = GroundTruthReader(GT_FILE)
    
    is_live = isinstance(VIDEO_SOURCE, int)

    # Caching variables to prevent UI flicker
    last_bpm, last_freqs, last_mag, last_comp, last_all = None, None, None, None, None

    with WebcamStream(source=VIDEO_SOURCE) as cam:
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)
        frame_counter = 0

        while True:
            success, frame = cam.read_frame()
            if not success: break
            frame_counter += 1

            # Context-Aware Timeline
            if is_live:
                timestamp = time.time() 
            else:
                if frame_counter - 1 < len(gt_reader.timestamps):
                    timestamp = float(gt_reader.timestamps[frame_counter - 1])
                else:
                    timestamp = frame_counter / cam.fps 

            # Spatial Extraction (Runs every frame)
            multi_rois = detector.get_multi_rois(frame)

            if multi_rois:
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)
                
                # Draw Polygons (MediaPipe compatibility)
                for region_name, polygon in multi_rois.items():
                    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # ==========================================
                # Decoupled ICA Processing (Runs every 15 frames)
                # ==========================================
                if frame_counter % 15 == 0:
                    bpm, freqs, filt_mag, best_comp, all_comps = processor.estimate_heart_rate()
                    if bpm is not None:
                        last_bpm, last_freqs, last_mag = bpm, freqs, filt_mag
                        last_comp, last_all = best_comp, all_comps

                current_fps = processor.get_current_fps()
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if last_bpm is not None:
                    cv2.putText(frame, f"BPM: {last_bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Calcul BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                gt_hr = gt_reader.get_hr_at_time(timestamp)
                if gt_hr is not None:
                    cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Decoupled UI Rendering
                if DEBUG_MODE and (frame_counter % 15 == 0):
                    raw_r, raw_g, raw_b = list(processor.raw_r), list(processor.raw_g), list(processor.raw_b)

                    if len(raw_g) > 10:
                        x_data = range(len(raw_g))
                        line_r.set_data(x_data, raw_r)
                        line_g.set_data(x_data, raw_g)
                        line_b.set_data(x_data, raw_b)
                        ax1.relim()
                        ax1.autoscale_view()

                    if last_comp is not None:
                        x_data_comp = range(len(last_comp))
                        line_c0.set_data(x_data_comp, last_all[0])
                        line_c1.set_data(x_data_comp, last_all[1])
                        line_c2.set_data(x_data_comp, last_all[2])
                        line_best.set_data(x_data_comp, last_comp)
                        ax2.relim()
                        ax2.autoscale_view()

                        line_fft.set_data(last_freqs, last_mag)
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