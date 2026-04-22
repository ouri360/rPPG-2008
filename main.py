"""
Main application for rPPG signal extraction (Jetson Orin Nano Optimized).
---------------------------------------
Integrates webcam streaming, ML face meshing, and signal processing.
Uses a multi-threaded architecture: 
- Main Thread: PyQtGraph Hardware-Accelerated Dashboard.
- Background Thread: OpenCV, MediaPipe (sub-sampled), and CuPy logic.
"""

import sys
import cv2
import logging
import time
import threading
import queue
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# 1. THE GUI CLASS (Runs on Main Thread)
# ==========================================
class RPPGDashboard(QMainWindow):
    def __init__(self, data_queue):
        super().__init__()
        self.setWindowTitle("rPPG Real-Time Telemetry (Jetson Optimized)")
        self.resize(800, 900)
        self.data_queue = data_queue

        # Create PyQtGraph layout
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graph_widget)
        self.graph_widget.setBackground('k')

        # 1. Raw Signal Plot
        self.plot_raw = self.graph_widget.addPlot(title="1. Raw Signal (Green Channel Average)")
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('g', width=2))
        self.plot_raw.showGrid(x=True, y=True)
        self.graph_widget.nextRow()

        # 2. Filtered Time Signal Plot
        self.plot_filt = self.graph_widget.addPlot(title="2. Filtered Signal (Time Domain)")
        self.curve_filt = self.plot_filt.plot(pen=pg.mkPen('c', width=2)) # Cyan
        self.plot_filt.showGrid(x=True, y=True)
        self.graph_widget.nextRow()

        # 3. Frequency Spectrum (FFT) Plot
        self.plot_fft = self.graph_widget.addPlot(title="3. Filtered FFT (Power Spectrum)")
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen('m', width=2)) # Magenta
        self.plot_fft.showGrid(x=True, y=True)
        self.plot_fft.setLabel('bottom', "Frequency", units='Hz')

        # Timer to poll the queue safely from the GUI thread without freezing
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33) # ~30 FPS GUI update rate

    def update_plots(self):
        try:
            # Pull the latest telemetry payload from the math thread
            data = self.data_queue.get_nowait()
            
            if data.get('raw_signal') is not None and len(data['raw_signal']) > 0:
                self.curve_raw.setData(data['raw_signal'])
            
            if data.get('filtered_signal') is not None and len(data['filtered_signal']) > 0:
                self.curve_filt.setData(data['filtered_signal'])
                
            if data.get('freqs') is not None and data.get('filt_mag') is not None:
                self.curve_fft.setData(data['freqs'], data['filt_mag'])
                
        except queue.Empty:
            pass # No new data, keep displaying the current graphs


# ==========================================
# 2. THE MATH WORKER (Runs on Background Thread)
# ==========================================
def rppg_processing_thread(data_queue):
    """
    Handles OpenCV, Camera I/O, MediaPipe, and Signal Processing.
    Operates entirely independently of the GUI to guarantee high FPS.
    """
    detector = FaceDetector()

    # Focusing on Live Webcam as requested
    VIDEO_SOURCE = "dataset/UBFC-rPPG-Set2-Realistic/vid_subject1.avi"
    # GT_FILE = "dataset/UBFC-rPPG-Set2-Realistic/gt_subject1.txt" 
    # gt_reader = GroundTruthReader(GT_FILE)
    
    with WebcamStream(source=VIDEO_SOURCE) as cam:
        frame_counter = 0
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)

        last_rois = None

        while True:
            success, frame = cam.read_frame()
            if not success: 
                break

            frame_counter += 1
            timestamp = time.time()

            # ==================================================
            # OPTIMIZATION: Run MediaPipe every 5 frames only
            # ==================================================
            if frame_counter % 5 == 0 or last_rois is None:
                rois = detector.get_face_mesh_rois(frame)
                if rois:
                    last_rois = rois
            else:
                rois = last_rois

            # Process the regions of interest
            if rois:
                # Extracts pixels (ideally utilizing CuPy inside processor.py)
                processor.extract_and_buffer_multi(frame, rois, timestamp)
                
                # Draw ML polygons
                for name, polygon in rois.items():
                    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # ==================================================
            # HEADS UP DISPLAY
            # ==================================================
            fps = processor.get_current_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            bpm, freqs, filt_mag = processor.estimate_heart_rate()
            if bpm is not None:
                cv2.putText(frame, f"Est BPM: {bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # gt_hr = gt_reader.get_hr_at_time(timestamp)
            # if gt_hr is not None:
            #    cv2.putText(frame, f"True HR: {gt_hr:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ==================================================
            # TELEMETRY DISPATCH (Send to GUI Thread)
            # ==================================================
            # Update the dashboard every 3 frames to save CPU cycles
            if frame_counter % 3 == 0:
                raw_data = list(processor.raw_signal)
                filtered_data = processor.get_filtered_signal()
                
                payload = {
                    'raw_signal': raw_data,
                    'filtered_signal': filtered_data,
                    'freqs': freqs,
                    'filt_mag': filt_mag
                }
                
                # Push safely to queue, discarding old data if GUI is lagging
                try:
                    data_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    data_queue.put_nowait(payload)

    # Clean up when loop breaks
    cv2.destroyAllWindows()
    # Tell PyQt to quit nicely
    QApplication.quit()


# ==========================================
# 3. APPLICATION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Create the queue (size 1 prevents memory buildup, always keeps only the freshest data)
    telemetry_queue = queue.Queue(maxsize=1)

    # Instantiate the PyQt Application
    app = QApplication(sys.argv)

    # Start the Math & Camera logic in a background daemon thread
    worker = threading.Thread(target=rppg_processing_thread, args=(telemetry_queue,), daemon=True)
    worker.start()

    # Create and show the PyQtGraph Dashboard on the main thread
    window = RPPGDashboard(telemetry_queue)
    window.show()

    # Enter the blocking Qt event loop
    sys.exit(app.exec_())