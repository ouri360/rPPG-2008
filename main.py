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

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
stop_event = threading.Event()

# ==========================================
# 1. THE GUI CLASS (Main Thread)
# ==========================================
class RPPGDashboard(QMainWindow):
    def __init__(self, data_queue):
        super().__init__()
        self.setWindowTitle("rPPG Real-Time Telemetry & Vision")
        self.resize(1400, 800) # Fenêtre plus large pour le Side-by-Side
        self.data_queue = data_queue

        # Layout Principal: Divisé en deux horizontalement
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- PANNEAU GAUCHE : LA VIDÉO ---
        self.video_widget = pg.GraphicsLayoutWidget()
        self.plot_video = self.video_widget.addPlot(title="Live Camera Feed")
        self.plot_video.hideAxis('bottom')
        self.plot_video.hideAxis('left')
        # 1. Empêche l'étirement (maintient le ratio original de la vidéo)
        self.plot_video.setAspectLocked(True)
        # 2. Inverse l'axe Y pour correspondre au format matriciel d'OpenCV
        self.plot_video.invertY(True)
        self.img_video = pg.ImageItem(axisOrder='row-major') 
        self.plot_video.addItem(self.img_video)
        layout.addWidget(self.video_widget, stretch=1) # Occupe 50% de l'écran

        # --- PANNEAU DROIT : LES GRAPHIQUES ---
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget, stretch=1) # Occupe l'autre 50%

        # 1. Raw Signal Plot
        self.plot_raw = self.graph_widget.addPlot(title="1. Raw Signal (Green Channel)")
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('g', width=2))
        self.graph_widget.nextRow()

        # 2. Filtered Signal Plot
        self.plot_filt = self.graph_widget.addPlot(title="2. Filtered Signal (Time Domain)")
        self.curve_filt = self.plot_filt.plot(pen=pg.mkPen('c', width=2))
        self.graph_widget.nextRow()

        # 3. Power Spectrum Plot
        self.plot_fft = self.graph_widget.addPlot(title="3. Power Spectrum")
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen('m', width=2))
        self.plot_fft.setLabel('bottom', "Frequency", units='Hz')

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33) 

    def update_plots(self):
        try:
            data = self.data_queue.get_nowait()
            
            # Mise à jour de la vidéo
            if data.get('frame') is not None:
                self.img_video.setImage(data['frame'], levels=(0, 255))
            
            # Mise à jour des graphiques
            if data.get('raw_signal') is not None and len(data['raw_signal']) > 0:
                self.curve_raw.setData(data['raw_signal'])
            if data.get('filtered_signal') is not None and len(data['filtered_signal']) > 0:
                self.curve_filt.setData(data['filtered_signal'])
            if data.get('freqs') is not None and data.get('filt_mag') is not None:
                self.curve_fft.setData(data['freqs'], data['filt_mag'])
                
        except queue.Empty:
            pass

    def closeEvent(self, event):
        logging.info("Shutting down safely...")
        stop_event.set() 
        event.accept()


# ==========================================
# 2. THE MATH WORKER (Runs on Background Thread)
# ==========================================
def rppg_processing_thread(data_queue):
    """
    Handles OpenCV, Camera I/O, MediaPipe, and Signal Processing.
    Operates entirely independently of the GUI to guarantee high FPS.
    """
    detector = FaceDetector()

    # Configuration de la source
    VIDEO_SOURCE = "dataset/UBFC-rPPG-Set2-Realistic/vid_subject1.avi"
    
    with WebcamStream(source=VIDEO_SOURCE) as cam:
        frame_counter = 0
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)

        last_rois = None

        # CORRECTION 1 : On boucle tant que l'utilisateur n'a pas fermé la fenêtre
        while not stop_event.is_set():
            success, frame = cam.read_frame()
            if not success: 
                logging.info("Fin de la vidéo atteinte. Analyse maintenue à l'écran.")
                break # On sort de la boucle de lecture, mais la fenêtre reste ouverte

            frame_counter += 1
            timestamp = time.time()

            # Optimisation MediaPipe (1 frame sur 5)
            if frame_counter % 5 == 0 or last_rois is None:
                rois = detector.get_face_mesh_rois(frame)
                if rois:
                    last_rois = rois
            else:
                rois = last_rois

            # Traitement des ROIs
            if rois:
                processor.extract_and_buffer_multi(frame, rois, timestamp)
                for name, polygon in rois.items():
                    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            # HUD
            fps = processor.get_current_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            bpm, freqs, filt_mag = processor.estimate_heart_rate()
            if bpm is not None:
                cv2.putText(frame, f"Est BPM: {bpm:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calc BPM...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # CORRECTION 2 : Conversion BGR vers RGB pour PyQtGraph
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Dispatch vers le GUI (toutes les 3 frames pour économiser le CPU)
            if frame_counter % 3 == 0:
                raw_data = list(processor.raw_signal)
                filtered_data = processor.get_filtered_signal()
                
                payload = {
                    'frame': frame_rgb, # La vidéo est maintenant envoyée !
                    'raw_signal': raw_data,
                    'filtered_signal': filtered_data,
                    'freqs': freqs,
                    'filt_mag': filt_mag
                }
                
                try:
                    data_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    data_queue.put_nowait(payload)

            # CORRECTION 3 : Simulateur de temps réel pour les vidéos
            if isinstance(VIDEO_SOURCE, str):
                time.sleep(1.0 / cam.fps)

    # CORRECTION 4 : Plus d'auto-destruction `QApplication.quit()`. 
    # Le thread se termine, mais PyQt reste ouvert pour afficher les résultats finaux.


# ==========================================
# 3. APPLICATION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    telemetry_queue = queue.Queue(maxsize=1)
    app = QApplication(sys.argv)

    worker = threading.Thread(target=rppg_processing_thread, args=(telemetry_queue,), daemon=True)
    worker.start()

    window = RPPGDashboard(telemetry_queue)
    window.show()

    sys.exit(app.exec_())