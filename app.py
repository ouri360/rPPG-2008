import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify
import numpy as np

# Importe tes classes ultra-optimisées
from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor

app = Flask(__name__)

# --- VARIABLES GLOBALES (Lien entre l'IA et le Web) ---
math_active = False
telemetry_data = {
    "bpm": "--",
    "snr": 0.0,
    "ecg_wave": [],
    "weights": {},
    "alphas": {}
}

detector = FaceDetector(decimation_rate=2)
processor = SignalProcessor(buffer_seconds=12, target_fps=25.0)

def generate_video_feed():
    """Générateur qui capture la caméra, applique MediaPipe et stream en MJPEG"""
    global math_active, telemetry_data
    
    with WebcamStream(source=0) as cam:
        frame_counter = 0
        
        while True:
            success, frame = cam.read_frame()
            if not success:
                break
                
            frame_counter += 1
            timestamp = time.time()
            
            # 1. Détection des ROI (Toujours active)
            current_weights = processor.get_latest_weights() if math_active else None
            multi_rois = detector.get_face_mesh_rois(frame, draw=True, ai_weights=current_weights)

            # 2. Mathématiques (Seulement si le bouton Start a été cliqué)
            if math_active and multi_rois:
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)
                
                # Exécution asynchrone (comme ton math_worker)
                if frame_counter % 3 == 0:
                    sig_math, sig_visuel = processor.get_filtered_signal()
                    if sig_math is not None and len(sig_math) > 30:
                        bpm, freqs, filt_mag = processor.estimate_heart_rate(filtered_signal=sig_math)
                        
                        if bpm is not None:
                            # Mise à jour des variables globales pour la page web
                            display_pts = int(cam.fps * 3)
                            wave_slice = sig_visuel[-display_pts:]
                            min_val, max_val = np.min(wave_slice), np.max(wave_slice)
                            
                            telemetry_data["bpm"] = round(bpm, 1)
                            telemetry_data["snr"] = round(np.max(filt_mag) / (np.mean(filt_mag) + 1e-8), 2)
                            telemetry_data["ecg_wave"] = ((wave_slice - min_val) / (max_val - min_val + 1e-8)).tolist()
                            telemetry_data["weights"] = processor.get_latest_weights()

            # 3. Encodage JPEG pour le Web
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Format MJPEG standard
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES FLASK ---

@app.route('/')
def index():
    """Charge la page HTML principale"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route de streaming vidéo"""
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_math', methods=['POST'])
def toggle_math():
    """Bouton Start/Stop depuis la page web"""
    global math_active, processor
    math_active = not math_active
    if not math_active:
        # Réinitialiser le processeur si on arrête
        processor = SignalProcessor(buffer_seconds=12, target_fps=25.0)
    return jsonify({"status": "running" if math_active else "stopped"})

@app.route('/get_telemetry')
def get_telemetry():
    """Envoie les données mathématiques à l'interface"""
    if not math_active:
        return jsonify({"status": "stopped"})
    return jsonify(telemetry_data)

if __name__ == '__main__':
    # Threaded=True est vital pour que Flask ne bloque pas ta boucle OpenCV !
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)