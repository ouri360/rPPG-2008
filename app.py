import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify
import numpy as np

# Importe tes classes optimisées
from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor

app = Flask(__name__)

# --- VARIABLES GLOBALES ---
math_active = False
telemetry_data = {
    "bpm": "--",
    "snr": 0.0,
    "ecg_wave": [],
    "weights": {},
    "alphas": {}
}

# --- INITIALISATION MATÉRIELLE UNIQUE (Anti-Freeze USB) ---
# On démarre la caméra, MediaPipe et le processeur UNE SEULE FOIS au lancement du script.
# La caméra tourne 24h/24 en tâche de fond. Rafraîchir la page ne coupe plus le bus USB !
detector = FaceDetector(decimation_rate=2)
processor = SignalProcessor(buffer_seconds=12, target_fps=25.0)
cam = WebcamStream(source=0)

def generate_video_feed():
    """Générateur qui capture la caméra, applique MediaPipe et stream en MJPEG"""
    global math_active, telemetry_data, processor, cam
    
    frame_counter = 0
    is_calculating = False
    
    while True:
        # La caméra est globale, on lit simplement sa dernière image disponible
        success, frame = cam.read_frame()
        if not success or frame is None:
            time.sleep(0.01) # Petite pause pour soulager le CPU si la caméra est en retard
            continue
            
        frame_counter += 1
        timestamp = time.time()
        
        # 1. Détection des ROI (Toujours active : contours blancs au minimum)
        current_weights = processor.get_latest_weights() if math_active else None
        multi_rois = detector.get_face_mesh_rois(frame, draw=True, ai_weights=current_weights)

        # 2. Mathématiques (Seulement si le bouton Start a été cliqué)
        if math_active and multi_rois:
            processor.extract_and_buffer_multi(frame, multi_rois, timestamp)
            
            # Multi-threading sécurisé
            if frame_counter % 3 == 0 and not is_calculating:
                def math_worker():
                    nonlocal is_calculating
                    is_calculating = True
                    try:
                        # A. On stocke la réponse mathématique
                        signals = processor.get_filtered_signal()
                        
                        # B. LA VÉRIFICATION CRITIQUE
                        if signals is not None:
                            # C. On sépare les ondes en toute sécurité
                            sig_math, sig_visuel = signals
                            
                            if len(sig_math) > 30:
                                bpm, freqs, filt_mag = processor.estimate_heart_rate(filtered_signal=sig_math)
                                
                                if bpm is not None:
                                    # Mise à jour des données pour le Web
                                    display_pts = int(cam.fps * 3)
                                    wave_slice = sig_visuel[-display_pts:]
                                    min_val, max_val = np.min(wave_slice), np.max(wave_slice)
                                    
                                    telemetry_data["bpm"] = round(bpm, 1)
                                    telemetry_data["snr"] = round(np.max(filt_mag) / (np.mean(filt_mag) + 1e-8), 2)
                                    telemetry_data["ecg_wave"] = ((wave_slice - min_val) / (max_val - min_val + 1e-8)).tolist()
                                    telemetry_data["weights"] = processor.get_latest_weights()
                    except Exception as e:
                        print(f"Erreur Math_Worker : {e}")
                    finally:
                        is_calculating = False

                # Lancer le calcul sans bloquer l'image
                threading.Thread(target=math_worker, daemon=True).start()

        # 3. Encodage JPEG pour le Web
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES FLASK ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_math', methods=['POST'])
def toggle_math():
    global math_active, processor
    math_active = not math_active
    
    if not math_active:
        # On vide les buffers et la mémoire si on arrête
        processor = SignalProcessor(buffer_seconds=12, target_fps=25.0)
        telemetry_data["bpm"] = "--"
        telemetry_data["ecg_wave"] = []
        
    return jsonify({"status": "running" if math_active else "stopped"})

@app.route('/get_telemetry')
def get_telemetry():
    if not math_active:
        return jsonify({"status": "stopped"})
    return jsonify(telemetry_data)

if __name__ == '__main__':
    # Threaded=True est vital pour autoriser plusieurs connexions (caméra + requêtes data)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)