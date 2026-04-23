"""
Environment Validator for rPPG Pipeline (Hybrid PC / Edge Edition)
---------------------------------------
Checks the installed libraries against the "Golden Stack" to prevent 
C-API crashes, Qt conflicts, and deprecated Google MediaPipe methods.
"""

import sys
import logging

# Set up clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def validate_environment():
    logging.info("=== rPPG Environment Validator (Hybrid PC/Jetson) ===")
    logging.info(f"Python Version: {sys.version.split(' ')[0]}\n")
    
    all_good = True

    # 1. Check NumPy (CRITICAL: Must be < 2.0.0)
    try:
        import numpy as np
        np_version = np.__version__
        np_major = int(np_version.split('.')[0])
        
        if np_major >= 2:
            logging.error(f"❌ NumPy Version: {np_version} (ERROR: Must be 1.x! NumPy 2.0+ will crash MediaPipe and CuPy.)")
            all_good = False
        else:
            logging.info(f"✅ NumPy Version: {np_version} (Perfect)")
    except ImportError:
        logging.error("❌ NumPy is not installed.")
        all_good = False

    # 2. Check MediaPipe (CRITICAL: Must be 0.10.21 or 0.10.18)
    try:
        import mediapipe as mp
        mp_version = mp.__version__
        
        if mp_version not in ["0.10.21", "0.10.18"]:
            logging.warning(f"⚠️ MediaPipe Version: {mp_version} (WARNING: Script optimized for 0.10.21 legacy API.)")
        else:
            logging.info(f"✅ MediaPipe Version: {mp_version} (Perfect)")
    except ImportError:
        logging.error("❌ MediaPipe is not installed.")
        all_good = False

    # 3. Check OpenCV (CRITICAL: Headless on PC, Native on Jetson)
    try:
        import cv2
        cv2_version = cv2.__version__
        
        if "site-packages" in cv2.__file__ and not cv2_version.startswith("4.9."):
            logging.warning(f"⚠️ OpenCV Version: {cv2_version} (WARNING: 4.9.0.80 Headless is recommended for PC stability.)")
        else:
            logging.info(f"✅ OpenCV Version: {cv2_version} (Loaded properly without Qt conflicts)")
    except ImportError:
        logging.error("❌ OpenCV is not installed.")
        all_good = False

    # 4. Check SciPy
    try:
        import scipy
        logging.info(f"✅ SciPy Version: {scipy.__version__} (Stable)")
    except ImportError:
        logging.error("❌ SciPy is not installed.")
        all_good = False

    # 5. Check PyQt5 & PyQtGraph (For Main Thread GUI)
    try:
        import PyQt5.QtCore
        import pyqtgraph as pg
        logging.info(f"✅ PyQt5 & PyQtGraph Version: {pg.__version__} (Dashboard Ready)")
    except ImportError:
        logging.error("❌ PyQt5 or PyQtGraph is missing. The Dashboard will crash.")
        all_good = False

    # 6. Check fastrlock (CuPy Dependency)
    try:
        import fastrlock
        logging.info("✅ fastrlock (CuPy Dependency: Present)")
    except ImportError:
        logging.warning("⚠️ fastrlock is missing. CuPy might fail to compile kernels.")
        # We don't set all_good = False because CuPy might still work depending on OS

    # 7. Check CuPy (GPU Acceleration)
    try:
        import cupy as cp
        cp_version = cp.__version__
        if cp_version != "13.0.0":
            logging.warning(f"⚠️ CuPy Version: {cp_version} (WARNING: 13.0.0 is strictly recommended for NumPy 1.x compatibility on PC.)")
        else:
            logging.info(f"✅ CuPy Version: {cp_version} (GPU Acceleration ACTIVE)")
    except ImportError as e:
        logging.warning(f"⚠️ CuPy is NOT loaded ({e}). Program will safely fallback to NumPy (CPU).")

    logging.info("\n==================================")
    
    if all_good:
        logging.info("SUCCESS: Your environment core is perfectly configured!")
    else:
        logging.info("ACTION REQUIRED: Please check the logs and install the missing/incorrect dependencies.")

if __name__ == "__main__":
    validate_environment()


# =========================================================================================
# =========================================================================================
# ================== 📖 RPPG PIPELINE INSTALLATION & DEPLOYMENT GUIDE =====================
# =========================================================================================
# =========================================================================================
"""
Le code est hybride : il s'exécute
sur le CPU (NumPy) s'il n'y a pas de GPU compatible, et bascule instantanément sur les 
Tensor Cores (CuPy) si l'environnement matériel le permet. 

L'interface graphique est gérée par PyQtGraph sur le Thread Principal pour éviter les 
crashs Linux XCB, tandis que la vision par ordinateur tourne en tâche de fond.

-----------------------------------------------------------------------------------------
💻 PARTIE 1 : INSTALLATION SUR PC DE DÉVELOPPEMENT (Ubuntu / Windows avec CUDA 12.x)
-----------------------------------------------------------------------------------------
Le but ici est de forcer l'installation de CuPy 13.0.0 (compatible NumPy 1.x) et de 
garantir l'utilisation d'OpenCV Headless (sans interface) pour ne pas faire crasher PyQt5.

1. Nettoyage de sécurité (dans le .venv) :
   pip uninstall numpy opencv-python opencv-contrib-python opencv-python-headless cupy cupy-cuda12x -y

2. Installation de la Stack de base :
   pip install "numpy==1.26.4" "opencv-python-headless==4.9.0.80"
   pip install mediapipe==0.10.21 scipy scikit-learn

3. Installation du Dashboard :
   pip install PyQt5 pyqtgraph

4. Installation de l'Accélération GPU (Strict) :
   pip install fastrlock
   pip install cupy-cuda12x==13.0.0 --no-deps --no-cache-dir

5. Si CuPy demande "libnvrtc.so.12" à l'exécution, ajouter le compilateur CUDA au terminal :
   pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
   export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH


-----------------------------------------------------------------------------------------
🚀 PARTIE 2 : PORTAGE SUR NVIDIA JETSON ORIN NANO (JetPack 6.2.1)
-----------------------------------------------------------------------------------------
ATTENTION : L'installation sur Jetson est radicalement différente de celle sur PC. 
Le Jetson possède sa propre version d'OpenCV matérielle (compilée avec GStreamer/CUDA) 
fournie par NVIDIA. Si on installe OpenCV via pip, on perd 80% des performances vidéo.

1. Création du VENV perméable (Étape la plus critique) :
   # Cela permet au VENV de "voir" l'OpenCV natif de l'OS.
   python3 -m venv rppg_env --system-site-packages
   source rppg_env/bin/activate

2. Installation des dépendances (SANS OPENCV) :
   # Attention : Ne tapez JAMAIS "pip install opencv-python" sur le Jetson.
   pip install "numpy<2.0.0"
   pip install mediapipe==0.10.21 scipy scikit-learn
   pip install PyQt5 pyqtgraph fastrlock

3. Installation de CuPy sur Jetson :
   # Sous JetPack 6.x (CUDA 12.2), on peut tenter la roue officielle aarch64 :
   pip install cupy-cuda12x
   
   # Si pip ne trouve pas de roue pré-compilée pour aarch64, compilez-le depuis les sources :
   # (Cela va prendre ~15 minutes et utiliser le compilateur nvcc intégré du Jetson)
   pip install cupy

4. Lancement matériel (GStreamer) :
   Dans `webcam.py`, assurez-vous d'utiliser la chaîne GStreamer (`v4l2src` ou `nvarguscamerasrc`)
   plutôt que `cv2.CAP_V4L2` pour forcer le Jetson à décoder la vidéo via son ISP matériel.
"""