"""
Environment Validator for rPPG Pipeline
---------------------------------------
Checks the installed libraries against the "Golden Stack" to prevent 
C-API crashes and deprecated Google MediaPipe methods.
"""

import sys
import logging

# Set up clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def validate_environment():
    logging.info("=== rPPG Environment Validator ===")
    logging.info(f"Python Version: {sys.version.split(' ')[0]}")
    
    all_good = True

    # 1. Check NumPy (CRITICAL: Must be < 2.0.0)
    try:
        import numpy as np
        np_version = np.__version__
        np_major = int(np_version.split('.')[0])
        
        if np_major >= 2:
            logging.error(f"❌ NumPy Version: {np_version} (ERROR: Must be 1.x! NumPy 2.0+ will crash MediaPipe.)")
            all_good = False
        else:
            logging.info(f"✅ NumPy Version: {np_version} (Perfect)")
    except ImportError:
        logging.error("❌ NumPy is not installed.")
        all_good = False

    # 2. Check MediaPipe (CRITICAL: Must be 0.10.21)
    try:
        import mediapipe as mp
        mp_version = mp.__version__
        
        if mp_version != "0.10.21":
            logging.warning(f"⚠️ MediaPipe Version: {mp_version} (WARNING: Script requires 0.10.21 for legacy mp.solutions API.)")
            all_good = False
        else:
            logging.info(f"✅ MediaPipe Version: {mp_version} (Perfect)")
    except ImportError:
        logging.error("❌ MediaPipe is not installed.")
        all_good = False

    # 3. Check OpenCV
    try:
        import cv2
        cv2_version = cv2.__version__
        
        if not cv2_version.startswith("4.9."):
            logging.warning(f"⚠️ OpenCV Version: {cv2_version} (WARNING: 4.9.0.80 is recommended for dependency stability.)")
        else:
            logging.info(f"✅ OpenCV Version: {cv2_version} (Perfect)")
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

    logging.info("==================================")
    
    if all_good:
        logging.info("SUCCESS: Your environment is perfectly configured to run the rPPG pipeline!")
    else:
        logging.info("ACTION REQUIRED: Please run `pip install -r requirements.txt` to fix dependencies.")

if __name__ == "__main__":
    validate_environment()