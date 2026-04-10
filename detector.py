"""
FaceDetector Module for rPPG Signal Extraction
---------------------------------------
This module implements a lightweight face detection algorithm using Haar Cascades, 
optimized for real-time applications in remote photoplethysmography (rPPG). 
It detects the largest face in the frame and isolates a central region of interest (ROI) to minimize noise from hair, 
background, and neck movement.
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Optional

class FaceDetector:
    """
    A lightweight face detection module using Haar Cascades method, optimized for rPPG applications.
    """

    def __init__(self):
        """Initializes the Haar Cascade classifier."""
        # OpenCV provides default cascades; we use the standard frontal face model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Critical Error: Failed to load Haar Cascade XML file.")
        logging.info("FaceDetector initialized with Haar Cascades.")

    def detect_largest_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects faces in the frame and returns the bounding box of the largest one.
        We only want to track one subject for the rPPG signal.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.

        Returns:
            Optional[Tuple[int, int, int, int]]: (x, y, w, h) of the face, or None if no face found.
        """
        # Haar Cascades require grayscale images for computation
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale parameters (scaleFactor, minNeighbors) can be tuned depending of the lightning
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100) # Ignore tiny background artifacts
        )

        if len(faces) == 0:
            return None

        # If multiple faces are found, assume the largest one is our subject
        if len(faces) > 1:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)

        return tuple(faces[0])

    def get_rppg_roi(self, face_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Shrinks the face bounding box to isolate the central region (cheeks/forehead),
        eliminating noise from hair, background, and neck movement.

        Args:
            face_box (Tuple[int, int, int, int]): Original (x, y, w, h) face box.

        Returns:
            Tuple[int, int, int, int]: Cropped ROI (x, y, w, h).
        """
        x, y, w, h = face_box
        
        # Crop 20% off the left, right, and bottom. Crop 10% off the top (keep forehead).
        roi_x = int(x + (w * 0.20))
        roi_y = int(y + (h * 0.10))
        roi_w = int(w * 0.60)
        roi_h = int(h * 0.70)
        
        return (roi_x, roi_y, roi_w, roi_h)