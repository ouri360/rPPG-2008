"""
FaceDetector Module for rPPG Signal Extraction
---------------------------------------
This module implements a lightweight face detection algorithm using Haar Cascades, 
optimized for real-time applications in remote photoplethysmography (rPPG). 
It detects the largest face in the frame and isolates a central region of interest (ROI) to minimize noise from hair, 
background, and neck movement. The module also includes temporal smoothing of the bounding box to reduce jitter 
and improve signal stability. It can also isolate specific sub-regions of the face (forehead, left cheek, right cheek) 
for more targeted signal extraction.
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Optional

# Exponential Moving Average factor for bounding box smoothing (lower = smoother, but lags more)
SMOOTHING_FACTOR = 0.2 

class FaceDetector:
    """
    A lightweight face detection module using Haar Cascades method, optimized for rPPG applications.
    """

    def __init__(self):
        """Initializes the Haar Cascade classifier."""
        # OpenCV provides default cascades; we use the standard frontal face model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Store the previous bounding box for smoothing
        self.prev_roi, self.prev_rois = None, None
        self.alpha = SMOOTHING_FACTOR

        #Cache to remember the last valid face position
        self.last_face_box = None
        
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
            scaleFactor=1.1,    # How much the image size is reduced at each image scale. Higher values = faster but less accurate
            minNeighbors=5,     # Must scane multiple times to confirm a face, reduces false positives
            minSize=(100, 100)  # Ignore tiny background artifacts
        )

        # 1. CACHING: If no face is found (cascade glitch), fallback to the last known position
        if len(faces) == 0:
            if self.last_face_box is not None:
                return self.last_face_box
            return None

        # Find the largest face detected in the current frame
        largest_face = tuple(max(faces, key=lambda rect: rect[2] * rect[3]))

        # 2. SPATIAL GATING: Prevent impossible sudden jumps (False Positives)
        if self.last_face_box is not None:
            x1, y1, w1, h1 = largest_face
            x2, y2, w2, h2 = self.last_face_box

            # Calculate the center (X, Y) of both boxes
            center_new = (x1 + w1 // 2, y1 + h1 // 2)
            center_old = (x2 + w2 // 2, y2 + h2 // 2)

            # Calculate Euclidean distance between the two centers
            distance = np.sqrt((center_new[0] - center_old[0])**2 + (center_new[1] - center_old[1])**2)

            # If the face jumped by more than 50% of its own width in a single frame, it's an artifact
            if distance > (w2 * 0.5):
                # Ignore the glitch, return the locked old face
                return self.last_face_box

        # If it passed the checks, update the cache with the new face
        self.last_face_box = largest_face
        return largest_face

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

        current_roi = (roi_x, roi_y, roi_w, roi_h)

        # Smooth the bounding box over time to prevent jitter
        if self.prev_roi is None:
            self.prev_roi = current_roi
        else:
            # Apply Exponential Moving Average (EMA)
            smoothed_roi = tuple(
                int(self.alpha * curr + (1 - self.alpha) * prev) 
                for curr, prev in zip(current_roi, self.prev_roi)
            )
            self.prev_roi = smoothed_roi
            return smoothed_roi
        
        return current_roi
    
    def get_multi_rois(self, face_box: Tuple[int, int, int, int]) -> dict:
        """
        Slices the main face bounding box into three specific regions: 
        Forehead, Left Cheek, and Right Cheek.
        Applies a temporal smoothing to prevent ROI jitter.
        """
        x, y, w, h = face_box
        
        # 1. Forehead: Top center, avoiding the hairline and eyebrows
        fh_x = int(x + (w * 0.25))
        fh_y = int(y + (h * 0.05))
        fh_w = int(w * 0.50)
        fh_h = int(h * 0.20)
        
        # 2. Left Cheek: Mid-left, avoiding the nose and eye
        lc_x = int(x + (w * 0.20))
        lc_y = int(y + (h * 0.50))
        lc_w = int(w * 0.15)
        lc_h = int(h * 0.20)
        
        # 3. Right Cheek: Mid-right, avoiding the nose and eye
        rc_x = int(x + (w * 0.65))
        rc_y = int(y + (h * 0.50))
        rc_w = int(w * 0.15)
        rc_h = int(h * 0.20)
        
        current_rois = {
            'forehead': (fh_x, fh_y, fh_w, fh_h),
            'left_cheek': (lc_x, lc_y, lc_w, lc_h),
            'right_cheek': (rc_x, rc_y, rc_w, rc_h)
        }

        # Apply Exponential Moving Average (EMA) to all regions
        if self.prev_rois is None:
            self.prev_rois = current_rois
            return current_rois
        else:
            smoothed_rois = {}
            for name in current_rois:
                # Extract the current and previous tuples for this specific ROI
                curr = current_rois[name]
                prev = self.prev_rois[name]
                
                # Apply smoothing to x, y, w, h simultaneously
                smoothed_tuple = tuple(
                    int(self.alpha * c + (1 - self.alpha) * p) 
                    for c, p in zip(curr, prev)
                )
                smoothed_rois[name] = smoothed_tuple
                
            # Update state for the next frame
            self.prev_rois = smoothed_rois
            return smoothed_rois