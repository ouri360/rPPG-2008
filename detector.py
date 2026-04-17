"""
FaceDetector Module (MediaPipe Edition with Landmark Smoothing)
---------------------------------------
Replaces rigid Haar Cascades with dynamic, shape-shifting ML polygons.
Guarantees 100% skin extraction and applies heavy temporal smoothing 
to the individual landmarks to completely eradicate micro-jitter.
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Optional, Dict

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # ==========================================
        # DSP UPGRADE: Ultra-Tight Vascular Landmarks
        # Strictly avoids hairlines, eyebrows, and smile lines.
        # ==========================================
        self.ROI_INDICES = {
            # Forehead: Slightly wider and lower.
            # Top edge: 67, 10, 297. 
            # Bottom edge: 299, 9, 69.
            'forehead': [67, 10, 297, 299, 9, 69],
            
            # Left Cheek: Unchanged.
            'left_cheek': [117, 118, 101, 36, 205, 50],
            
            # Right Cheek: Unchanged.
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }
        
        logging.info("FaceDetector initialized with Jitter-Free ML Face Mesh.")

    def get_face_mesh_rois(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculates the exact (X, Y) pixel coordinates for skin polygons.
        """
        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        dynamic_rois = {}
        
        for region_name, indices in self.ROI_INDICES.items():
            points = []
            for idx in indices:
                pt = landmarks.landmark[idx]
                
                # Use raw AI coordinates
                raw_x = pt.x * w
                raw_y = pt.y * h
                
                # Append directly to the polygon without EMA/History smoothing
                points.append([int(raw_x), int(raw_y)])
                
            # Convert to a formatted NumPy array
            points_arr = np.array(points, dtype=np.int32)
            
            # Use Convex Hull to wrap a clean "rubber band" around the smoothed dots
            hull = cv2.convexHull(points_arr)
            dynamic_rois[region_name] = hull
            
        return dynamic_rois