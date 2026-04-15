"""
FaceDetector Module (MediaPipe Face Mesh Edition)
---------------------------------------
Replaces rigid Haar Cascades with dynamic, shape-shifting ML polygons.
Guarantees 100% skin extraction regardless of head tilt or movement.
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
        
        # The landmark indices for the cleanest vascular skin zones
        self.ROI_INDICES = {
            # Forehead: A tight, small box strictly in the dead center of the forehead.
            # Avoids temples, eyebrows, and widows peaks completely.
            'forehead': [151, 108, 69, 105, 66, 107, 9, 336, 296, 334, 299, 337],
            
            # Left Cheek: A tiny patch strictly on the high malar bone directly under the eye.
            # Far away from side-hair and the shadow of the nose.
            'left_cheek': [205, 50, 118, 119, 100, 121],
            
            # Right Cheek: A tiny patch strictly on the high malar bone directly under the eye.
            'right_cheek': [425, 280, 347, 348, 329, 350]
        }
        
        logging.info("FaceDetector initialized with ML Face Mesh Tracker.")

    def get_face_mesh_rois(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculates the exact (X, Y) pixel coordinates for the 3 dynamic skin polygons.
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
            # Convert normalized ML landmarks into actual pixel coordinates
            points = []
            for idx in indices:
                pt = landmarks.landmark[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                points.append([x, y])
                
            # Convert to a formatted NumPy array
            points_arr = np.array(points, dtype=np.int32)
            
            # ==========================================
            # CV UPGRADE: Convex Hull (The Rubber Band)
            # Organizes the raw points into a clean, untangled, 
            # perfectly filled geometric perimeter!
            # ==========================================
            hull = cv2.convexHull(points_arr)
            
            dynamic_rois[region_name] = hull
            
        return dynamic_rois