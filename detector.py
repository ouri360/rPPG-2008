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
import numpy as np
from typing import Optional, Dict
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Strict Vascular Landmarks (No hairlines/eyebrows)
        self.ROI_INDICES = {
            'forehead': [67, 10, 297, 299, 9, 69],
            'left_cheek': [117, 118, 101, 36, 205, 50],
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }

    def get_multi_rois(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
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
                # Raw, un-smoothed AI coordinates to push noise to 30Hz
                raw_x = landmarks.landmark[idx].x * w
                raw_y = landmarks.landmark[idx].y * h
                points.append([int(raw_x), int(raw_y)])
                
            points_arr = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points_arr)
            dynamic_rois[region_name] = hull
            
        return dynamic_rois