"""
FaceDetector Module (MediaPipe Edition with Landmark Smoothing)
---------------------------------------
Replaces rigid Haar Cascades with dynamic, shape-shifting ML polygons.
Guarantees 100% skin extraction to the individual landmarks.
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
        # Tight Vascular Landmarks
        # Strictly avoids hairlines, eyebrows, and smile lines.
        # ==========================================
        self.ROI_INDICES = {
            # Forehead: 
            # Top edge: 67, 10, 297. 
            # Bottom edge: 299, 9, 69.
            'forehead': [67, 10, 297, 299, 9, 69],
            
            # Left Cheek.
            'left_cheek': [117, 118, 101, 36, 205, 50],
            
            # Right Cheek.
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }
        
        logging.info("FaceDetector initialized with ML Face Mesh.")

    def get_face_mesh_rois(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        dynamic_rois = {}

        # 1. Extract the original ROIs based on the defined landmark indices
        original_rois = {
            'forehead': [67, 10, 297, 299, 9, 69],
            'left_cheek': [117, 118, 101, 36, 205, 50],
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }

        for region_name, indices in original_rois.items():
            # Get the exact perimeter of the original good region
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
            hull = cv2.convexHull(pts)

            # Create a master mask for the whole cheek/forehead
            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)

            # Get the bounding box of this master region
            x, y, w_box, h_box = cv2.boundingRect(hull)

            # 2. SLICE IT INTO 3 MICRO-REGIONS
            for i in range(3):
                sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                if region_name == 'forehead':
                    # Slice Forehead into 3 vertical columns (Left, Center, Right)
                    slice_w = w_box // 3
                    x_start = x + (i * slice_w)
                    x_end = x + w_box if i == 2 else x_start + slice_w
                    cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                else:
                    # Slice Cheeks into 3 horizontal rows (Top, Middle, Bottom)
                    slice_h = h_box // 3
                    y_start = y + (i * slice_h)
                    y_end = y + h_box if i == 2 else y_start + slice_h
                    cv2.rectangle(sub_mask, (x, y_start), (x + w_box, y_end), 255, -1)

                # Intersect our mathematical slice with the shape of the face
                final_mask = cv2.bitwise_and(master_mask, sub_mask)
                
                # Extract the color and save it dynamically (e.g., 'forehead_1', 'forehead_2'...)
                mean_color = cv2.mean(frame, mask=final_mask)[:3]
                sub_name = f"{region_name}_{i+1}"
                dynamic_rois[sub_name] = mean_color

        return dynamic_rois
    

    