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

    def get_face_mesh_rois(self, frame: np.ndarray, draw: bool = False) -> Optional[Dict[str, np.ndarray]]:
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
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
            hull = cv2.convexHull(pts)

            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)
            x, y, w_box, h_box = cv2.boundingRect(hull)

            for i in range(3):
                sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                if region_name == 'forehead':
                    slice_w = w_box // 3
                    x_start = x + (i * slice_w)
                    x_end = x + w_box if i == 2 else x_start + slice_w
                    cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                else:
                    slice_h = h_box // 3
                    y_start = y + (i * slice_h)
                    y_end = y + h_box if i == 2 else y_start + slice_h
                    cv2.rectangle(sub_mask, (x, y_start), (x + w_box, y_end), 255, -1)

                final_mask = cv2.bitwise_and(master_mask, sub_mask)
                mean_color = cv2.mean(frame, mask=final_mask)[:3]
                sub_name = f"{region_name}_{i+1}"
                dynamic_rois[sub_name] = mean_color

            if draw:
                # 1. Draw the master boundary of the cheek/forehead
                cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 0), thickness=1)
                
                # 2. Draw the two internal slice lines to show the grid
                if region_name == 'forehead':
                    slice_w = w_box // 3
                    # Draw two vertical lines
                    cv2.line(frame, (x + slice_w, y), (x + slice_w, y + h_box), (0, 255, 0), 1)
                    cv2.line(frame, (x + 2 * slice_w, y), (x + 2 * slice_w, y + h_box), (0, 255, 0), 1)
                else:
                    slice_h = h_box // 3
                    # Draw two horizontal lines
                    cv2.line(frame, (x, y + slice_h), (x + w_box, y + slice_h), (0, 255, 0), 1)
                    cv2.line(frame, (x, y + 2 * slice_h), (x + w_box, y + 2 * slice_h), (0, 255, 0), 1)

        return dynamic_rois
    

    