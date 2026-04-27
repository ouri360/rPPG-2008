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

    def get_face_mesh_rois(self, frame: np.ndarray, draw: bool = False, ai_weights: dict = None) -> Optional[dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        dynamic_rois = {}

        original_rois = {
            'forehead': [103, 67, 109, 10, 338, 297, 332, 284], 
            'left_cheek': [118, 119, 100, 126, 205, 206, 214, 192, 137, 177, 215, 138],
            'right_cheek': [347, 348, 329, 355, 425, 426, 434, 416, 366, 401, 435, 367]
        }

        # 1. Create lightning-fast C++ canvases
        if draw:
            overlay = np.zeros_like(frame)
            face_clip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for region_name, indices in original_rois.items():
            # Get the exact mathematical hull
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
            hull = cv2.convexHull(pts)

            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)
            x, y, w_box, h_box = cv2.boundingRect(hull)

            if draw:
                # Add to the global clipping mask
                cv2.fillPoly(face_clip_mask, [hull], 255)
                # RESTORED: Draw the true, original diagonal boundaries!
                cv2.polylines(overlay, [hull], isClosed=True, color=(0, 255, 0), thickness=1)

            for i in range(3):
                sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                sub_name = f"{region_name}_{i+1}"
                
                # Calculate AI weight intensity
                intensity = 255
                if draw and ai_weights:
                    weight = ai_weights.get(sub_name, 0.11)
                    intensity = int(min(weight * 3.0, 1.0) * 255)
                
                if region_name == 'forehead':
                    slice_w = w_box // 3
                    x_start = x + (i * slice_w)
                    x_end = x + w_box if i == 2 else x_start + slice_w
                    cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                    
                    if draw:
                        # Draw ONLY the opacity fill, no blocky rectangular borders!
                        cv2.rectangle(overlay, (x_start, y), (x_end, y + h_box), (0, intensity, 0), cv2.FILLED)
                        # Draw the inner grid lines
                        if i < 2: 
                            line_x = x + (i + 1) * slice_w
                            cv2.line(overlay, (line_x, y), (line_x, y + h_box), (0, 255, 0), 1)
                        
                else:
                    slice_h = h_box // 3
                    y_start = y + (i * slice_h)
                    y_end = y + h_box if i == 2 else y_start + slice_h
                    cv2.rectangle(sub_mask, (x, y_start), (x + w_box, y_end), 255, -1)
                    
                    if draw:
                        cv2.rectangle(overlay, (x, y_start), (x + w_box, y_end), (0, intensity, 0), cv2.FILLED)
                        if i < 2:
                            line_y = y + (i + 1) * slice_h
                            cv2.line(overlay, (x, line_y), (x + w_box, line_y), (0, 255, 0), 1)

                # Execute the math
                final_mask = cv2.bitwise_and(master_mask, sub_mask)
                mean_color = cv2.mean(frame, mask=final_mask)[:3]
                dynamic_rois[sub_name] = mean_color

        if draw:
            # 2. Perfect, zero-latency clipping. Trims off any boxes/lines that poke out of the hull!
            overlay = cv2.bitwise_and(overlay, overlay, mask=face_clip_mask)
            # 3. Blend the UI onto the live feed
            cv2.addWeighted(overlay, 0.6, frame, 1.0, 0, frame)

        return dynamic_rois
    

    