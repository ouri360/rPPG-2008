"""
FaceDetector Module (Jetson Optimized with Inferno Colormap)
---------------------------------------
Replaces green transparency with a solid, mathematical Inferno colormap.
Extracts 10 dynamic ROIs (Forehead x3, Left Cheek x3, Right Cheek x3, Lips x1).
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Optional


class FaceDetector:
    def __init__(self, decimation_rate: int = 3):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.decimation_rate = decimation_rate
        self.frame_count = 0
        self.cached_hulls = {}
        
        # ROI INDICES for Forehead, Cheeks
        self.ROI_INDICES = {
            'forehead': [67, 10, 297, 299, 9, 69],
            'left_cheek': [117, 118, 101, 36, 205, 50],
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }
        
        logging.info(f"FaceDetector initialized. Decimation Rate: 1/{self.decimation_rate} frames.")

    def get_face_mesh_rois(self, frame: np.ndarray, draw: bool = False, ai_weights: dict = None) -> Optional[dict]:
        h, w = frame.shape[:2]
        self.frame_count += 1
        
        if self.frame_count % self.decimation_rate == 1 or not self.cached_hulls:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                self.cached_hulls = {}
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            for region_name, indices in self.ROI_INDICES.items():
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
                self.cached_hulls[region_name] = cv2.convexHull(pts)
                
        if not self.cached_hulls: return None
        dynamic_rois = {}
        if draw:
            overlay = np.zeros_like(frame)
            face_clip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for region_name, hull in self.cached_hulls.items():
            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)
            x, y, w_box, h_box = cv2.boundingRect(hull)
            if draw: cv2.fillPoly(face_clip_mask, [hull], 255)

            # --- Front : découpé en 3 ---
            if region_name == 'forehead':
                for i in range(3):
                    sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    sub_name = f"forehead_{i+1}"
                    weight = ai_weights.get(sub_name, 0.20) if ai_weights else 0.20
                    
                    slice_w = w_box // 3
                    x_start = x + (i * slice_w)
                    x_end = x + w_box if i == 2 else x_start + slice_w
                    cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                    
                    if draw:
                        intensity = np.clip(weight * 5.0 * 255, 0, 255).astype(np.uint8)
                        color = tuple(map(int, cv2.applyColorMap(np.array([[intensity]]), cv2.COLORMAP_INFERNO)[0,0]))
                        cv2.rectangle(overlay, (x_start, y), (x_end, y + h_box), color, -1)

                    final_mask = cv2.bitwise_and(master_mask, sub_mask)
                    dynamic_rois[sub_name] = cv2.mean(frame, mask=final_mask)[:3]
            
            # --- Joues : Un seul bloc  ---
            else:
                weight = ai_weights.get(region_name, 0.20) if ai_weights else 0.20
                if draw:
                    intensity = np.clip(weight * 5.0 * 255, 0, 255).astype(np.uint8)
                    color = tuple(map(int, cv2.applyColorMap(np.array([[intensity]]), cv2.COLORMAP_INFERNO)[0,0]))
                    cv2.fillPoly(overlay, [hull], color)
                
                dynamic_rois[region_name] = cv2.mean(frame, mask=master_mask)[:3]

        if draw:
            overlay = cv2.bitwise_and(overlay, overlay, mask=face_clip_mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(face_clip_mask))
            np.copyto(frame, cv2.add(frame_bg, overlay))
            for hull in self.cached_hulls.values():
                cv2.polylines(frame, [hull], True, (255, 255, 255), 1)

        return dynamic_rois