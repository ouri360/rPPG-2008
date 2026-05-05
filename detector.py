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
        
        # Added the continuous outer ring of the lips
        self.ROI_INDICES = {
            'forehead': [67, 10, 297, 299, 9, 69],
            'left_cheek': [117, 118, 101, 36, 205, 50],
            'right_cheek': [346, 347, 330, 266, 425, 280],
            'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
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
                
        if not self.cached_hulls:
            return None

        dynamic_rois = {}

        if draw:
            overlay = np.zeros_like(frame)
            face_clip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for region_name, hull in self.cached_hulls.items():
            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)
            x, y, w_box, h_box = cv2.boundingRect(hull)

            if draw:
                cv2.fillPoly(face_clip_mask, [hull], 255)

            # --- NEW: Handle the single Lips ROI ---
            if region_name == 'lips':
                sub_name = 'lips'
                
                if draw and ai_weights:
                    weight = ai_weights.get(sub_name, 0.10)
                    # Multiplier adjusted for 10 ROIs (Average weight is 0.10)
                    intensity_val = np.clip(weight * 8.0 * 255, 0, 255).astype(np.uint8)
                    color_sample = cv2.applyColorMap(np.array([[intensity_val]]), cv2.COLORMAP_INFERNO)
                    bgr_color = tuple(map(int, color_sample[0, 0]))
                else:
                    bgr_color = (0, 0, 0)
                
                if draw:
                    cv2.fillPoly(overlay, [hull], bgr_color)
                
                # Extract mean for the entire lip hull
                mean_color = cv2.mean(frame, mask=master_mask)[:3]
                dynamic_rois[sub_name] = mean_color

            # --- Handle the Sliced Cheek/Forehead ROIs ---
            else:
                for i in range(3):
                    sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    sub_name = f"{region_name}_{i+1}"
                    
                    if draw and ai_weights:
                        weight = ai_weights.get(sub_name, 0.10)
                        intensity_val = np.clip(weight * 8.0 * 255, 0, 255).astype(np.uint8)
                        color_sample = cv2.applyColorMap(np.array([[intensity_val]]), cv2.COLORMAP_INFERNO)
                        bgr_color = tuple(map(int, color_sample[0, 0]))
                    else:
                        bgr_color = (0, 0, 0)
                    
                    if region_name == 'forehead':
                        slice_w = w_box // 3
                        x_start = x + (i * slice_w)
                        x_end = x + w_box if i == 2 else x_start + slice_w
                        cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                        if draw:
                            cv2.rectangle(overlay, (x_start, y), (x_end, y + h_box), bgr_color, cv2.FILLED)
                    else:
                        slice_h = h_box // 3
                        y_start = y + (i * slice_h)
                        y_end = y + h_box if i == 2 else y_start + slice_h
                        cv2.rectangle(sub_mask, (x, y_start), (x + w_box, y_end), 255, -1)
                        if draw:
                            cv2.rectangle(overlay, (x, y_start), (x + w_box, y_end), bgr_color, cv2.FILLED)

                    final_mask = cv2.bitwise_and(master_mask, sub_mask)
                    mean_color = cv2.mean(frame, mask=final_mask)[:3]
                    dynamic_rois[sub_name] = mean_color

        if draw:
            overlay = cv2.bitwise_and(overlay, overlay, mask=face_clip_mask)
            inv_mask = cv2.bitwise_not(face_clip_mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            np.copyto(frame, cv2.add(frame_bg, overlay))

            for hull in self.cached_hulls.values():
                cv2.polylines(frame, [hull], isClosed=True, color=(255, 255, 255), thickness=1)

        return dynamic_rois