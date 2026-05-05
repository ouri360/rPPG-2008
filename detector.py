"""
FaceDetector Module (Jetson Optimized with Inferno Colormap)
---------------------------------------
Replaces green transparency with a solid, mathematical Inferno colormap.
Uses Frame Decimation to halve CPU usage by caching polygon coordinates.
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Optional


class FaceDetector:
    """Detects facial regions using MediaPipe Face Mesh and extracts mean RGB values."""

    def __init__(self, decimation_rate: int = 3):
        """Initializes the FaceDetector with MediaPipe and sets up decimation."""
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
        
        self.ROI_INDICES = {
            'forehead': [67, 10, 297, 299, 9, 69],
            'left_cheek': [117, 118, 101, 36, 205, 50],
            'right_cheek': [346, 347, 330, 266, 425, 280]
        }
        
        logging.info(f"FaceDetector initialized. Decimation Rate: 1/{self.decimation_rate} frames.")

    def get_face_mesh_rois(self, frame: np.ndarray, draw: bool = False, ai_weights: dict = None) -> Optional[dict]:
        """Detects facial regions and extracts mean RGB values. Visualizes weights with solid Inferno colors."""
        h, w = frame.shape[:2]
        self.frame_count += 1
        
        # 1. THE DECIMATION LOGIC
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

        # 2. COLOR EXTRACTION & INFERNO VISUALIZATION
        dynamic_rois = {}

        if draw:
            # We create a black canvas to build our solid Inferno blocks
            overlay = np.zeros_like(frame)
            face_clip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for region_name, hull in self.cached_hulls.items():
            master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(master_mask, [hull], 255)
            x, y, w_box, h_box = cv2.boundingRect(hull)

            if draw:
                cv2.fillPoly(face_clip_mask, [hull], 255)

            for i in range(3):
                sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                sub_name = f"{region_name}_{i+1}"
                
                # --- INFERNO COLOR LOGIC ---
                if draw and ai_weights:
                    weight = ai_weights.get(sub_name, 0.11)
                    # We scale the weight (usually ~0.11) to a full 0-255 range for the colormap
                    # A factor of 4.0 makes an average region look red/orange
                    intensity_val = np.clip(weight * 4.0 * 255, 0, 255).astype(np.uint8)
                    
                    # Convert single intensity to Inferno BGR
                    # We use a 1x1 image to lookup the color from the colormap
                    color_sample = cv2.applyColorMap(np.array([[intensity_val]]), cv2.COLORMAP_INFERNO)
                    bgr_color = tuple(map(int, color_sample[0, 0]))
                else:
                    bgr_color = (0, 0, 0) # Default to black if no weights
                
                # Sub-ROI Slicing
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

                # Extract signal (unaffected by drawing)
                final_mask = cv2.bitwise_and(master_mask, sub_mask)
                mean_color = cv2.mean(frame, mask=final_mask)[:3]
                dynamic_rois[sub_name] = mean_color

        if draw:
            # Mask the overlay to the face shape
            overlay = cv2.bitwise_and(overlay, overlay, mask=face_clip_mask)
            
            # --- 100% SOLID REPLACEMENT ---
            # We clear the area on the frame where the ROIs are, then add the solid overlay
            inv_mask = cv2.bitwise_not(face_clip_mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            # Combine the face (minus ROIs) with the solid Inferno ROIs
            np.copyto(frame, cv2.add(frame_bg, overlay))

            # Add thin borders for definition
            for hull in self.cached_hulls.values():
                cv2.polylines(frame, [hull], isClosed=True, color=(255, 255, 255), thickness=1)

        return dynamic_rois