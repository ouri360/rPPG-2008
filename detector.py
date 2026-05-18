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
    def __init__(self, decimation_rate: int = 6):
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
        
        # 1. On ne recalcule les masques complexes que si MediaPipe se déclenche
        if self.frame_count % self.decimation_rate == 1 or not hasattr(self, 'cached_masks') or not self.cached_masks:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.cached_masks = {}
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            self.cached_masks = {}  # On va stocker les masques binaires prêts à l'emploi
            self.cached_hulls = {}  # Pour le dessin OpenCV

            for region_name, indices in self.ROI_INDICES.items():
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
                hull = cv2.convexHull(pts)
                self.cached_hulls[region_name] = hull
                
                # Création du masque mathématique brut
                master_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(master_mask, [hull], 255)
                
                x, y, w_box, h_box = cv2.boundingRect(hull)

                # Découpage du front
                if region_name == 'forehead':
                    slice_w = w_box // 3
                    for i in range(3):
                        sub_name = f"forehead_{i+1}"
                        sub_mask = np.zeros((h, w), dtype=np.uint8)
                        x_start = x + (i * slice_w)
                        x_end = x + w_box if i == 2 else x_start + slice_w
                        cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                        # On sauvegarde le masque final prêt à l'emploi
                        self.cached_masks[sub_name] = cv2.bitwise_and(master_mask, sub_mask)
                # Découpage des joues
                else:
                    self.cached_masks[region_name] = master_mask

        if not hasattr(self, 'cached_masks') or not self.cached_masks:
            return None

        # 2. EXTRACTION (S'exécute à chaque frame)
        dynamic_rois = {}
        for roi_name, precomputed_mask in self.cached_masks.items():
            dynamic_rois[roi_name] = cv2.mean(frame, mask=precomputed_mask)[:3]

        # 3. Dessin OpenCV 
        # 3. Dessin OpenCV 
        if draw:
            # A. Les couleurs Inferno (Seulement si le prog. est activée avec des poids)
            if ai_weights:
                overlay = np.zeros_like(frame)
                face_clip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                for roi_name, precomputed_mask in self.cached_masks.items():
                    weight = ai_weights.get(roi_name, 0.20)
                    intensity = np.clip(weight * 6.0 * 255, 0, 255).astype(np.uint8)
                    color = tuple(map(int, cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0,0]))
                    
                    overlay[precomputed_mask == 255] = color
                    cv2.bitwise_or(face_clip_mask, precomputed_mask, dst=face_clip_mask)

                frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(face_clip_mask))
                face_pixels = cv2.bitwise_and(frame, frame, mask=face_clip_mask)
                overlay_blended = cv2.addWeighted(face_pixels, 0.2, overlay, 0.8, 0)
                
                np.copyto(frame, cv2.add(frame_bg, overlay_blended))

            # B. Les contours blancs (Toujours actifs pour vérifier le tracking)
            for hull in self.cached_hulls.values():
                cv2.polylines(frame, [hull], True, (255, 255, 255), 1)

        return dynamic_rois