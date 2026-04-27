import cv2
import numpy as np
import mediapipe as mp

def test_sliced_rois():
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Your ORIGINAL 3 stable regions
    original_rois = {
        'forehead': [67, 10, 297, 299, 9, 69],
        'left_cheek': [117, 118, 101, 36, 205, 50],
        'right_cheek': [346, 347, 330, 266, 425, 280]
    }

    # Generate 9 fixed colors so the regions don't flash randomly
    colors = {}
    for region in original_rois.keys():
        for i in range(3):
            colors[f"{region}_{i+1}"] = tuple(np.random.randint(50, 255, 3).tolist())

    cap = cv2.VideoCapture("dataset/UBFC-rPPG-Set2-Realistic/vid_subject1.avi")
    print("Press 'q' to close the window.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        # Create a copy of the frame to draw our semi-transparent masks
        overlay = frame.copy()
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            for region_name, indices in original_rois.items():
                # 1. Get the exact perimeter of the original good region
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
                hull = cv2.convexHull(pts)

                # Create the master mask for the whole cheek/forehead
                master_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(master_mask, [hull], 255)

                # Get the bounding box of this master region
                x, y, w_box, h_box = cv2.boundingRect(hull)

                # 2. Slice it into 3 micro-regions
                for i in range(3):
                    sub_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    
                    if region_name == 'forehead':
                        # Slice Forehead into 3 vertical columns
                        slice_w = w_box // 3
                        x_start = x + (i * slice_w)
                        x_end = x + w_box if i == 2 else x_start + slice_w
                        cv2.rectangle(sub_mask, (x_start, y), (x_end, y + h_box), 255, -1)
                    else:
                        # Slice Cheeks into 3 horizontal rows
                        slice_h = h_box // 3
                        y_start = y + (i * slice_h)
                        y_end = y + h_box if i == 2 else y_start + slice_h
                        cv2.rectangle(sub_mask, (x, y_start), (x + w_box, y_end), 255, -1)

                    # Intersect our mathematical slice with the shape of the face
                    final_mask = cv2.bitwise_and(master_mask, sub_mask)
                    
                    # Apply the specific color to the overlay where the mask is active
                    sub_name = f"{region_name}_{i+1}"
                    overlay[final_mask == 255] = colors[sub_name]

        # Blend the colored overlay with the original frame (40% opacity)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.imshow('9-Region Sliced ROIs Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_sliced_rois()