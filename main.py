"""Main application for rPPG signal extraction."""

import cv2
import logging
from webcam import WebcamStream
from detector import FaceDetector

def main():
    detector = FaceDetector()

    # Context manager safely handles camera hardware
    with WebcamStream(camera_index=0) as cam:
        logging.info("Starting rPPG processing loop...")
        
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            # 1. Detect the face
            face_box = detector.detect_largest_face(frame)

            if face_box:
                # 2. Extract the Region of Interest for rPPG
                roi_box = detector.get_rppg_roi(face_box)
                
                rx, ry, rw, rh = roi_box
                
                # Draw a green rectangle to visualize the ROI
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

            # Display the result
            cv2.imshow("rPPG - ROI Extraction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested shutdown.")
                break

if __name__ == "__main__":
    main()