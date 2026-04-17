"""
Webcam Streaming module with OpenCV library.
---------------------------------------
This module provides a robust interface for capturing video frames from a USB webcam using OpenCV.
It includes error handling for hardware issues and ensures proper resource management,
such as releasing the camera when done. It also attempts to lock hardware parameters like 
exposure and white balance for consistent rPPG signal quality. 
The class supports both live webcam feeds and pre-recorded video files, making it versatile 
for testing and deployment. 
"""

import cv2
import logging
from typing import Tuple, Optional, Union
import numpy as np

# Configure basic logging (Standard practice over using 'print')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebcamStream:
    """
    Interface for capturing video frames from a live webcam OR a video file.
    """
    def __init__(self, source: Union[int, str] = 0):
        """
        Initializes the video stream.
        Args:
            source: int (e.g., 0) for live webcam, or str (e.g., "video.mp4") for a file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Critical Error: Could not open video source: {self.source}.")
        
        # Dynamically extract the exact FPS of the video file or camera
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Fallback in case OpenCV fails to read the metadata
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0 
            logging.warning("Could not read FPS from source. Defaulting to 30.0 FPS.")

        logging.info(f"Video source initialized successfully. Operating at {self.fps} FPS.")

        # Only apply hardware locking (Exposure/WB) if the source is a live physical camera (integer)
        if isinstance(self.source, int):
            logging.info("Live camera detected. Attempting to lock hardware parameters...")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:
            # If playing a video file, ret=False means the video is finished.
            logging.info("End of video stream reached or hardware failed.")
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Video source released safely.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# --- Testing Block ---
if __name__ == "__main__":
    # Test script of the module
    try:
        with WebcamStream(source=0) as cam:
            while True:
                success, current_frame = cam.read_frame()
                if not success:
                    break
                
                cv2.imshow("Webcam Test (Press 'q' to quit)", current_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        logging.error(f"Application crashed: {e}")
    finally:
        cv2.destroyAllWindows()