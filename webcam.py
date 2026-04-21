"""
Webcam Streaming module with OpenCV library.
---------------------------------------
This module provides a robust interface for capturing video frames from a USB webcam using OpenCV.
It includes error handling for hardware issues and ensures proper resource management.
The WebcamStream class can be used as a context manager to automatically release the camera resource when done. 
It can also handle file input for testing purposes, making it versatile for both live and pre-recorded 
video processing in rPPG applications. 
"""

import cv2
import logging
from typing import Tuple, Optional, Union
import numpy as np

# Configure basic logging (Standard practice over using 'print')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebcamStream:
    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        
        # Explicit V4L2 Backend for Linux generic webcams
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Critical Error: Could not open video source: {self.source}.")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0 
            logging.warning("Could not read FPS from source. Defaulting to 30.0.")

        # V4L2 Hardware Locking
        if isinstance(self.source, int):
            logging.info("Live camera detected. Locking V4L2 parameters...")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = Manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 100)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return (True, frame) if ret else (False, None)

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

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