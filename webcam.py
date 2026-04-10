"""
Webcam Streaming module with OpenCV library.
---------------------------------------
This module provides a robust interface for capturing video frames from a USB webcam using OpenCV.
It includes error handling for hardware issues and ensures proper resource management.
"""

import cv2
import logging
from typing import Tuple, Optional
import numpy as np

# Configure basic logging (Standard practice over using 'print')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebcamStream:
    """
    A robust interface for capturing video frames from a USB webcam.

    Attributes:
        camera_index (int): The index of the camera device (default is 0 for built-in/primary).
        cap (cv2.VideoCapture): The OpenCV video capture object.
    """

    def __init__(self, camera_index: int = 0):
        """Initializes the webcam stream and verifies hardware availability."""
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Fail if the hardware is unavailable
        if not self.cap.isOpened():
            raise RuntimeError(f"Critical Error: Could not open camera at index {self.camera_index}. Check connections.")
        logging.info(f"Camera initialized successfully at index {self.camera_index}.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single frame from the webcam.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A boolean indicating success, and the frame array (or None if failed).
        """
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Hardware Warning: Failed to grab frame from camera.")
            return False, None
        return True, frame

    def release(self) -> None:
        """Safely releases the camera hardware."""
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Camera hardware released safely.")

    # Context Manager Magic Methods (__enter__ and __exit__)
    # These allow the use of: "with WebcamStream() as stream:"
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# --- Testing Block ---
if __name__ == "__main__":
    # Test script of the module
    try:
        with WebcamStream(camera_index=0) as cam:
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