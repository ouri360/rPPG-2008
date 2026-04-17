"""
Webcam Streaming module with OpenCV library.
---------------------------------------
This module provides a robust interface for capturing video frames from a USB webcam using OpenCV.
It includes error handling for hardware issues and ensures proper resource management.
It also attempts to lock camera parameters (exposure, white balance, focus) for consistent rPPG signal quality.
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
        
        # ==========================================
        # DSP UPGRADE: Explicit V4L2 Backend
        # Forces OpenCV to talk directly to the Linux Kernel, 
        # bypassing GStreamer to allow hardware locking without admin rights.
        # ==========================================
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Critical Error: Could not open video source: {self.source}.")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0 
            logging.warning("Could not read FPS from source. Defaulting to 30.0 FPS.")

        logging.info(f"Video source initialized successfully. Operating at {self.fps} FPS.")

        # Attempt strict hardware locking natively via OpenCV V4L2
        if isinstance(self.source, int):
            logging.info("Live camera detected. Attempting to lock V4L2 hardware parameters via OpenCV...")
            
            # 1 = Manual Exposure, 3 = Auto Exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 100) # Adjust if image is too dark/bright
            
            # Disable Auto-White Balance
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            
            # Disable Auto-Focus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
            self.cap.set(cv2.CAP_PROP_FOCUS, 0) 
            
            # Disable Auto-Gain (if supported by your camera)
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            
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