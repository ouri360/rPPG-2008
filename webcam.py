"""
Webcam Streaming module with OpenCV library (Multithreaded).
---------------------------------------
Provides a robust, zero-blocking interface for capturing video frames.
Automatically spawns a background I/O thread for live webcams to prevent 
the USB controller from bottlenecking the Jetson Orin CPU.
"""

import cv2
import logging
import threading
from typing import Tuple, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebcamStream:
    """
    Interface for capturing video frames from a live webcam OR a video file.
    """
    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.is_live = isinstance(self.source, int)
        
        # 1. Initialize Backend
        if self.is_live:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG) 
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Critical Error: Could not open video source: {self.source}.")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0 
            logging.warning("Could not read FPS from source. Defaulting to 30.0 FPS.")

        logging.info(f"Video source initialized successfully. Operating at {self.fps} FPS.")

        # 2. Hardware Locking
        if self.is_live:
            logging.info("Live camera detected. Locking V4L2 hardware parameters...")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 100) 
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
            self.cap.set(cv2.CAP_PROP_FOCUS, 0) 
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 3. Multithreading Setup
        self.ret = False
        self.frame = None
        self.stopped = False

        if self.is_live:
            # Read the very first frame to establish the connection before threading
            self.ret, self.frame = self.cap.read()
            
            # Spawn the background I/O Thread
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            logging.info("Background I/O Thread started! Main CPU loop is now unblocked.")
        else:
            logging.info("Video file detected. Using standard sequential reading for testing.")

    def _update(self) -> None:
        """
        Runs continuously in the background thread (ONLY for live webcams).
        Constantly pulls the USB bus and stores the absolute latest frame in RAM.
        """
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            self.ret = ret
            self.frame = frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the instantly available frame without waiting for hardware."""
        if self.is_live:
            # Return the cached frame instantly
            if self.stopped and self.frame is None:
                return False, None
            
            # Return a copy to prevent the background thread from overwriting it while main.py draws the UI
            return self.ret, self.frame.copy() if self.frame is not None else None
        else:
            # Standard blocking read for exact frame-by-frame video processing
            return self.cap.read()

    def release(self) -> None:
        """Shuts down the thread and the camera."""
        self.stopped = True
        
        if self.is_live and hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
            
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        logging.info("Video source released safely.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

if __name__ == "__main__":
    try:
        with WebcamStream(source=0) as cam:
            while True:
                success, current_frame = cam.read_frame()
                if not success:
                    break
                
                cv2.imshow("Multithreaded Webcam Test (Press 'q' to quit)", current_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        logging.error(f"Application crashed: {e}")
    finally:
        cv2.destroyAllWindows()