"""
SignalProcessor Module for rPPG Signal Extraction
---------------------------------------
This module handles the extraction and buffering of the raw rPPG signal from video frames.
It crops the detected face region, isolates the green channel, calculates the spatial average,
and maintains a rolling buffer of the signal values and their corresponding timestamps for real-time processing.
"""

import numpy as np
import time
import logging
from collections import deque
from typing import Tuple

class SignalProcessor:
    """
    Handles the extraction and buffering of the raw rPPG signal from video frames.
    """

    def __init__(self, buffer_seconds: int = 10, target_fps: int = 30):
        """
        Initializes rolling buffers for the signal and timestamps.
        
        Args:
            buffer_seconds (int): How many seconds of data to hold in memory.
            target_fps (int): Expected framerate of the camera.
        """
        # Calculate maximum buffer size
        self.max_length = buffer_seconds * target_fps
        
        # deques automatically pop the oldest item when maxlen is reached
        self.raw_signal = deque(maxlen=self.max_length)
        self.timestamps = deque(maxlen=self.max_length)
        
        logging.info(f"SignalProcessor initialized with a {buffer_seconds}-second buffer.")

    def extract_and_buffer(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
        """
        Crops the ROI, isolates the green channel, calculates the spatial average, 
        and appends it to the time-series buffer.

        Args:
            frame (np.ndarray): The full BGR frame.
            roi (Tuple[int, int, int, int]): The (x, y, w, h) bounding box.

        Returns:
            float: The spatial average of the green channel for this frame.
        """
        x, y, w, h = roi
        
        # 1. NumPy Slicing: Extract the region of interest
        # Array shape is (rows, cols, channels), which translates to (y, x, c)
        cropped_roi = frame[y:y+h, x:x+w]

        # 2. Channel Isolation: Isolate the Green channel (Index 1 in BGR)
        # The syntax [:, :, 1] means "take all rows, all cols, but only channel 1 (green)"
        green_channel = cropped_roi[:, :, 1]

        # 3. Spatial Averaging: Calculate the mean pixel value
        spatial_average = float(np.mean(green_channel))

        # 4. Buffering: Store the value and the exact time it was captured
        self.raw_signal.append(spatial_average)
        self.timestamps.append(time.time())

        return spatial_average

    def get_signal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the current buffers as NumPy arrays for filtering/FFT.
        """
        return np.array(self.raw_signal), np.array(self.timestamps)