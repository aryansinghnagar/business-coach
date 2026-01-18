"""
Video Source Handler Module

This module provides a unified interface for handling different video sources:
- Webcam (default camera)
- Local video files
- Video streams (WebRTC, RTSP, HTTP streams, etc.)

It abstracts away the differences between source types and provides a consistent
API for reading frames from any supported source.
"""

import cv2
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class VideoSourceType(Enum):
    """Enumeration of supported video source types."""
    WEBCAM = "webcam"
    FILE = "file"
    STREAM = "stream"


class VideoSourceHandler:
    """
    Handler for managing video sources of different types.
    
    This class provides a unified interface for reading frames from various
    video sources, abstracting away the implementation details of each source type.
    
    Usage:
        handler = VideoSourceHandler()
        handler.initialize_source(VideoSourceType.WEBCAM)
        
        while True:
            ret, frame = handler.read_frame()
            if not ret:
                break
            # Process frame
    """
    
    def __init__(self):
        """Initialize the video source handler."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type: Optional[VideoSourceType] = None
        self.source_path: Optional[str] = None
    
    def initialize_source(
        self,
        source_type: VideoSourceType,
        source_path: Optional[str] = None
    ) -> bool:
        """
        Initialize a video source.
        
        Args:
            source_type: Type of video source (WEBCAM, FILE, STREAM)
            source_path: Path to video file or stream URL (required for FILE/STREAM)
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Release existing source if any
        self.release()
        
        self.source_type = source_type
        self.source_path = source_path
        
        try:
            if source_type == VideoSourceType.WEBCAM:
                # Default webcam (index 0)
                self.cap = cv2.VideoCapture(0)
                
                # Set webcam properties for better quality
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
            elif source_type == VideoSourceType.FILE:
                if not source_path:
                    raise ValueError("source_path is required for FILE source type")
                
                self.cap = cv2.VideoCapture(source_path)
                
            elif source_type == VideoSourceType.STREAM:
                # For STREAM type, if source_path is None, try to use webcam as fallback
                # This handles the case where "stream" is selected but no URL is provided
                if not source_path:
                    print("Warning: STREAM source type selected but no path provided, using webcam as fallback")
                    self.cap = cv2.VideoCapture(0)
                    # Set webcam properties for better quality
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                else:
                    # For streams, use a longer timeout
                    self.cap = cv2.VideoCapture(source_path)
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Verify that the source opened successfully
            if not self.cap.isOpened():
                return False
            
            return True
            
        except Exception as e:
            print(f"Error initializing video source: {e}")
            self.release()
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame):
            - success: True if frame was read successfully, False otherwise
            - frame: BGR image array if successful, None otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return False, None
        
        return True, frame
    
    def get_properties(self) -> dict:
        """
        Get properties of the current video source.
        
        Returns:
            Dictionary containing video properties:
            - width: Frame width in pixels
            - height: Frame height in pixels
            - fps: Frames per second
            - frame_count: Total frames (for files) or -1 (for streams/webcam)
        """
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
    
    def set_frame_position(self, frame_number: int) -> bool:
        """
        Set the current frame position (for file sources only).
        
        Args:
            frame_number: Frame number to seek to
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False
        
        if self.source_type != VideoSourceType.FILE:
            return False  # Only works for file sources
        
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self) -> None:
        """Release the current video source and free resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.source_type = None
        self.source_path = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
