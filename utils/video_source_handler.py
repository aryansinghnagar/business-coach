"""
Video Source Handler Module

This module provides a unified interface for handling different video sources:
- Webcam (default camera)
- Local video files
- Video streams (WebRTC, RTSP, HTTP streams, etc.)
- Partner (browser-fed: frames sent from the frontend via getDisplayMedia)

It abstracts away the differences between source types and provides a consistent
API for reading frames from any supported source.
"""

import sys
import cv2
from enum import Enum
from typing import Optional, Tuple
import numpy as np
import threading

# Shared state for partner (browser) source: latest frame from frontend
_partner_frame: Optional[np.ndarray] = None
_partner_frame_lock = threading.Lock()


def set_partner_frame(frame_bgr: Optional[np.ndarray]) -> None:
    """Set the latest frame received from the browser (partner source)."""
    global _partner_frame
    with _partner_frame_lock:
        _partner_frame = frame_bgr.copy() if frame_bgr is not None else None


def get_partner_frame() -> Optional[np.ndarray]:
    """Get a copy of the latest partner frame (does not clear). Returns None if none available."""
    with _partner_frame_lock:
        out = _partner_frame
        return out.copy() if out is not None else None


def has_partner_frame() -> bool:
    """Return True if a partner frame is available."""
    with _partner_frame_lock:
        return _partner_frame is not None


# Max width for partner frames (resize larger frames to reduce memory and detection latency)
PARTNER_FRAME_MAX_WIDTH = 1280


def set_partner_frame_from_bytes(image_bytes: bytes) -> bool:
    """
    Decode image bytes (e.g. JPEG) to BGR and set as latest partner frame.
    Frames wider than PARTNER_FRAME_MAX_WIDTH are resized to reduce memory and processing time.
    Returns True if decoding and set succeeded, False otherwise.
    """
    if not image_bytes:
        return False
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return False
    h, w = frame.shape[:2]
    if w > PARTNER_FRAME_MAX_WIDTH:
        scale = PARTNER_FRAME_MAX_WIDTH / w
        new_w = PARTNER_FRAME_MAX_WIDTH
        new_h = int(round(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    set_partner_frame(frame)
    return True


class VideoSourceType(Enum):
    """Enumeration of supported video source types."""
    WEBCAM = "webcam"
    FILE = "file"
    STREAM = "stream"
    PARTNER = "partner"


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
        source_path: Optional[str] = None,
        lightweight: bool = False,
    ) -> bool:
        """
        Initialize a video source.

        Args:
            source_type: Type of video source (WEBCAM, FILE, STREAM)
            source_path: Path to video file or stream URL (required for FILE/STREAM)
            lightweight: If True, use lower webcam resolution (640x360) for faster processing
        """
        self.release()
        self.source_type = source_type
        self.source_path = source_path

        try:
            if source_type == VideoSourceType.WEBCAM:
                apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
                self.cap = None
                for api in apis:
                    for index in (0, 1, 2):
                        try:
                            cap = cv2.VideoCapture(index, api)
                            if cap.isOpened() and cap.read()[0]:
                                self.cap = cap
                                break
                        except Exception:
                            pass
                    if self.cap is not None:
                        break
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    w, h = (640, 360) if lightweight else (1280, 720)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            elif source_type == VideoSourceType.FILE:
                if not source_path:
                    raise ValueError("source_path is required for FILE source type")
                
                self.cap = cv2.VideoCapture(source_path)
                
            elif source_type == VideoSourceType.STREAM:
                # For STREAM type, if source_path is None, try to use webcam as fallback
                if not source_path:
                    print("Warning: STREAM source type selected but no path provided, using webcam as fallback")
                    apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
                    self.cap = None
                    for api in apis:
                        for index in (0, 1):
                            try:
                                cap = cv2.VideoCapture(index, api)
                                if cap.isOpened():
                                    self.cap = cap
                                    break
                            except Exception:
                                pass
                        if self.cap is not None:
                            break
                    if not self.cap or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                else:
                    self.cap = cv2.VideoCapture(source_path)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            elif source_type == VideoSourceType.PARTNER:
                # Partner source: frames pushed from browser (getDisplayMedia).
                self.cap = None
                return True
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Verify that the source opened successfully (not for PARTNER)
            if self.cap is not None and not self.cap.isOpened():
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
        if self.source_type == VideoSourceType.PARTNER:
            frame = get_partner_frame()
            return (True, frame) if frame is not None else (False, None)
        
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
            - frame_count: Total frames (for files) or -1 (for streams/webcam/partner)
        """
        if self.source_type == VideoSourceType.PARTNER:
            return {'width': 0, 'height': 0, 'fps': 30, 'frame_count': -1}
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
        if self.source_type == VideoSourceType.PARTNER:
            set_partner_frame(None)
        self.source_type = None
        self.source_path = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
