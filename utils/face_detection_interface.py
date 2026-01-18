"""
Face Detection Interface Module

This module defines an abstract interface for face detection implementations,
allowing the engagement detection system to work with different face detection
backends (MediaPipe, Azure Face API, etc.) interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class FaceDetectionResult:
    """
    Standardized face detection result.
    
    This structure provides a common format for face detection results
    regardless of the underlying detection method.
    """
    landmarks: np.ndarray  # Facial landmarks (N, 2) or (N, 3) array
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)
    confidence: float = 1.0  # Detection confidence (0-1)
    emotions: Optional[Dict[str, float]] = None  # Emotion scores if available
    head_pose: Optional[Dict[str, float]] = None  # Head pose (pitch, yaw, roll) if available
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes if available


class FaceDetectorInterface(ABC):
    """
    Abstract interface for face detection implementations.
    
    All face detection backends (MediaPipe, Azure Face API, etc.) must
    implement this interface to work with the engagement detection system.
    """
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image array (OpenCV format)
        
        Returns:
            List of FaceDetectionResult objects, one per detected face
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this face detection method is available and configured.
        
        Returns:
            True if the detector can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this face detection method.
        
        Returns:
            String name (e.g., "mediapipe", "azure_face_api")
        """
        pass
    
    def close(self) -> None:
        """
        Clean up resources. Override if needed.
        
        Default implementation does nothing.
        """
        pass
