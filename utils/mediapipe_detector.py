"""
MediaPipe Face Detection Implementation

This module provides a MediaPipe-based implementation of the FaceDetectorInterface.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Dict, Any

from utils.face_detection_interface import FaceDetectorInterface, FaceDetectionResult


class MediaPipeFaceDetector(FaceDetectorInterface):
    """
    MediaPipe-based face detector implementation.
    
    Uses MediaPipe Face Mesh to detect faces and extract 468 facial landmarks.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0-1)
            min_tracking_confidence: Minimum confidence for face tracking (0-1)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._available = True
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces using MediaPipe.
        
        Args:
            image: BGR image array
        
        Returns:
            List of FaceDetectionResult objects
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return []
        
        face_results = []
        height, width = image.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks (468 points)
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z * width  # Z is normalized by width
                landmarks.append([x, y, z])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # Calculate bounding box from landmarks
            x_coords = landmarks_array[:, 0]
            y_coords = landmarks_array[:, 1]
            left = int(np.min(x_coords))
            top = int(np.min(y_coords))
            right = int(np.max(x_coords))
            bottom = int(np.max(y_coords))
            bbox = (left, top, right - left, bottom - top)
            
            face_results.append(
                FaceDetectionResult(
                    landmarks=landmarks_array,
                    bounding_box=bbox,
                    confidence=1.0  # MediaPipe doesn't provide confidence per face
                )
            )
        
        return face_results
    
    def is_available(self) -> bool:
        """Check if MediaPipe is available."""
        return self._available
    
    def get_name(self) -> str:
        """Get detector name."""
        return "mediapipe"
    
    def close(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
