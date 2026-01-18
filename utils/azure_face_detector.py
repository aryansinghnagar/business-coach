"""
Azure Face API Detection Implementation

This module provides an Azure Face API-based implementation of the FaceDetectorInterface.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from utils.face_detection_interface import FaceDetectorInterface, FaceDetectionResult
from services.azure_face_api import get_azure_face_api_service


class AzureFaceAPIDetector(FaceDetectorInterface):
    """
    Azure Face API-based face detector implementation.
    
    Uses Azure Face API to detect faces and extract facial landmarks and attributes.
    """
    
    def __init__(self):
        """Initialize Azure Face API detector."""
        self.service = get_azure_face_api_service()
        self._available = self.service is not None
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces using Azure Face API.
        
        Args:
            image: BGR image array
        
        Returns:
            List of FaceDetectionResult objects
        """
        if not self.service:
            return []
        
        try:
            # Detect faces with landmarks and attributes
            face_data_list = self.service.detect_faces(
                image,
                return_face_landmarks=True,
                return_face_attributes=True
            )
            
            if not face_data_list:
                return []
            
            face_results = []
            
            for face_data in face_data_list:
                # Extract landmarks
                landmarks = self.service.extract_landmarks_from_face(face_data)
                
                if landmarks is None:
                    continue
                
                # Extract bounding box
                bbox = self.service.get_face_rectangle(face_data)
                
                # Extract emotions
                emotions = self.service.extract_emotion_from_face(face_data)
                
                # Extract head pose
                head_pose = self.service.extract_head_pose_from_face(face_data)
                
                # Convert 2D landmarks to 3D (add z=0 for compatibility)
                if landmarks.shape[1] == 2:
                    z_coords = np.zeros((landmarks.shape[0], 1), dtype=landmarks.dtype)
                    landmarks = np.hstack([landmarks, z_coords])
                
                # Extract additional attributes
                attributes = {}
                if "faceAttributes" in face_data:
                    attrs = face_data["faceAttributes"]
                    attributes = {
                        "age": attrs.get("age"),
                        "gender": attrs.get("gender"),
                        "smile": attrs.get("smile"),
                        "glasses": attrs.get("glasses"),
                        "facialHair": attrs.get("facialHair"),
                    }
                
                face_results.append(
                    FaceDetectionResult(
                        landmarks=landmarks,
                        bounding_box=bbox,
                        confidence=1.0,  # Azure Face API doesn't provide per-face confidence
                        emotions=emotions,
                        head_pose=head_pose,
                        attributes=attributes
                    )
                )
            
            return face_results
        
        except Exception as e:
            print(f"Azure Face API detection error: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Azure Face API is available and configured."""
        return self._available
    
    def get_name(self) -> str:
        """Get detector name."""
        return "azure_face_api"
    
    def close(self) -> None:
        """Clean up resources."""
        # Azure Face API service doesn't need cleanup
        pass
