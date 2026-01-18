"""
Azure Face API service module.

This module handles all interactions with Azure Face API for face detection,
facial landmark detection, and emotion/attribute analysis. It provides
an alternative to MediaPipe for face detection in the engagement system.
"""

import requests
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import config


class AzureFaceAPIService:
    """
    Service class for interacting with Azure Face API.
    
    Handles face detection, facial landmark detection, and emotion analysis
    for engagement state detection.
    """
    
    def __init__(self):
        """Initialize the Azure Face API client."""
        if not config.is_azure_face_api_enabled():
            raise ValueError(
                "Azure Face API is not configured. "
                "Please set AZURE_FACE_API_KEY and AZURE_FACE_API_ENDPOINT environment variables."
            )
        
        self.api_key = config.AZURE_FACE_API_KEY
        self.endpoint = config.AZURE_FACE_API_ENDPOINT.rstrip('/')
        self.region = config.AZURE_FACE_API_REGION
        
        # Face detection endpoint
        self.detect_url = f"{self.endpoint}/face/v1.0/detect"
        
        # Request headers
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream"
        }
    
    def detect_faces(
        self,
        image: np.ndarray,
        return_face_landmarks: bool = True,
        return_face_attributes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using Azure Face API.
        
        Args:
            image: BGR image array (OpenCV format)
            return_face_landmarks: Whether to return facial landmarks
            return_face_attributes: Whether to return face attributes (emotion, etc.)
        
        Returns:
            List of face detection results, each containing:
            - faceId: Unique face identifier
            - faceRectangle: Bounding box coordinates
            - faceLandmarks: Facial landmark coordinates (if requested)
            - faceAttributes: Face attributes like emotion, age, etc. (if requested)
        
        Raises:
            requests.RequestException: If the API call fails
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_data = buffer.tobytes()
        
        # Build request parameters
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "true" if return_face_landmarks else "false",
            "returnFaceAttributes": (
                "age,gender,emotion,headPose,smile,facialHair,glasses,"
                "hair,makeup,occlusion,accessories,blur,exposure,noise"
                if return_face_attributes else "false"
            )
        }
        
        try:
            response = requests.post(
                self.detect_url,
                headers=self.headers,
                params=params,
                data=image_data,
                timeout=5
            )
            response.raise_for_status()
            
            faces = response.json()
            return faces if isinstance(faces, list) else []
        
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Azure Face API request failed: {str(e)}"
            )
    
    def extract_landmarks_from_face(
        self,
        face_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from Azure Face API response.
        
        Azure Face API provides 27 facial landmarks. This method converts
        them to a format compatible with the engagement detection system.
        
        Args:
            face_data: Face detection result from Azure Face API
        
        Returns:
            numpy array of landmark coordinates (N, 2) or None if not available
        """
        if "faceLandmarks" not in face_data:
            return None
        
        landmarks = face_data["faceLandmarks"]
        
        # Azure Face API provides 27 landmarks with x, y coordinates
        # We'll extract all available landmarks
        landmark_points = []
        
        # Key landmarks in Azure Face API (27 total)
        landmark_keys = [
            "pupilLeft", "pupilRight",
            "noseTip", "mouthLeft", "mouthRight",
            "eyebrowLeftOuter", "eyebrowLeftInner",
            "eyeLeftOuter", "eyeLeftTop", "eyeLeftBottom", "eyeLeftInner",
            "eyebrowRightInner", "eyebrowRightOuter",
            "eyeRightInner", "eyeRightTop", "eyeRightBottom", "eyeRightOuter",
            "noseRootLeft", "noseRootRight",
            "noseLeftAlarTop", "noseRightAlarTop",
            "noseLeftAlarOutTip", "noseRightAlarOutTip",
            "upperLipTop", "upperLipBottom",
            "underLipTop", "underLipBottom"
        ]
        
        for key in landmark_keys:
            if key in landmarks:
                point = landmarks[key]
                landmark_points.append([point["x"], point["y"]])
        
        if not landmark_points:
            return None
        
        return np.array(landmark_points, dtype=np.float32)
    
    def extract_emotion_from_face(
        self,
        face_data: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Extract emotion scores from Azure Face API response.
        
        Args:
            face_data: Face detection result from Azure Face API
        
        Returns:
            Dictionary of emotion scores (anger, contempt, disgust, fear, happiness,
            neutral, sadness, surprise) or None if not available
        """
        if "faceAttributes" not in face_data:
            return None
        
        attributes = face_data["faceAttributes"]
        
        if "emotion" not in attributes:
            return None
        
        return attributes["emotion"]
    
    def extract_head_pose_from_face(
        self,
        face_data: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Extract head pose (pitch, yaw, roll) from Azure Face API response.
        
        Args:
            face_data: Face detection result from Azure Face API
        
        Returns:
            Dictionary with 'pitch', 'yaw', 'roll' angles in degrees, or None
        """
        if "faceAttributes" not in face_data:
            return None
        
        attributes = face_data["faceAttributes"]
        
        if "headPose" not in attributes:
            return None
        
        return attributes["headPose"]
    
    def get_face_rectangle(
        self,
        face_data: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract face bounding box from Azure Face API response.
        
        Args:
            face_data: Face detection result from Azure Face API
        
        Returns:
            Tuple of (left, top, width, height) or None if not available
        """
        if "faceRectangle" not in face_data:
            return None
        
        rect = face_data["faceRectangle"]
        return (rect["left"], rect["top"], rect["width"], rect["height"])


# Global service instance
azure_face_api_service: Optional[AzureFaceAPIService] = None


def get_azure_face_api_service() -> Optional[AzureFaceAPIService]:
    """
    Get or create the global Azure Face API service instance.
    
    Returns:
        AzureFaceAPIService instance if configured, None otherwise
    """
    global azure_face_api_service
    
    if azure_face_api_service is None:
        if config.is_azure_face_api_enabled():
            try:
                azure_face_api_service = AzureFaceAPIService()
            except Exception as e:
                print(f"Warning: Failed to initialize Azure Face API service: {e}")
                return None
    
    return azure_face_api_service
