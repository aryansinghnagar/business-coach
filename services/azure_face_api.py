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
        
        self.api_key = config.AZURE_FACE_API_KEY.strip() if config.AZURE_FACE_API_KEY else ""
        self.endpoint = config.AZURE_FACE_API_ENDPOINT.strip().rstrip('/') if config.AZURE_FACE_API_ENDPOINT else ""
        self.region = config.AZURE_FACE_API_REGION
        
        # Validate endpoint format
        if not self.endpoint:
            raise ValueError("Azure Face API endpoint is empty")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError(f"Invalid Azure Face API endpoint format: {self.endpoint}. Must start with http:// or https://")
        
        if not self.api_key:
            raise ValueError("Azure Face API key is empty or invalid")
        
        # Face detection endpoint
        # Format: {endpoint}/face/{apiVersion}/detect
        # Azure supports v1.0 and v1.2 - v1.0 is more widely available
        self.api_version = "v1.0"
        
        # Handle case where endpoint might already include /face
        if "/face/" in self.endpoint.lower():
            # Endpoint already includes /face/, just append version and /detect
            if self.endpoint.endswith("/face") or self.endpoint.endswith("/face/"):
                self.detect_url = f"{self.endpoint}/{self.api_version}/detect"
            else:
                # Extract base endpoint and reconstruct
                base_endpoint = self.endpoint.split("/face")[0].rstrip('/')
                self.detect_url = f"{base_endpoint}/face/{self.api_version}/detect"
        else:
            # Standard format: endpoint/face/v1.0/detect
            self.detect_url = f"{self.endpoint}/face/{self.api_version}/detect"
        
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
        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image format: expected BGR image with shape (H, W, 3), got {image.shape}")
        
        # Azure Face API requirements:
        # - Image size: 36x36 to 4096x4096 pixels
        # - File size: < 6MB
        h, w = image.shape[:2]
        if h < 36 or w < 36:
            raise ValueError(f"Image too small: {w}x{h}. Minimum size is 36x36 pixels")
        if h > 4096 or w > 4096:
            # Resize if too large
            scale = min(4096 / w, 4096 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            h, w = new_h, new_w
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode image to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, buffer = cv2.imencode('.jpg', rgb_image, encode_params)
        
        if not success or buffer is None:
            raise ValueError("Failed to encode image to JPEG")
        
        image_data = buffer.tobytes()
        
        # Check file size (Azure limit: 6MB)
        if len(image_data) > 6 * 1024 * 1024:
            # Reduce quality and retry
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            success, buffer = cv2.imencode('.jpg', rgb_image, encode_params)
            if not success or buffer is None:
                raise ValueError("Failed to encode image to JPEG (even with reduced quality)")
            image_data = buffer.tobytes()
            if len(image_data) > 6 * 1024 * 1024:
                raise ValueError(f"Image file size too large: {len(image_data)} bytes. Maximum is 6MB")
        
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
                timeout=10  # Increased timeout for reliability
            )
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"Azure Face API returned status {response.status_code}"
                try:
                    error_body = response.json()
                    if isinstance(error_body, dict) and "error" in error_body:
                        error_info = error_body["error"]
                        error_msg += f": {error_info.get('message', 'Unknown error')}"
                        if "code" in error_info:
                            error_msg += f" (code: {error_info['code']})"
                except:
                    error_msg += f": {response.text[:200]}"
                
                raise requests.RequestException(error_msg)
            
            faces = response.json()
            if not isinstance(faces, list):
                raise ValueError(f"Unexpected response format: expected list, got {type(faces)}")
            
            return faces
        
        except requests.Timeout:
            raise requests.RequestException(
                f"Azure Face API request timed out after 10 seconds. "
                f"Endpoint: {self.detect_url}"
            )
        except requests.ConnectionError as e:
            raise requests.RequestException(
                f"Azure Face API connection error: {str(e)}. "
                f"Check network connectivity and endpoint: {self.endpoint}"
            )
        except requests.RequestException as e:
            # Re-raise with more context
            raise requests.RequestException(
                f"Azure Face API request failed: {str(e)}. "
                f"Endpoint: {self.detect_url}, Image size: {w}x{h}"
            )
        except Exception as e:
            raise ValueError(
                f"Unexpected error in Azure Face API detection: {str(e)}"
            )
    
    def test_connection(self) -> bool:
        """
        Test Azure Face API connection with a simple request.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create a minimal test image (1x1 pixel, but Azure requires at least 36x36)
            # So we'll create a 36x36 test image
            test_image = np.ones((36, 36, 3), dtype=np.uint8) * 128  # Gray image
            
            # Try to detect faces (will likely return empty, but tests connectivity)
            result = self.detect_faces(test_image, return_face_landmarks=False, return_face_attributes=False)
            return True  # If we get here, connection works (even if no faces found)
        except requests.RequestException as e:
            print(f"Azure Face API connection test failed: {e}")
            return False
        except Exception as e:
            print(f"Azure Face API connection test error: {e}")
            return False
    
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
        
        if not isinstance(landmarks, dict):
            return None
        
        # Azure Face API provides 27 landmarks with x, y coordinates
        # We'll extract all available landmarks
        landmark_points = []
        
        # Key landmarks in Azure Face API (27 total)
        # Order matters for compatibility with MediaPipe mapping
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
                if isinstance(point, dict) and "x" in point and "y" in point:
                    try:
                        x = float(point["x"])
                        y = float(point["y"])
                        landmark_points.append([x, y])
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid landmark point for {key}: {point}, error: {e}")
                        continue
        
        if len(landmark_points) < 10:  # Need at least 10 landmarks for basic detection
            print(f"Warning: Insufficient landmarks extracted: {len(landmark_points)}/27")
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
                print(f"Azure Face API service initialized successfully. Endpoint: {azure_face_api_service.endpoint}")
            except ValueError as e:
                print(f"Error: Azure Face API configuration issue: {e}")
                return None
            except Exception as e:
                import traceback
                print(f"Error: Failed to initialize Azure Face API service: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                return None
        else:
            print("Warning: Azure Face API is not enabled in configuration")
            return None
    
    return azure_face_api_service
