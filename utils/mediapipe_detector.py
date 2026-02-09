"""
MediaPipe Face Detection Implementation

This module provides a MediaPipe-based implementation of the FaceDetectorInterface.
Uses multiple strategies for robust face detection:
1. Primary: FaceMesh in tracking mode (fast, continuous)
2. Fallback: FaceMesh in static mode (more reliable for new faces)
3. Last resort: Simple face detection for bounding box
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
    Includes fallback strategies for robust detection in various conditions.
    """
    
    def __init__(self, min_detection_confidence: float = 0.15, min_tracking_confidence: float = 0.15):
        """
        Initialize MediaPipe face detector with multiple detection strategies.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0-1). Lower = more permissive.
            min_tracking_confidence: Minimum confidence for face tracking (0-1)
        """
        # Use very low confidence for maximum permissiveness
        self._det_conf = max(0.01, min(0.99, float(min_detection_confidence)))
        self._track_conf = max(0.01, min(0.99, float(min_tracking_confidence)))
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        # Primary: tracking mode for continuous video (fast) — always created
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf
        )
        
        # Fallback models: created on first use to reduce memory and startup cost
        self._face_mesh_static = None
        self._face_detection = None
        
        self._consecutive_tracking_failures = 0
        self._available = True
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces using MediaPipe with multiple fallback strategies.
        
        Strategy:
        1. Try tracking mode (fast, works well once face is acquired)
        2. If fails, try static mode (better for initial detection)
        3. If still fails but simple detection finds a face, log for debugging
        
        Args:
            image: BGR image array
        
        Returns:
            List of FaceDetectionResult objects
        """
        if image is None or image.size == 0:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Strategy 1: Tracking mode (primary)
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            self._consecutive_tracking_failures = 0
            return self._extract_landmarks(results, width, height)
        
        # Strategy 2: Static mode fallback (more reliable for new/lost faces) — lazy init
        self._consecutive_tracking_failures += 1
        results_static = self._get_face_mesh_static().process(rgb_image)
        if results_static.multi_face_landmarks:
            # Reset tracking failures since we found a face
            self._consecutive_tracking_failures = 0
            return self._extract_landmarks(results_static, width, height)
        
        # Strategy 3: Simple face detection (just to verify face exists) — lazy init
        if self._consecutive_tracking_failures >= 5:
            simple_results = self._get_face_detection().process(rgb_image)
            if simple_results.detections:
                # Face exists but FaceMesh can't get landmarks
                # Try resetting the tracking-mode mesh
                self._reset_tracking_mesh()
                self._consecutive_tracking_failures = 0
        
        return []
    
    def _get_face_mesh_static(self):
        """Lazy init: create static FaceMesh only when tracking fails."""
        if self._face_mesh_static is None:
            self._face_mesh_static = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.05,
                min_tracking_confidence=0.05
            )
        return self._face_mesh_static
    
    def _get_face_detection(self):
        """Lazy init: create simple FaceDetection only when needed (5+ tracking failures)."""
        if self._face_detection is None:
            self._face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3
            )
        return self._face_detection
    
    def _extract_landmarks(self, results, width: int, height: int) -> List[FaceDetectionResult]:
        """Extract landmarks from MediaPipe FaceMesh results."""
        face_results = []
        
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
    
    def _reset_tracking_mesh(self) -> None:
        """Reset the tracking-mode FaceMesh to re-initialize detection."""
        try:
            self.face_mesh.close()
        except Exception:
            pass
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf
        )
    
    def is_available(self) -> bool:
        """Check if MediaPipe is available."""
        return self._available
    
    def get_name(self) -> str:
        """Get detector name."""
        return "mediapipe"
    
    def close(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            try:
                self.face_mesh.close()
            except Exception:
                pass
        if self._face_mesh_static is not None:
            try:
                self._face_mesh_static.close()
            except Exception:
                pass
        if self._face_detection is not None:
            try:
                self._face_detection.close()
            except Exception:
                pass
