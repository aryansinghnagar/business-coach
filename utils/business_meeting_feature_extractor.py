"""
Business Meeting Feature Extractor Module

This module provides advanced feature extraction specifically optimized for
business meeting engagement detection. It extracts complex blendshape features
that are most relevant to understanding participant engagement, attention,
and emotional state in professional meeting contexts.

Features extracted include:
- Attention and focus indicators (eye tracking, gaze direction)
- Emotional engagement signals (micro-expressions, facial activity)
- Participation readiness (mouth activity, head orientation)
- Professional demeanor indicators (posture, symmetry, stability)
- Cognitive load indicators (facial tension, expression complexity)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from utils.face_detection_interface import FaceDetectionResult


class BusinessMeetingFeatureExtractor:
    """
    Advanced feature extractor optimized for business meeting engagement analysis.
    
    This class extracts 100+ complex features from facial landmarks and blendshapes
    that are specifically relevant to understanding engagement in professional
    meeting contexts.
    """
    
    # MediaPipe landmark indices for key facial regions
    # These are optimized for engagement detection in business meetings
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    EYEBROW_INDICES = [107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276]
    FACE_OUTLINE_INDICES = [10, 151, 9, 175, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    NOSE_INDICES = [4, 6, 19, 20, 51, 94, 125, 141, 168, 197, 220, 237, 242, 250, 290, 305, 326, 327, 358, 360, 392, 393, 401, 412, 417, 420, 429, 456]
    
    def __init__(self):
        """Initialize the business meeting feature extractor."""
        self.feature_cache = {}
    
    def extract_features(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int, int],
        face_result: Optional[FaceDetectionResult] = None
    ) -> np.ndarray:
        """
        Extract comprehensive business-meeting focused features.
        
        Args:
            landmarks: numpy array of landmarks (N, 2) or (N, 3)
            frame_shape: Shape of the frame (height, width, channels)
            face_result: Optional FaceDetectionResult for additional data
        
        Returns:
            numpy array of 100 feature values optimized for business meetings
        """
        height, width = frame_shape[:2]
        
        # Normalize landmarks
        landmarks = self._normalize_landmarks(landmarks, width, height)
        num_landmarks = landmarks.shape[0]
        
        # Extract features based on detection method
        if num_landmarks >= 400:  # MediaPipe (468 landmarks)
            features = self._extract_mediapipe_features(landmarks, face_result, width, height)
        else:  # Azure Face API (27 landmarks)
            features = self._extract_azure_face_features(landmarks, face_result, width, height)
        
        # Ensure exactly 100 features
        while len(features) < 100:
            features.append(0.0)
        
        features_array = np.array(features[:100], dtype=np.float32)
        
        # Validate features are finite and reasonable
        if not np.all(np.isfinite(features_array)):
            print("Warning: Non-finite features detected, replacing with zeros")
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme values to reasonable ranges
        features_array = np.clip(features_array, -1000.0, 1000.0)
        
        return features_array
    
    def _normalize_landmarks(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """Normalize landmarks to pixel coordinates."""
        # Ensure 3D coordinates
        if landmarks.shape[1] == 2:
            z_coords = np.zeros((landmarks.shape[0], 1), dtype=landmarks.dtype)
            landmarks = np.hstack([landmarks, z_coords])
        
        # Convert normalized coordinates to pixels if needed
        if np.max(landmarks[:, :2]) <= 1.0:
            landmarks[:, 0] *= width
            landmarks[:, 1] *= height
            if landmarks.shape[1] > 2:
                landmarks[:, 2] *= width
        
        return landmarks
    
    def _extract_mediapipe_features(
        self,
        landmarks: np.ndarray,
        face_result: Optional[FaceDetectionResult],
        width: int,
        height: int
    ) -> List[float]:
        """Extract features from MediaPipe landmarks (468 points)."""
        features = []
        
        # ========================================================================
        # ATTENTION & FOCUS FEATURES (Features 0-19)
        # ========================================================================
        attention_features = self._extract_attention_features(landmarks, width, height)
        features.extend(attention_features)
        
        # ========================================================================
        # EYE CONTACT & GAZE FEATURES (Features 20-29)
        # ========================================================================
        gaze_features = self._extract_gaze_features(landmarks, width, height)
        features.extend(gaze_features)
        
        # ========================================================================
        # EMOTIONAL ENGAGEMENT FEATURES (Features 30-44)
        # ========================================================================
        emotion_features = self._extract_emotional_features(landmarks, face_result)
        features.extend(emotion_features)
        
        # ========================================================================
        # PARTICIPATION READINESS FEATURES (Features 45-59)
        # ========================================================================
        participation_features = self._extract_participation_features(landmarks)
        features.extend(participation_features)
        
        # ========================================================================
        # PROFESSIONAL DEMEANOR FEATURES (Features 60-74)
        # ========================================================================
        demeanor_features = self._extract_demeanor_features(landmarks)
        features.extend(demeanor_features)
        
        # ========================================================================
        # COGNITIVE LOAD & STRESS FEATURES (Features 75-89)
        # ========================================================================
        cognitive_features = self._extract_cognitive_features(landmarks)
        features.extend(cognitive_features)
        
        # ========================================================================
        # TEMPORAL STABILITY FEATURES (Features 90-99)
        # ========================================================================
        stability_features = self._extract_stability_features(landmarks)
        features.extend(stability_features)
        
        return features
    
    def _extract_attention_features(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> List[float]:
        """Extract attention and focus indicators (20 features)."""
        features = []
        
        # Safely extract eye landmarks
        try:
            left_eye = landmarks[self.LEFT_EYE_INDICES] if len(landmarks) > max(self.LEFT_EYE_INDICES) else landmarks[:min(16, len(landmarks))]
            right_eye = landmarks[self.RIGHT_EYE_INDICES] if len(landmarks) > max(self.RIGHT_EYE_INDICES) else landmarks[:min(16, len(landmarks))]
        except (IndexError, ValueError) as e:
            print(f"Warning: Error extracting eye landmarks: {e}")
            # Fallback: use first available landmarks
            left_eye = landmarks[:min(16, len(landmarks))] if len(landmarks) >= 8 else landmarks
            right_eye = landmarks[:min(16, len(landmarks))] if len(landmarks) >= 8 else landmarks
        
        # Eye Aspect Ratio (EAR) - indicator of eye openness
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0 if (left_ear > 0 or right_ear > 0) else 0.2  # Default if both zero
        
        # Ensure EAR values are reasonable
        left_ear = max(0.05, min(0.5, left_ear)) if left_ear > 0 else 0.2
        right_ear = max(0.05, min(0.5, right_ear)) if right_ear > 0 else 0.2
        avg_ear = max(0.1, min(0.4, avg_ear))
        
        features.extend([left_ear, right_ear, avg_ear])
        
        # Eye area (engagement indicator)
        left_eye_area = self._calculate_polygon_area(left_eye[:, :2])
        right_eye_area = self._calculate_polygon_area(right_eye[:, :2])
        total_eye_area = left_eye_area + right_eye_area
        features.extend([left_eye_area, right_eye_area, total_eye_area])
        
        # Eye symmetry (both eyes equally open = focused attention)
        eye_symmetry = 1.0 - abs(left_ear - right_ear) / max(avg_ear, 1e-6)
        features.append(eye_symmetry)
        
        # Eye center positions
        left_eye_center = np.mean(left_eye[:, :2], axis=0)
        right_eye_center = np.mean(right_eye[:, :2], axis=0)
        eye_center_distance = np.linalg.norm(left_eye_center - right_eye_center)
        features.extend([left_eye_center[0], left_eye_center[1], 
                        right_eye_center[0], right_eye_center[1], eye_center_distance])
        
        # Eye variance (stability indicator)
        left_eye_variance = np.var(left_eye[:, :2], axis=0)
        right_eye_variance = np.var(right_eye[:, :2], axis=0)
        features.extend([left_eye_variance[0], left_eye_variance[1],
                        right_eye_variance[0], right_eye_variance[1]])
        
        # Blink detection (low EAR = potential blink)
        blink_indicator = 1.0 if avg_ear > 0.2 else 0.0
        features.append(blink_indicator)
        
        return features
    
    def _extract_gaze_features(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> List[float]:
        """Extract gaze direction and eye contact features (10 features)."""
        features = []
        
        frame_center = np.array([width / 2, height / 2])
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        # Eye center positions
        left_eye_center = np.mean(left_eye[:, :2], axis=0)
        right_eye_center = np.mean(right_eye[:, :2], axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Distance from frame center (gaze direction)
        distance_from_center = np.linalg.norm(eye_center - frame_center)
        max_distance = np.sqrt(width**2 + height**2) / 2
        normalized_gaze_distance = min(1.0, distance_from_center / max_distance)
        features.append(normalized_gaze_distance)
        
        # Gaze angle (direction of looking)
        gaze_vector = eye_center - frame_center
        gaze_angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi
        features.append(gaze_angle)
        
        # Head orientation (affects perceived eye contact)
        nose_tip = landmarks[4]
        chin = landmarks[175]
        head_orientation = np.arctan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0]) * 180 / np.pi
        features.append(head_orientation)
        
        # Eye alignment (both eyes looking in same direction)
        eye_alignment = np.linalg.norm(left_eye_center - right_eye_center)
        features.append(eye_alignment)
        
        # Combined eye contact score
        eye_contact_score = (1.0 - normalized_gaze_distance) * 100
        features.append(eye_contact_score)
        
        # Additional gaze stability features
        features.extend([0.0, 0.0, 0.0, 0.0])  # Placeholder for temporal features
        
        return features
    
    def _extract_emotional_features(
        self,
        landmarks: np.ndarray,
        face_result: Optional[FaceDetectionResult]
    ) -> List[float]:
        """Extract emotional engagement features (15 features)."""
        features = []
        
        # Mouth features (smile, engagement)
        mouth = landmarks[self.MOUTH_INDICES]
        mouth_center = np.mean(mouth[:, :2], axis=0)
        mouth_width = np.max(mouth[:, 0]) - np.min(mouth[:, 0])
        mouth_height = np.max(mouth[:, 1]) - np.min(mouth[:, 1])
        mouth_area = self._calculate_polygon_area(mouth[:, :2])
        features.extend([mouth_center[0], mouth_center[1], mouth_width, mouth_height, mouth_area])
        
        # Mouth corners (smile indicator)
        left_mouth_corner = mouth[0]
        right_mouth_corner = mouth[6]
        mouth_corner_distance = np.linalg.norm(left_mouth_corner[:2] - right_mouth_corner[:2])
        mouth_corner_asymmetry = abs(left_mouth_corner[1] - right_mouth_corner[1])
        features.extend([mouth_corner_distance, mouth_corner_asymmetry])
        
        # Eyebrow position (emotional expression)
        eyebrows = landmarks[self.EYEBROW_INDICES]
        left_eyebrow = eyebrows[:6]
        right_eyebrow = eyebrows[6:]
        left_eyebrow_center = np.mean(left_eyebrow[:, :2], axis=0)
        right_eyebrow_center = np.mean(right_eyebrow[:, :2], axis=0)
        
        # Eyebrow height relative to eyes
        left_eye_center = np.mean(landmarks[self.LEFT_EYE_INDICES][:, :2], axis=0)
        right_eye_center = np.mean(landmarks[self.RIGHT_EYE_INDICES][:, :2], axis=0)
        left_eyebrow_height = left_eyebrow_center[1] - left_eye_center[1]
        right_eyebrow_height = right_eyebrow_center[1] - right_eye_center[1]
        features.extend([left_eyebrow_height, right_eyebrow_height])
        
        # Emotion scores from Azure Face API (if available)
        if face_result and face_result.emotions:
            emotions = face_result.emotions
            features.extend([
                emotions.get('happiness', 0.0),
                emotions.get('neutral', 0.0),
                emotions.get('sadness', 0.0),
                emotions.get('anger', 0.0),
                emotions.get('surprise', 0.0)
            ])
        else:
            # Estimate from facial features
            smile_estimate = max(0.0, min(1.0, (mouth_corner_distance - 50) / 50))
            features.extend([smile_estimate, 0.5, 0.0, 0.0, 0.0])
        
        return features
    
    def _extract_participation_features(
        self,
        landmarks: np.ndarray
    ) -> List[float]:
        """Extract participation readiness features (15 features)."""
        features = []
        
        # Mouth activity (speaking readiness)
        mouth = landmarks[self.MOUTH_INDICES]
        mouth_openness = np.max(mouth[:, 1]) - np.min(mouth[:, 1])
        mouth_width = np.max(mouth[:, 0]) - np.min(mouth[:, 0])
        mouth_aspect_ratio = mouth_openness / max(mouth_width, 1e-6)
        features.extend([mouth_openness, mouth_width, mouth_aspect_ratio])
        
        # Head orientation (facing forward = ready to participate)
        nose_tip = landmarks[4]
        chin = landmarks[175]
        left_face = landmarks[234]
        right_face = landmarks[454]
        
        head_yaw = np.arctan2((left_face[0] - right_face[0]) / 2, 
                             abs(left_face[2] - right_face[2])) * 180 / np.pi
        head_pitch = np.arctan2(chin[1] - nose_tip[1], 
                               abs(chin[2] - nose_tip[2])) * 180 / np.pi
        features.extend([head_yaw, head_pitch])
        
        # Face orientation towards camera
        face_center = np.mean(landmarks[self.FACE_OUTLINE_INDICES][:, :2], axis=0)
        face_orientation_score = 1.0 - min(1.0, abs(head_yaw) / 45.0)
        features.append(face_orientation_score)
        
        # Additional participation indicators
        features.extend([0.0] * 9)  # Placeholder for future features
        
        return features
    
    def _extract_demeanor_features(
        self,
        landmarks: np.ndarray
    ) -> List[float]:
        """Extract professional demeanor features (15 features)."""
        features = []
        
        # Face symmetry (professional appearance)
        face_outline = landmarks[self.FACE_OUTLINE_INDICES]
        face_center = np.mean(face_outline[:, :2], axis=0)
        left_face = face_outline[face_outline[:, 0] < face_center[0]]
        right_face = face_outline[face_outline[:, 0] > face_center[0]]
        
        if len(left_face) > 0 and len(right_face) > 0:
            left_center = np.mean(left_face[:, :2], axis=0)
            right_center = np.mean(right_face[:, :2], axis=0)
            face_symmetry = 1.0 - abs(left_center[1] - right_center[1]) / max(
                np.max(face_outline[:, 1]) - np.min(face_outline[:, 1]), 1e-6)
        else:
            face_symmetry = 0.5
        features.append(face_symmetry)
        
        # Face dimensions (posture indicator)
        face_width = np.max(face_outline[:, 0]) - np.min(face_outline[:, 0])
        face_height = np.max(face_outline[:, 1]) - np.min(face_outline[:, 1])
        face_aspect_ratio = face_width / max(face_height, 1e-6)
        features.extend([face_width, face_height, face_aspect_ratio])
        
        # Head stability (professional composure)
        head_stability = 1.0 - min(1.0, np.var(face_outline[:, :2]))
        features.append(head_stability)
        
        # Additional demeanor features
        features.extend([0.0] * 10)  # Placeholder for future features
        
        return features
    
    def _extract_cognitive_features(
        self,
        landmarks: np.ndarray
    ) -> List[float]:
        """Extract cognitive load and stress indicators (15 features)."""
        features = []
        
        # Facial tension (eyebrow position, mouth tightness)
        eyebrows = landmarks[self.EYEBROW_INDICES]
        eyebrow_variance = np.var(eyebrows[:, :2])
        features.append(eyebrow_variance)
        
        # Eye strain (eye openness variance)
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        eye_variance = np.var(np.vstack([left_eye[:, :2], right_eye[:, :2]]))
        features.append(eye_variance)
        
        # Overall facial activity (cognitive engagement)
        key_points = np.vstack([
            left_eye[:, :2],
            right_eye[:, :2],
            landmarks[self.MOUTH_INDICES][:, :2],
            eyebrows[:, :2]
        ])
        facial_activity = np.var(key_points)
        features.append(facial_activity)
        
        # Additional cognitive load indicators
        features.extend([0.0] * 12)  # Placeholder for future features
        
        return features
    
    def _extract_stability_features(
        self,
        landmarks: np.ndarray
    ) -> List[float]:
        """Extract temporal stability features (10 features)."""
        features = []
        
        # Overall landmark variance (movement indicator)
        x_variance = np.var(landmarks[:, 0])
        y_variance = np.var(landmarks[:, 1])
        if landmarks.shape[1] > 2:
            z_variance = np.var(landmarks[:, 2])
        else:
            z_variance = 0.0
        features.extend([x_variance, y_variance, z_variance])
        
        # Key region stability
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        mouth = landmarks[self.MOUTH_INDICES]
        
        left_eye_stability = 1.0 / (1.0 + np.var(left_eye[:, :2]))
        right_eye_stability = 1.0 / (1.0 + np.var(right_eye[:, :2]))
        mouth_stability = 1.0 / (1.0 + np.var(mouth[:, :2]))
        features.extend([left_eye_stability, right_eye_stability, mouth_stability])
        
        # Overall stability score
        overall_stability = (left_eye_stability + right_eye_stability + mouth_stability) / 3
        features.append(overall_stability)
        
        # Additional stability features
        features.extend([0.0] * 3)  # Placeholder for temporal features
        
        return features
    
    def _extract_azure_face_features(
        self,
        landmarks: np.ndarray,
        face_result: Optional[FaceDetectionResult],
        width: int,
        height: int
    ) -> List[float]:
        """Extract features from Azure Face API landmarks (27 points)."""
        features = []
        
        # Basic geometric features
        face_center = np.mean(landmarks[:, :2], axis=0)
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        features.extend([face_center[0], face_center[1], face_width, face_height])
        
        # Face outline features
        face_outline = landmarks[:17] if len(landmarks) >= 17 else landmarks
        outline_center = np.mean(face_outline[:, :2], axis=0)
        features.extend([outline_center[0], outline_center[1]])
        
        # Eyebrow features
        if len(landmarks) >= 27:
            right_eyebrow = landmarks[17:22]
            left_eyebrow = landmarks[22:27]
            right_eyebrow_center = np.mean(right_eyebrow[:, :2], axis=0)
            left_eyebrow_center = np.mean(left_eyebrow[:, :2], axis=0)
            features.extend([
                right_eyebrow_center[0], right_eyebrow_center[1],
                left_eyebrow_center[0], left_eyebrow_center[1]
            ])
        else:
            features.extend([0.0] * 4)
        
        # Head pose
        head_pose = face_result.head_pose if face_result else None
        if head_pose:
            features.extend([
                head_pose.get('pitch', 0.0),
                head_pose.get('yaw', 0.0),
                head_pose.get('roll', 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Emotion features
        if face_result and face_result.emotions:
            emotions = face_result.emotions
            features.extend([
                emotions.get('happiness', 0.0),
                emotions.get('neutral', 0.0),
                emotions.get('sadness', 0.0),
                emotions.get('anger', 0.0),
                emotions.get('surprise', 0.0),
                emotions.get('fear', 0.0),
                emotions.get('contempt', 0.0),
                emotions.get('disgust', 0.0)
            ])
        else:
            features.extend([0.0] * 8)
        
        # Fill remaining features
        remaining = 100 - len(features)
        features.extend([0.0] * remaining)
        
        return features
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Uses the standard 6-point method:
        - Points 0, 3: outer and inner eye corners (horizontal)
        - Points 1, 5: top and bottom of eye (vertical)
        - Points 2, 4: additional vertical points
        """
        if len(eye_landmarks) < 6:
            # Fallback: use first 6 points if available
            if len(eye_landmarks) >= 4:
                # Simplified EAR using 4 points
                top = np.mean(eye_landmarks[:len(eye_landmarks)//2, 1])
                bottom = np.mean(eye_landmarks[len(eye_landmarks)//2:, 1])
                left = np.min(eye_landmarks[:, 0])
                right = np.max(eye_landmarks[:, 0])
                
                vertical = abs(top - bottom)
                horizontal = abs(right - left)
                
                if horizontal > 1e-6:
                    return vertical / horizontal
            return 0.2  # Default EAR for closed/unknown state
        
        # Standard 6-point EAR calculation
        # For MediaPipe, we need to find the appropriate points
        # Use outer corners and vertical midpoints
        try:
            # Get bounding box of eye
            x_coords = eye_landmarks[:, 0]
            y_coords = eye_landmarks[:, 1]
            
            left_x = np.min(x_coords)
            right_x = np.max(x_coords)
            top_y = np.min(y_coords)
            bottom_y = np.max(y_coords)
            
            # Vertical distance (eye height)
            vertical = abs(bottom_y - top_y)
            
            # Horizontal distance (eye width)
            horizontal = abs(right_x - left_x)
            
            # EAR formula: height / width
            if horizontal > 1e-6:
                ear = vertical / horizontal
                # Clamp to reasonable range (typical EAR: 0.15-0.4)
                ear = max(0.05, min(0.5, ear))
                return float(ear)
            else:
                return 0.2  # Default if horizontal is too small
        except Exception as e:
            print(f"Warning: Error calculating EAR: {e}")
            return 0.2  # Safe default
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
