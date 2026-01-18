"""
Engagement Scorer Module

This module provides engagement scoring based on facial features extracted from
face detection (MediaPipe or Azure Face API). It computes multiple metrics and
combines them into an overall engagement score (0-100).

The scoring system considers:
- Attention level (eye openness, focus)
- Eye contact (gaze direction)
- Facial expressiveness (micro-expressions, muscle activity)
- Head movement (stability vs. excessive movement)
- Facial symmetry (indicator of engagement vs. distraction)
- Mouth activity (speaking, smiling, etc.)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EngagementMetrics:
    """
    Detailed engagement metrics breakdown.
    
    Each metric is normalized to 0-100 scale where:
    - 0-30: Low engagement
    - 30-60: Medium engagement
    - 60-100: High engagement
    """
    attention: float = 0.0  # Overall attention level (0-100)
    eye_contact: float = 0.0  # Eye contact quality (0-100)
    facial_expressiveness: float = 0.0  # Facial expression activity (0-100)
    head_movement: float = 0.0  # Head movement score (0-100, higher = more stable)
    symmetry: float = 0.0  # Facial symmetry (0-100, higher = more symmetric)
    mouth_activity: float = 0.0  # Mouth movement/activity (0-100)


class EngagementScorer:
    """
    Computes engagement scores from facial features.
    
    This class analyzes blendshape features and face landmarks to compute
    various engagement metrics and an overall engagement score.
    
    Usage:
        scorer = EngagementScorer()
        metrics = scorer.compute_metrics(blendshape_features, face_landmarks, frame_shape)
        score = scorer.calculate_score(metrics)
    """
    
    def __init__(self):
        """Initialize the engagement scorer with default weights."""
        # Weights for combining metrics into overall score
        # These weights can be adjusted based on domain knowledge or calibration
        self.weights = {
            'attention': 0.25,
            'eye_contact': 0.20,
            'facial_expressiveness': 0.15,
            'head_movement': 0.15,
            'symmetry': 0.10,
            'mouth_activity': 0.15
        }
        
        # Normalization ranges for different features
        # These are based on typical values from MediaPipe landmarks
        self.normalization_ranges = {
            'eye_aspect_ratio': (0.15, 0.4),  # Typical EAR range
            'mouth_aspect_ratio': (0.1, 0.5),
            'head_angle': (-30, 30),  # Degrees
        }
    
    def compute_metrics(
        self,
        blendshape_features: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: tuple
    ) -> EngagementMetrics:
        """
        Compute detailed engagement metrics from features.
        
        Args:
            blendshape_features: Array of 100 blendshape feature values
            landmarks: numpy array of landmarks (N, 2) or (N, 3)
            frame_shape: Shape of the frame (height, width, channels)
        
        Returns:
            EngagementMetrics object with computed values
        """
        height, width = frame_shape[:2]
        
        # Ensure landmarks are in correct format
        if landmarks.shape[1] == 2:
            z_coords = np.zeros((landmarks.shape[0], 1), dtype=landmarks.dtype)
            landmarks = np.hstack([landmarks, z_coords])
        
        # Normalize if needed (MediaPipe provides normalized coordinates)
        if np.max(landmarks[:, :2]) <= 1.0:
            landmarks[:, 0] *= width
            landmarks[:, 1] *= height
            if landmarks.shape[1] > 2:
                landmarks[:, 2] *= width
        
        # Compute individual metrics
        attention = self._compute_attention(blendshape_features, landmarks)
        eye_contact = self._compute_eye_contact(blendshape_features, landmarks, frame_shape)
        facial_expressiveness = self._compute_facial_expressiveness(blendshape_features)
        head_movement = self._compute_head_movement_score(blendshape_features, landmarks)
        symmetry = self._compute_symmetry(blendshape_features, landmarks)
        mouth_activity = self._compute_mouth_activity(blendshape_features, landmarks)
        
        return EngagementMetrics(
            attention=attention,
            eye_contact=eye_contact,
            facial_expressiveness=facial_expressiveness,
            head_movement=head_movement,
            symmetry=symmetry,
            mouth_activity=mouth_activity
        )
    
    def calculate_score(self, metrics: EngagementMetrics) -> float:
        """
        Calculate overall engagement score from metrics.
        
        Args:
            metrics: EngagementMetrics object
        
        Returns:
            Overall engagement score (0-100)
        """
        # Validate metrics are finite
        attention = metrics.attention if np.isfinite(metrics.attention) else 50.0
        eye_contact = metrics.eye_contact if np.isfinite(metrics.eye_contact) else 50.0
        facial_expressiveness = metrics.facial_expressiveness if np.isfinite(metrics.facial_expressiveness) else 50.0
        head_movement = metrics.head_movement if np.isfinite(metrics.head_movement) else 50.0
        symmetry = metrics.symmetry if np.isfinite(metrics.symmetry) else 50.0
        mouth_activity = metrics.mouth_activity if np.isfinite(metrics.mouth_activity) else 50.0
        
        score = (
            attention * self.weights['attention'] +
            eye_contact * self.weights['eye_contact'] +
            facial_expressiveness * self.weights['facial_expressiveness'] +
            head_movement * self.weights['head_movement'] +
            symmetry * self.weights['symmetry'] +
            mouth_activity * self.weights['mouth_activity']
        )
        
        # Ensure score is within valid range and is finite
        score = max(0.0, min(100.0, score))
        if not np.isfinite(score):
            print(f"Warning: Calculated invalid score from metrics, using fallback")
            return 50.0
        
        return float(score)
    
    def _compute_attention(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Compute attention level based on eye openness and focus.
        
        Higher attention = eyes open, focused forward, minimal blinking.
        
        Args:
            features: Blendshape features array
            landmarks: Face landmarks array
        
        Returns:
            Attention score (0-100)
        """
        # Features 0-2 contain eye aspect ratios
        if len(features) < 3:
            return 50.0
        
        left_ear = features[0]
        right_ear = features[1]
        avg_ear = features[2]
        
        # Normalize EAR to 0-100 scale
        # Optimal EAR is around 0.25-0.3 for engaged attention
        ear_min, ear_max = self.normalization_ranges['eye_aspect_ratio']
        
        # Map EAR to attention score
        # Too low (< 0.15) = eyes closed or drowsy
        # Too high (> 0.4) = wide-eyed surprise or distraction
        # Optimal (0.25-0.3) = focused attention
        
        # Ensure avg_ear is in reasonable range
        if avg_ear < 0.01:
            # Very low EAR - likely eyes closed or calculation error
            attention = 10.0
        elif avg_ear < ear_min:
            # Low but not zero - map linearly
            attention = 20.0 + (avg_ear / ear_min) * 20.0  # 20-40 range
        elif avg_ear > 0.4:
            # Very wide eyes - might indicate surprise/distraction
            attention = max(50.0, 80.0 - (avg_ear - 0.4) * 50.0)
        elif avg_ear > 0.35:
            # Slightly wide
            attention = 70.0 - (avg_ear - 0.35) * 40.0  # 70-66 range
        else:
            # Optimal range (0.15-0.35)
            # Map to 40-80 range for good engagement
            if avg_ear <= 0.3:
                attention = 40.0 + (avg_ear - ear_min) / (0.3 - ear_min) * 40.0  # 40-80
            else:
                attention = 80.0 - (avg_ear - 0.3) / (0.35 - 0.3) * 10.0  # 80-70
        
        # Bonus for symmetric eye openness (both eyes equally open)
        if avg_ear > 1e-6:
            eye_symmetry = 1.0 - abs(left_ear - right_ear) / avg_ear
            eye_symmetry = max(0.0, min(1.0, eye_symmetry))  # Clamp to 0-1
            attention += eye_symmetry * 10
        else:
            # If eyes are closed, no symmetry bonus
            pass
        
        return max(0.0, min(100.0, attention))
    
    def _compute_eye_contact(
        self,
        features: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: tuple
    ) -> float:
        """
        Compute eye contact quality based on gaze direction.
        
        Direct eye contact = looking at camera/center.
        Looking away = reduced engagement.
        
        Args:
            features: Blendshape features array
            landmarks: Face landmarks array
            frame_shape: Frame dimensions
        
        Returns:
            Eye contact score (0-100)
        """
        height, width = frame_shape[:2]
        frame_center = np.array([width / 2, height / 2])
        
        # Get eye center positions
        # Left eye center (approximate)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        if len(landmarks) < max(max(left_eye_indices), max(right_eye_indices)) + 1:
            return 50.0
        
        left_eye_center = np.mean(landmarks[left_eye_indices], axis=0)[:2]
        right_eye_center = np.mean(landmarks[right_eye_indices], axis=0)[:2]
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Calculate distance from frame center (normalized)
        distance_from_center = np.linalg.norm(eye_center - frame_center)
        max_distance = np.sqrt(width**2 + height**2) / 2
        
        # Score decreases as distance from center increases
        normalized_distance = min(1.0, distance_from_center / max_distance)
        eye_contact = (1.0 - normalized_distance) * 100
        
        # Consider head yaw (feature 33-35 contain head angles)
        if len(features) > 35:
            head_yaw = features[33]  # Yaw angle
            # Penalize if head is turned away significantly
            # Normalize yaw (assuming radians, typical range -0.5 to 0.5)
            yaw_penalty = min(1.0, abs(head_yaw) / 0.5)  # Clamp to 0-1
            eye_contact *= (1.0 - yaw_penalty * 0.3)
        
        # Ensure minimum eye contact score (even if looking away, still some engagement)
        eye_contact = max(20.0, eye_contact)  # Minimum 20 for face detected
        
        return max(0.0, min(100.0, eye_contact))
    
    def _compute_facial_expressiveness(
        self,
        features: np.ndarray
    ) -> float:
        """
        Compute facial expressiveness based on feature variance.
        
        More expressive faces show more variation in blendshape features.
        However, too much variation might indicate distraction.
        
        Args:
            features: Blendshape features array
        
        Returns:
            Expressiveness score (0-100)
        """
        if len(features) < 10:
            return 50.0
        
        # Compute variance of key expressive features (mouth, eyebrows, etc.)
        # Use features that indicate expression changes
        expressive_feature_indices = list(range(16, 30)) + list(range(45, 60))  # Mouth and participation features
        
        if len(features) > max(expressive_feature_indices):
            expressive_features = features[expressive_feature_indices]
            # Remove zeros to get meaningful variance
            non_zero_features = expressive_features[expressive_features != 0]
            if len(non_zero_features) > 0:
                avg_variance = np.var(non_zero_features)
            else:
                # All zeros - very flat expression
                avg_variance = 0.0
        else:
            # Fallback: compute variance of all non-zero features
            non_zero_features = features[features != 0]
            if len(non_zero_features) > 0:
                avg_variance = np.var(non_zero_features)
            else:
                avg_variance = 0.0
        
        # Normalize variance to 0-100 scale
        # Optimal expressiveness is moderate (not too flat, not too chaotic)
        # For feature variance, typical range is 0.001 to 0.1
        if avg_variance < 0.001:
            expressiveness = 30.0 + (avg_variance / 0.001) * 20.0  # 30-50 range
        elif avg_variance < 0.01:
            expressiveness = 50.0 + (avg_variance - 0.001) / 0.009 * 30.0  # 50-80 range
        elif avg_variance < 0.1:
            expressiveness = 80.0 - (avg_variance - 0.01) / 0.09 * 20.0  # 80-60 range
        else:
            expressiveness = max(40.0, 60.0 - (avg_variance - 0.1) * 20.0)  # Too chaotic
        
        return max(0.0, min(100.0, expressiveness))
    
    def _compute_head_movement_score(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Compute head movement score (stability).
        
        Moderate stability is good (not too rigid, not too fidgety).
        Excessive movement indicates distraction or disengagement.
        
        Args:
            features: Blendshape features array
            landmarks: Face landmarks array
        
        Returns:
            Head movement score (0-100, higher = more stable/engaged)
        """
        # Features 31-34 contain head pose information
        if len(features) < 35:
            # Fallback: use landmark-based head orientation
            if len(landmarks) > 10:
                # Estimate head orientation from face outline
                face_center = np.mean(landmarks[:10, :2], axis=0) if len(landmarks) >= 10 else np.mean(landmarks[:, :2], axis=0)
                frame_center = np.array([landmarks[:, 0].mean(), landmarks[:, 1].mean()])
                offset = np.linalg.norm(face_center - frame_center)
                # Normalize offset (typical range: 0-100 pixels)
                normalized_offset = min(1.0, offset / 100.0)
                stability = 80.0 - normalized_offset * 30.0  # 80-50 range
                return max(40.0, min(100.0, stability))
            return 60.0  # Default moderate stability
        
        head_pitch = abs(features[31]) if np.isfinite(features[31]) else 0.0
        head_yaw = abs(features[32]) if np.isfinite(features[32]) else 0.0
        head_roll = abs(features[33]) if np.isfinite(features[33]) else 0.0
        
        # Convert angles to degrees (check if already in degrees or radians)
        # If values are > 1, assume degrees; otherwise assume radians
        if head_pitch < 1.0 and head_yaw < 1.0 and head_roll < 1.0:
            # Likely radians
            head_pitch_deg = np.degrees(head_pitch)
            head_yaw_deg = np.degrees(head_yaw)
            head_roll_deg = np.degrees(head_roll)
        else:
            # Already in degrees
            head_pitch_deg = head_pitch
            head_yaw_deg = head_yaw
            head_roll_deg = head_roll
        
        # Score based on head angle deviation from forward-facing
        # Optimal: small angles (0-10 degrees) = engaged
        # Poor: large angles (>30 degrees) = distracted
        
        max_angle = max(head_pitch_deg, head_yaw_deg, head_roll_deg)
        
        if max_angle < 5:
            stability = 90.0  # Excellent stability
        elif max_angle < 10:
            stability = 85.0 - (max_angle - 5) * 2  # 85-75 range
        elif max_angle < 20:
            stability = 75.0 - (max_angle - 10) * 2  # 75-55 range
        elif max_angle < 30:
            stability = 55.0 - (max_angle - 20) * 1.5  # 55-40 range
        else:
            stability = max(30.0, 40.0 - (max_angle - 30) * 0.3)  # 40-30 range
        
        return max(30.0, min(100.0, stability))  # Minimum 30 for detected face
    
    def _compute_symmetry(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Compute facial symmetry score.
        
        More symmetric faces typically indicate engagement and focus.
        Asymmetry can indicate distraction, fatigue, or disengagement.
        
        Args:
            features: Blendshape features array
            landmarks: Face landmarks array
        
        Returns:
            Symmetry score (0-100, higher = more symmetric)
        """
        # Calculate symmetry from landmarks directly if feature not available
        if len(features) < 55 or features[51] == 0.0:
            # Fallback: compute symmetry from landmark positions
            if len(landmarks) >= 10:
                # Compare left and right side of face
                face_center_x = np.mean(landmarks[:, 0])
                
                # Left side points (assuming first half)
                left_points = landmarks[:len(landmarks)//2, 0]
                # Right side points (mirrored)
                right_points = landmarks[len(landmarks)//2:, 0]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    # Mirror right side
                    right_points_mirrored = 2 * face_center_x - right_points
                    
                    # Calculate average distance (symmetry error)
                    if len(left_points) == len(right_points_mirrored):
                        symmetry_error = np.mean(np.abs(left_points - right_points_mirrored))
                    else:
                        # Interpolate or use min length
                        min_len = min(len(left_points), len(right_points_mirrored))
                        symmetry_error = np.mean(np.abs(left_points[:min_len] - right_points_mirrored[:min_len]))
                    
                    # Normalize (typical range: 0-50 pixels)
                    normalized_error = min(1.0, symmetry_error / 50.0)
                    symmetry = 90.0 - normalized_error * 40.0  # 90-50 range
                    return max(50.0, min(100.0, symmetry))
            
            return 70.0  # Default good symmetry
        
        avg_symmetry = features[51]
        
        # Lower symmetry score (distance) = higher symmetry
        # Normalize to 0-100 scale
        # Typical symmetry distances: 0-50 pixels for good symmetry
        
        if avg_symmetry < 5:
            symmetry = 95.0  # Excellent symmetry
        elif avg_symmetry < 15:
            symmetry = 85.0 + (15 - avg_symmetry) / 10 * 10  # 85-95 range
        elif avg_symmetry < 30:
            symmetry = 70.0 + (30 - avg_symmetry) / 15 * 15  # 70-85 range
        elif avg_symmetry < 50:
            symmetry = 50.0 + (50 - avg_symmetry) / 20 * 20  # 50-70 range
        else:
            symmetry = max(40.0, 50.0 - (avg_symmetry - 50) * 0.2)  # 50-40 range
        
        return max(40.0, min(100.0, symmetry))  # Minimum 40 for detected face
    
    def _compute_mouth_activity(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Compute mouth activity score.
        
        Moderate mouth activity (speaking, occasional smiles) indicates engagement.
        No activity (closed, still) or excessive activity might indicate disengagement.
        
        Args:
            features: Blendshape features array
            landmarks: Face landmarks array
        
        Returns:
            Mouth activity score (0-100)
        """
        # Features 16-21 contain mouth information
        if len(features) < 22:
            # Fallback: compute from landmarks
            if len(landmarks) >= 15:
                # Try to find mouth region (typically last few landmarks or specific indices)
                mouth_region = landmarks[-5:] if len(landmarks) >= 5 else landmarks
                mouth_width = np.max(mouth_region[:, 0]) - np.min(mouth_region[:, 0])
                mouth_height = np.max(mouth_region[:, 1]) - np.min(mouth_region[:, 1])
                
                if mouth_width > 1e-6:
                    mar = mouth_height / mouth_width
                    # Normalize MAR (typical: 0.1-0.5)
                    if mar < 0.1:
                        return 45.0  # Closed
                    elif mar < 0.3:
                        return 65.0  # Moderate
                    elif mar < 0.5:
                        return 75.0  # Open
                    else:
                        return 55.0  # Very open
            return 60.0  # Default moderate activity
        
        mouth_aspect_ratio = features[18] if np.isfinite(features[18]) else 0.2
        mouth_openness = features[19] if np.isfinite(features[19]) else 0.0
        
        # Normalize mouth features
        mar_min, mar_max = self.normalization_ranges['mouth_aspect_ratio']
        
        # Ensure MAR is in reasonable range
        if mouth_aspect_ratio < 0.01:
            mouth_aspect_ratio = 0.1  # Default for closed mouth
        elif mouth_aspect_ratio > 1.0:
            mouth_aspect_ratio = 0.5  # Clamp very high values
        
        # Score based on mouth state
        # Closed mouth (low MAR) = neutral, moderate engagement
        # Slightly open (moderate MAR) = speaking/engaged
        # Wide open (high MAR) = surprise or distraction
        
        if mouth_aspect_ratio < mar_min:
            activity = 50.0  # Closed mouth - neutral engagement
        elif mouth_aspect_ratio < (mar_min + mar_max) / 2:
            # Increasing activity
            ratio = (mouth_aspect_ratio - mar_min) / ((mar_min + mar_max) / 2 - mar_min)
            activity = 50.0 + ratio * 30.0  # 50-80 range
        elif mouth_aspect_ratio < mar_max:
            # Optimal speaking range
            ratio = (mouth_aspect_ratio - (mar_min + mar_max) / 2) / ((mar_max - mar_min) / 2)
            activity = 80.0 - ratio * 15.0  # 80-65 range
        else:
            activity = 55.0  # Too wide open - slight penalty
        
        # Consider mouth openness separately
        # Moderate openness is good for engagement
        if mouth_openness > 0:
            # Normalize openness (typical range: 10-100 pixels for mouth height)
            normalized_openness = min(1.0, mouth_openness / 100.0)
            if 0.15 < normalized_openness < 0.5:
                activity += 8  # Bonus for moderate openness (speaking)
            elif normalized_openness > 0.7:
                activity -= 5  # Small penalty for excessive openness
        
        return max(40.0, min(100.0, activity))  # Minimum 40 for detected face
