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
        self.weights = {
            'attention': 0.25,
            'eye_contact': 0.20,
            'facial_expressiveness': 0.15,
            'head_movement': 0.15,
            'symmetry': 0.10,
            'mouth_activity': 0.15,
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
        Uses weighted average with a "weakest link" factor: any low metric
        pulls the score down so low engagement cannot slip through.
        """
        # Validate metrics are finite (use 0 for invalid to avoid hiding problems)
        attention = metrics.attention if np.isfinite(metrics.attention) else 0.0
        eye_contact = metrics.eye_contact if np.isfinite(metrics.eye_contact) else 0.0
        facial_expressiveness = metrics.facial_expressiveness if np.isfinite(metrics.facial_expressiveness) else 0.0
        head_movement = metrics.head_movement if np.isfinite(metrics.head_movement) else 0.0
        symmetry = metrics.symmetry if np.isfinite(metrics.symmetry) else 0.0
        mouth_activity = metrics.mouth_activity if np.isfinite(metrics.mouth_activity) else 0.0
        
        # Weighted average (wider effective range 0-100)
        raw = (
            attention * self.weights['attention'] +
            eye_contact * self.weights['eye_contact'] +
            facial_expressiveness * self.weights['facial_expressiveness'] +
            head_movement * self.weights['head_movement'] +
            symmetry * self.weights['symmetry'] +
            mouth_activity * self.weights['mouth_activity']
        )
        
        # Weakest-link: one low metric pulls down, but not as harsh.
        # Factor 0.55 + 0.45*min/100: min=0 -> 0.55, min=50 -> 0.775, min=100 -> 1.0.
        min_m = min(attention, eye_contact, facial_expressiveness,
                    head_movement, symmetry, mouth_activity)
        factor = 0.55 + 0.45 * (min_m / 100.0)
        score = raw * factor
        
        score = max(0.0, min(100.0, score))
        if not np.isfinite(score):
            return 0.0
        return float(score)
    
    def _compute_attention(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Compute attention from eye openness. Sensitive: small drops in EAR
        (e.g. drowsiness, looking down) reduce score sharply. Full range 0-100.
        """
        if len(features) < 3:
            return 0.0
        
        left_ear = features[0]
        right_ear = features[1]
        avg_ear = features[2]
        
        # Sensitive EAR mapping: peak 85-100 for 0.22-0.28; steep drop outside.
        # EAR 0.02-0.12 -> 0-25; 0.12-0.18 -> 25-45; 0.18-0.22 -> 45-70;
        # 0.22-0.28 -> 70-100; 0.28-0.38 -> 70-55; 0.38-0.55 -> 55-25.
        if avg_ear < 0.02:
            attention = 0.0
        elif avg_ear < 0.12:
            attention = (avg_ear - 0.02) / 0.10 * 25.0  # 0-25
        elif avg_ear < 0.18:
            attention = 25.0 + (avg_ear - 0.12) / 0.06 * 20.0  # 25-45
        elif avg_ear < 0.22:
            attention = 45.0 + (avg_ear - 0.18) / 0.04 * 25.0  # 45-70
        elif avg_ear <= 0.28:
            attention = 70.0 + (avg_ear - 0.22) / 0.06 * 30.0  # 70-100
        elif avg_ear < 0.38:
            attention = 70.0 - (avg_ear - 0.28) / 0.10 * 15.0  # 70-55
        elif avg_ear < 0.55:
            attention = 55.0 - (avg_ear - 0.38) / 0.17 * 30.0  # 55-25
        else:
            attention = max(0.0, 25.0 - (avg_ear - 0.55) * 50.0)
        
        # Asymmetry penalty: subtract up to 25 for very uneven eyes (squint, wink)
        if avg_ear > 1e-6:
            asym = abs(left_ear - right_ear) / max(avg_ear, 1e-6)
            attention -= asym * 25.0
        return max(0.0, min(100.0, attention))
    
    def _compute_eye_contact(
        self,
        features: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: tuple
    ) -> float:
        """
        Eye contact from gaze vs frame center and head yaw. Sensitive: small
        offsets or head turn reduce score sharply. Full range 0-100.
        """
        height, width = frame_shape[:2]
        frame_center = np.array([width / 2, height / 2])
        
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        if len(landmarks) < max(max(left_eye_indices), max(right_eye_indices)) + 1:
            return 0.0
        
        left_eye_center = np.mean(landmarks[left_eye_indices], axis=0)[:2]
        right_eye_center = np.mean(landmarks[right_eye_indices], axis=0)[:2]
        eye_center = (left_eye_center + right_eye_center) / 2
        
        distance_from_center = np.linalg.norm(eye_center - frame_center)
        max_distance = max(1e-6, np.sqrt(width**2 + height**2) / 2)
        nd = min(1.0, distance_from_center / max_distance)
        # Steeper: use squared term so small offsets still reduce score
        eye_contact = (1.0 - nd ** 1.3) * 100.0
        
        # Head yaw: participation 43,44 (degrees). Avoid over-penalizing small turns.
        yaw_deg = 0.0
        if len(features) > 44:
            y, p = features[43], features[44]
            if np.isfinite(y): yaw_deg = abs(y) if abs(y) <= 180 else min(90, abs(y))
            if np.isfinite(p): yaw_deg = max(yaw_deg, abs(p) if abs(p) <= 180 else min(90, abs(p)))
        yaw_penalty = min(1.0, yaw_deg / 35.0) * 0.5  # cap penalty at 50%
        eye_contact *= (1.0 - yaw_penalty)
        
        return max(0.0, min(100.0, eye_contact))
    
    def _compute_facial_expressiveness(
        self,
        features: np.ndarray
    ) -> float:
        """
        Expressiveness from feature variance. Sensitive: very flat (disengaged)
        maps to 0-30; moderate to 40-90; chaotic to 30-60. Full range 0-100.
        """
        if len(features) < 10:
            return 0.0
        
        idx = list(range(16, 30)) + list(range(45, 60))
        if len(features) > max(idx):
            arr = features[idx]
            nz = arr[arr != 0]
            v = np.var(nz) if len(nz) > 0 else 0.0
        else:
            nz = features[features != 0]
            v = np.var(nz) if len(nz) > 0 else 0.0
        
        if v < 1e-6:
            return 50.0  # No variance: treat as neutral, don't penalize
        elif v < 0.0005:
            return 50.0  # Very low: neutral expression, benefit of the doubt
        elif v < 0.003:
            return 30.0 + (v - 0.0005) / 0.0025 * 40.0  # 30-70
        elif v < 0.015:
            return 70.0 + (v - 0.003) / 0.012 * 25.0    # 70-95
        elif v < 0.08:
            return 95.0 - (v - 0.015) / 0.065 * 45.0    # 95-50
        elif v < 0.2:
            return 50.0 - (v - 0.08) / 0.12 * 40.0      # 50-10
        else:
            return max(0.0, 10.0 - (v - 0.2) * 20.0)
    
    def _compute_head_movement_score(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Head stability (forward-facing). Sensitive: small pitch/yaw/roll
        (e.g. 10-15°) already reduce score; 35°+ -> 0-15. Full range 0-100.
        """
        # Participation 43=head_yaw, 44=head_pitch (degrees)
        pitch_deg, yaw_deg = 0.0, 0.0
        if len(features) > 44:
            y, p = features[43], features[44]
            if np.isfinite(y): yaw_deg = abs(y) if abs(y) <= 180 else min(90, abs(y))
            if np.isfinite(p): pitch_deg = abs(p) if abs(p) <= 180 else min(90, abs(p))
        
        # Landmark fallback only when no pose from participation
        if pitch_deg == 0 and yaw_deg == 0 and len(landmarks) >= 350:
            nose = landmarks[4, :2]
            chin = landmarks[175, :2]
            lf = landmarks[234, 0] if len(landmarks) > 234 else landmarks[0, 0]
            rf = landmarks[454, 0] if len(landmarks) > 454 else landmarks[-1, 0]
            face_cx = (lf + rf) / 2
            face_hw = max(1e-6, abs(rf - lf) / 2)
            dy = chin[1] - nose[1]
            dx = chin[0] - nose[0]
            if abs(dy) > 1e-6:
                pitch_deg = min(90, abs(np.degrees(np.arctan(dx / dy))))
            nose_off = abs(nose[0] - face_cx)
            yaw_deg = min(90, (nose_off / face_hw) * 45.0)
        
        max_angle = max(pitch_deg, yaw_deg)
        # Steep curve: 0-4° -> 95-100; 4-8° -> 85-95; 8-15° -> 60-85; 15-25° -> 30-60; 25-40° -> 5-30; 40°+ -> 0-5
        if max_angle < 4:
            stability = 95.0 + (4 - max_angle) / 4 * 5.0
        elif max_angle < 8:
            stability = 85.0 - (max_angle - 4) / 4 * 10.0
        elif max_angle < 15:
            stability = 60.0 - (max_angle - 8) / 7 * 30.0
        elif max_angle < 25:
            stability = 30.0 - (max_angle - 15) / 10 * 25.0
        elif max_angle < 40:
            stability = 5.0 - (max_angle - 25) / 15 * 5.0
        else:
            stability = max(0.0, 5.0 - (max_angle - 40) / 30 * 5.0)
        return max(0.0, min(100.0, stability))
    
    def _compute_symmetry(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Symmetry from landmarks. Sensitive: small asymmetry (tilt, one-sided
        expression) reduces score; strong asymmetry -> 0-30. Full range 0-100.
        """
        if len(landmarks) < 10:
            return 0.0
        
        face_center_x = np.mean(landmarks[:, 0])
        n2 = len(landmarks) // 2
        left_x = landmarks[:n2, 0]
        right_x = landmarks[n2:2*n2, 0] if len(landmarks) >= 2*n2 else landmarks[n2:, 0]
        right_mirrored = 2 * face_center_x - right_x
        min_len = min(len(left_x), len(right_mirrored))
        if min_len == 0:
            return 50.0
        err = np.mean(np.abs(left_x[:min_len] - right_mirrored[:min_len]))
        # Normalize: 0-15px -> 85-100; 15-40 -> 55-85; 40-80 -> 20-55; 80+ -> 0-20
        if err < 15:
            symmetry = 85.0 + (15 - err) / 15 * 15.0
        elif err < 40:
            symmetry = 55.0 + (40 - err) / 25 * 30.0
        elif err < 80:
            symmetry = 20.0 + (80 - err) / 40 * 35.0
        else:
            symmetry = max(0.0, 20.0 - (err - 80) / 60 * 20.0)
        return max(0.0, min(100.0, symmetry))
    
    def _compute_mouth_activity(
        self,
        features: np.ndarray,
        landmarks: np.ndarray
    ) -> float:
        """
        Mouth activity from MAR and openness (participation indices 42, 40).
        Closed -> 15-40; moderate/speaking -> 55-95; wide -> 25-55. Full range 0-100.
        """
        # Participation: 40=mouth_openness, 41=mouth_width, 42=MAR
        if len(features) > 44:
            mar = features[42] if np.isfinite(features[42]) else 0.2
            opn = features[40] if np.isfinite(features[40]) else 15.0
        else:
            mar, opn = 0.2, 15.0
            if len(landmarks) >= 15:
                mouth_region = landmarks[-5:] if len(landmarks) >= 5 else landmarks
                mw = np.max(mouth_region[:, 0]) - np.min(mouth_region[:, 0])
                mh = np.max(mouth_region[:, 1]) - np.min(mouth_region[:, 1])
                if mw > 1e-6:
                    mar = mh / mw
                    opn = mh
        mar = max(0.02, min(0.9, mar))
        
        if mar < 0.08:
            activity = 10.0
        elif mar < 0.18:
            activity = 10.0 + (mar - 0.08) / 0.10 * 30.0   # 10-40
        elif mar < 0.28:
            activity = 40.0 + (mar - 0.18) / 0.10 * 45.0   # 40-85
        elif mar < 0.38:
            activity = 85.0 + (mar - 0.28) / 0.10 * 10.0   # 85-95
        elif mar < 0.5:
            activity = 95.0 - (mar - 0.38) / 0.12 * 35.0   # 95-60
        elif mar < 0.7:
            activity = 60.0 - (mar - 0.5) / 0.2 * 35.0     # 60-25
        else:
            activity = max(0.0, 25.0 - (mar - 0.7) * 50.0)
        
        if opn > 0:
            no = min(1.0, opn / 100.0)
            if 0.12 < no < 0.45:
                activity = min(100.0, activity + 8.0)
            elif no > 0.65:
                activity = max(0.0, activity - 12.0)
        return max(0.0, min(100.0, activity))
