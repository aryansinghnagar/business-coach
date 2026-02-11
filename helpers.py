"""
Business Meeting Copilot — consolidated utils.

Single module: video source, engagement scorer, context generator, face detection,
expression signifiers, engagement composites, capability, B2B opportunity detector.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

try:
    import psutil
except ImportError:
    psutil = None
try:
    import requests
except ImportError:
    requests = None


# ============== 1. Video source handler ==============
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
            lightweight: 720p, every-2nd-frame when FPS allows; both modes 720p
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
                            if cap.isOpened():
                                self.cap = cap
                                break
                        except Exception:
                            pass
                    if self.cap is not None:
                        break
                if not self.cap or not self.cap.isOpened():
                    try:
                        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
                    except Exception:
                        self.cap = cv2.VideoCapture(0)
                if self.cap and self.cap.isOpened():
                    w, h = (1280, 720)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    self.cap.set(cv2.CAP_PROP_FPS, 30 if lightweight else 60)
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
            Dictionary with keys width, height, fps, frame_count. When no source
            is open, returns width=0, height=0, fps=0, frame_count=-1 so callers
            need not guard for missing keys.
        """
        if self.source_type == VideoSourceType.PARTNER:
            return {'width': 0, 'height': 0, 'fps': 30, 'frame_count': -1}
        if not self.cap or not self.cap.isOpened():
            return {'width': 0, 'height': 0, 'fps': 0, 'frame_count': -1}
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

# ============== 2. Engagement scorer ==============
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


def _ear_from_pts(pts: np.ndarray) -> float:
    """Eye aspect ratio from eye landmark points (from expression_signifiers logic)."""
    if len(pts) < 4:
        return 0.2
    x, y = pts[:, 0], pts[:, 1]
    v = np.abs(np.max(y) - np.min(y))
    h = np.max(x) - np.min(x) + 1e-6
    ear = v / max(h, 2.0)
    return float(np.clip(ear, 0.05, 0.8))


# MediaPipe face mesh indices for eye regions
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE_TIP, _CHIN, _FACE_LEFT, _FACE_RIGHT = 4, 175, 234, 454


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
        """Weights aligned with meeting psychology: eye contact and attention dominate; expressiveness and head stability support."""
        self.weights = {
            'attention': 0.24,
            'eye_contact': 0.26,
            'facial_expressiveness': 0.14,
            'head_movement': 0.14,
            'symmetry': 0.11,
            'mouth_activity': 0.11,
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
        Calculate overall engagement score (0-100) from metrics.
        Weighted average with weakest-link factor: any low metric pulls the score
        down so engagement state is accurate and re-engage/capitalize advice is
        triggered appropriately (meeting-psychology research: don't miss low engagement).
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
        
        # Weakest-link: one very low metric pulls down (captures disengagement)
        # Softer curve: 0.58 + 0.42*min/100 so single low metric doesn't over-penalize
        min_m = min(attention, eye_contact, facial_expressiveness,
                    head_movement, symmetry, mouth_activity)
        factor = 0.58 + 0.42 * (min_m / 100.0)
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

    def compute_attention_eye_contact_from_landmarks(
        self, landmarks: np.ndarray, frame_shape: tuple
    ) -> Tuple[float, float]:
        """
        Compute attention and eye_contact from landmarks only (EAR + gaze + head pose).
        Used for hybrid basic metrics when blendshape features are unavailable.
        """
        lm = np.array(landmarks, dtype=np.float64)
        h, w = frame_shape[:2]
        if lm.shape[1] == 2:
            lm = np.hstack([lm, np.zeros((lm.shape[0], 1))])
        if np.max(lm[:, :2]) <= 1.0:
            lm[:, 0] *= w
            lm[:, 1] *= h

        # EAR from eye landmarks
        def _safe_idx(idx: int) -> np.ndarray:
            if idx < lm.shape[0]:
                return lm[idx, :2]
            return np.zeros(2)

        left_pts = np.array([_safe_idx(i) for i in _LEFT_EYE if i < lm.shape[0]])
        right_pts = np.array([_safe_idx(i) for i in _RIGHT_EYE if i < lm.shape[0]])
        left_ear = _ear_from_pts(left_pts) if len(left_pts) >= 4 else 0.2
        right_ear = _ear_from_pts(right_pts) if len(right_pts) >= 4 else 0.2
        avg_ear = (left_ear + right_ear) / 2.0

        # Head pose (yaw, pitch) from landmarks for eye_contact penalty
        yaw_deg, pitch_deg = 0.0, 0.0
        if lm.shape[0] > max(_NOSE_TIP, _CHIN, _FACE_LEFT, _FACE_RIGHT):
            nose = lm[_NOSE_TIP, :2]
            chin = lm[_CHIN, :2]
            lf = lm[_FACE_LEFT, 0]
            rf = lm[_FACE_RIGHT, 0]
            face_cx = (lf + rf) / 2
            face_hw = max(1e-6, abs(rf - lf) / 2)
            yaw_deg = np.clip((nose[0] - face_cx) / face_hw, -1.5, 1.5) * 45.0
            ley = np.mean(left_pts[:, 1]) if len(left_pts) else nose[1]
            rey = np.mean(right_pts[:, 1]) if len(right_pts) else nose[1]
            eye_center_y = (ley + rey) / 2
            nose_offset_y = eye_center_y - nose[1]
            pitch_deg = np.clip(nose_offset_y / (face_hw * 2.0) * 45.0, -30.0, 30.0)

        features = np.zeros(100)
        features[0], features[1], features[2] = left_ear, right_ear, avg_ear
        features[43], features[44] = yaw_deg, pitch_deg

        attention = self._compute_attention(features, lm)
        eye_contact = self._compute_eye_contact(features, lm, frame_shape)
        return (attention, eye_contact)

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

# ============== 3. Context generator ==============
from dataclasses import dataclass
from typing import Dict, List, Optional

# Centralized metric thresholds (0-100 scale). Research: Mehrabian (eye contact 70-80% ideal);
# engagement band literature; avoid hard cliffs by using consistent bands.
METRIC_LOW = 40   # Below = low engagement signal (re-engage territory)
METRIC_HIGH = 70  # Above = strong engagement signal (capitalize territory)
METRIC_STRONG = 75  # Eye contact: Mehrabian ideal band; "strong" = full attention
METRIC_CRITICAL = 30  # Below = critical risk (e.g. very low attention)
COMPOSITE_RISK = 50   # Composite (confusion, resistance, etc.) ≥ this = notable risk
COMPOSITE_RISK_HIGH = 55  # Higher composite risk threshold for disengagement/cognitive load

# Human-readable labels for composite metrics (used when passing context to Azure Foundry)
COMPOSITE_LABELS: Dict[str, str] = {
    # Multimodal composites
    "verbal_nonverbal_alignment": "Verbal-nonverbal alignment",
    "cognitive_load_multimodal": "Cognitive load",
    "rapport_engagement": "Rapport",
    "skepticism_objection_strength": "Skepticism/objection",
    "decision_readiness_multimodal": "Decision readiness",
    "trust_rapport": "Trust/rapport",
    "disengagement_risk_multimodal": "Disengagement risk",
    "confusion_multimodal": "Confusion",
    "tension_objection_multimodal": "Tension/objection",
    "loss_of_interest_multimodal": "Loss of interest",
    # Condition-based composites (higher-level mental states)
    "topic_interest_facial": "Topic interest (forward lean + eye contact)",
    "active_listening_facial": "Active listening (nodding + open mouth + eye contact)",
    "agreement_signals_facial": "Agreement signals (nodding + Duchenne smile + eye contact)",
    "evaluating_thinking_facial": "Evaluating/thinking (brow + lip pucker + look up)",
    "cognitive_processing_facial": "Cognitive processing (stillness + gaze shift + brow)",
    "resistance_cluster_facial": "Resistance cluster (contempt/gaze aversion or lip compression)",
    "receptivity_facial": "Receptivity (parted lips + softened forehead + Duchenne)",
    "withdrawal_facial": "Withdrawal (lip compression + gaze aversion + brow lower)",
    "disagreement_facial": "Disagreement (head shake + contempt)",
    "closing_window_facial": "Closing window (smile transition + fixed gaze + mouth relax)",
    "passive_listening_facial": "Passive listening (low eye contact + no nod + stillness)",
    "trust_openness_facial": "Trust/openness (facial symmetry + softened forehead + eye contact)",
    "curious_engaged_facial": "Curious/engaged (brow raise + eye contact + mouth open)",
}

# Import EngagementLevel - using lazy import to avoid circular dependency
_EngagementLevel = None

def _get_engagement_level():
    """Lazy import of EngagementLevel to avoid circular dependency."""
    global _EngagementLevel
    if _EngagementLevel is None:
        from engagement_detector import EngagementLevel as EL
        _EngagementLevel = EL
    return _EngagementLevel


@dataclass
class EngagementContext:
    """
    Contextual information for AI coaching.
    
    This structure contains natural language descriptions and actionable
    insights derived from engagement metrics.
    """
    summary: str  # Brief summary of engagement state
    level_description: str  # Description of engagement level
    key_indicators: List[str]  # List of key behavioral indicators
    suggested_actions: List[str]  # Suggested actions for the user
    risk_factors: List[str]  # Potential concerns or risk factors
    opportunities: List[str]  # Opportunities to capitalize on


class ContextGenerator:
    """
    Generates contextual information from engagement metrics.
    
    This class translates technical engagement scores into human-readable
    context that can be used by the AI coaching system to provide tailored
    recommendations.
    
    Usage:
        generator = ContextGenerator()
        context = generator.generate_context(score, metrics, level)
    """
    
    def generate_context_no_face(self) -> EngagementContext:
        """Generate context when no face is detected. AI still gets a minimal, actionable block."""
        summary = "No face detected. Assume neutral engagement; consider requesting video or checking in."
        level_description = "No video signal available. Cannot assess engagement from facial cues. Consider asking the partner to enable video or to confirm they are still present."
        key_indicators = ["No facial metrics available - video may be off or face not in frame"]
        suggested_actions = [
            "Ask an open-ended question to gauge engagement",
            "Consider requesting video if the meeting is important",
            "Continue with verbal check-ins (e.g. 'Does that make sense?')",
        ]
        risk_factors = ["No visual engagement data - may miss non-verbal cues"]
        opportunities = ["No positive or negative signals available from face; use verbal cues and tone"]
        return EngagementContext(
            summary=summary,
            level_description=level_description,
            key_indicators=key_indicators,
            suggested_actions=suggested_actions,
            risk_factors=risk_factors,
            opportunities=opportunities,
        )

    def generate_context(
        self,
        score: float,
        metrics: EngagementMetrics,
        level,  # EngagementLevel enum (imported lazily to avoid circular dependency)
        composite_metrics: Optional[Dict[str, float]] = None,
        acoustic_tags: Optional[List[str]] = None,
    ) -> EngagementContext:
        """
        Generate contextual information from engagement data.
        
        Args:
            score: Overall engagement score (0-100)
            metrics: Detailed engagement metrics
            level: Engagement level category
            composite_metrics: Optional composite metrics (confusion, tension, disengagement, etc.)
            acoustic_tags: Optional acoustic tags (e.g. acoustic_uncertainty, acoustic_tension)
        
        Returns:
            EngagementContext object with generated information
        """
        summary = self._generate_summary(score, level)
        level_description = self._describe_level(level)
        key_indicators = self._identify_key_indicators(metrics, level)
        suggested_actions = self._suggest_actions(score, metrics, level)
        risk_factors = self._identify_risks(score, metrics, level, composite_metrics, acoustic_tags)
        opportunities = self._identify_opportunities(score, metrics, level)
        # Absence of negative: when no risks and not low engagement, add opportunity line
        EngagementLevel = _get_engagement_level()
        if not risk_factors and level not in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            opportunities.append("No confusion or resistance detected; room to deepen engagement or capitalize with a clear ask or commitment.")
        
        return EngagementContext(
            summary=summary,
            level_description=level_description,
            key_indicators=key_indicators,
            suggested_actions=suggested_actions,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
    
    def _generate_summary(self, score: float, level) -> str:
        """Generate a brief summary of the engagement state."""
        EngagementLevel = _get_engagement_level()
        score_int = int(round(score))
        
        summaries = {
            EngagementLevel.VERY_LOW: f"Very low engagement (score: {score_int}/100). Partner highly disengaged—re-engage immediately (engagement-band research: immediate intervention).",
            EngagementLevel.LOW: f"Low engagement (score: {score_int}/100). Distraction or disinterest—simplify, check understanding, re-engage before continuing.",
            EngagementLevel.MEDIUM: f"Moderate engagement (score: {score_int}/100). Listening but not fully invested—build interest, address concerns, invite participation.",
            EngagementLevel.HIGH: f"High engagement (score: {score_int}/100). Actively engaged and receptive—capitalize: present key proposals, seek commitments (receptivity high).",
            EngagementLevel.VERY_HIGH: f"Very high engagement (score: {score_int}/100). Peak receptivity—optimal moment to advance proposals, seek decisions, or close (Cialdini: strike while commitment cues are high)."
        }
        
        return summaries.get(level, f"Engagement level: {score_int}/100")
    
    def _describe_level(self, level) -> str:
        """Provide a detailed description of the engagement level."""
        EngagementLevel = _get_engagement_level()
        descriptions = {
            EngagementLevel.VERY_LOW: "Partner highly disengaged—distracted or mentally absent. Re-establish connection; simplify and check understanding (meeting psychology: re-engage before content).",
            EngagementLevel.LOW: "Low engagement: minimal participation, glances away, or distraction. Re-engage with a direct question or relevance check; avoid overloading (cognitive load research).",
            EngagementLevel.MEDIUM: "Moderate engagement: listening but not fully participating. Build interest with value-focused content; invite participation and address concerns early.",
            EngagementLevel.HIGH: "Actively engaged: strong eye contact, facial expressiveness, attention. Capitalize now—present key proposals, ask for commitments, advance to next steps (receptivity window).",
            EngagementLevel.VERY_HIGH: "Peak engagement: strong interest and receptivity. Optimal moment to close or seek decisions; advance proposals and concrete next steps (Cialdini: commitment consistency)."
        }
        
        return descriptions.get(level, "Engagement level is being assessed.")
    
    def _identify_key_indicators(
        self,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        """Identify key behavioral indicators from metrics using centralized thresholds."""
        indicators = []
        # Attention: low < METRIC_LOW, high >= METRIC_HIGH
        if metrics.attention < METRIC_LOW:
            indicators.append("Low attention level - eyes may be closing or unfocused")
        elif metrics.attention >= METRIC_HIGH:
            indicators.append("High attention level - actively focused")
        # Eye contact: Mehrabian 70-80% ideal; strong >= METRIC_STRONG
        if metrics.eye_contact < METRIC_LOW:
            indicators.append("Limited eye contact - looking away from camera/speaker")
        elif metrics.eye_contact >= METRIC_STRONG:
            indicators.append("Strong eye contact - maintaining focus")
        # Facial expressiveness: flat = withdrawal; active = engaged
        if metrics.facial_expressiveness < METRIC_CRITICAL:
            indicators.append("Minimal facial expressions - appears passive or disengaged")
        elif metrics.facial_expressiveness >= METRIC_HIGH:
            indicators.append("Active facial expressions - showing interest and engagement")
        # Head movement: higher = more stable (lower = restless/distracted)
        if metrics.head_movement < METRIC_LOW:
            indicators.append("Excessive head movement - may indicate distraction or restlessness")
        elif metrics.head_movement >= METRIC_HIGH:
            indicators.append("Stable head position - focused and attentive")
        # Symmetry: balanced = focused; asymmetry = discomfort, skepticism,
        if metrics.symmetry < 50:
            indicators.append("Asymmetric facial features - possible fatigue or distraction")
        # Mouth activity: backchanneling, participation readiness
        if metrics.mouth_activity < METRIC_CRITICAL:
            indicators.append("Minimal mouth activity - not speaking or responding")
        elif metrics.mouth_activity >= METRIC_HIGH:
            indicators.append("Active mouth movement - speaking or showing interest")
        if not indicators:
            return ["Mixed or neutral indicators; no strong positive or negative signals observed."]
        return indicators
    
    def _suggest_actions(
        self,
        score: float,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Generate business-meeting specific suggested actions based on engagement state."""
        actions = []
        
        if level in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            # Low engagement - re-engagement strategies
            actions.append("Pause and ask a direct, open-ended question: 'What's your perspective on this?' or 'How does this align with your priorities?'")
            actions.append("Use their name to regain attention and create personal connection")
            actions.append("Check in professionally: 'I want to make sure this is valuable for you. Should we adjust our focus?'")
            actions.append("Simplify and reframe: Break complex topics into digestible pieces and use concrete examples")
            actions.append("Create urgency or relevance: Connect the topic to their immediate business needs or challenges")
            
            if metrics.eye_contact < METRIC_LOW:
                actions.append("Wait for visual acknowledgment before continuing - make brief eye contact and pause")
            if metrics.attention < METRIC_LOW:
                actions.append("Take a strategic pause: 'Let me pause here - what questions do you have so far?'")
            if metrics.facial_expressiveness < METRIC_CRITICAL:
                actions.append("Change your energy: Increase vocal variety, use gestures, or introduce a brief story to re-engage")
        
        elif level == EngagementLevel.MEDIUM:
            # Medium engagement - building momentum
            actions.append("Invite active participation: 'I'd love to hear your thoughts on this' or 'What's your experience with similar situations?'")
            actions.append("Provide compelling business context: Share ROI data, success stories, or industry benchmarks")
            actions.append("Address concerns proactively: 'You might be wondering about X - let me address that'")
            actions.append("Build buy-in: Highlight specific benefits, value propositions, or competitive advantages")
            actions.append("Use strategic questions: Ask thought-provoking questions that require engagement to answer")
            actions.append("Create interactive moments: 'Let's think through this together' or 'What would you do in this scenario?'")
        
        elif level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            # High engagement - capitalize on momentum
            actions.append("Present your strongest proposals NOW - this is the optimal moment for key decisions")
            actions.append("Seek commitments while engagement is peak: 'Based on what we've discussed, can we move forward with X?'")
            actions.append("Advance to concrete next steps: Propose specific action items, timelines, and ownership")
            actions.append("Leverage the momentum: Introduce additional value propositions or expand the opportunity")
            actions.append("Request feedback and input: 'What would make this even better?' or 'What else should we consider?'")
            actions.append("Build on their interest: Deepen the discussion in areas where they're most engaged")
            
            if metrics.mouth_activity >= METRIC_HIGH:
                actions.append("Encourage them to speak: 'I can see you have thoughts on this - please share'")
            if metrics.eye_contact >= METRIC_STRONG:
                actions.append("Make your most important point now - full attention is guaranteed")
        # Metric-specific tactical adjustments
        if metrics.head_movement < 50:
            actions.append("Check for environmental distractions or consider if they're multitasking - may need to refocus")
        
        if metrics.symmetry < 50:
            actions.append("Consider offering a brief break - facial asymmetry may indicate fatigue or cognitive load")
        
        if metrics.facial_expressiveness >= METRIC_HIGH and level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            actions.append("They're showing strong interest - this is an ideal time to ask for referrals, testimonials, or introductions")
        
        if not actions:
            actions.append("Continue monitoring engagement levels")
            actions.append("Ask an open-ended question to gauge engagement")
        return actions
    
    def _identify_risks(
        self,
        score: float,
        metrics: EngagementMetrics,
        level,
        composite_metrics: Optional[Dict[str, float]] = None,
        acoustic_tags: Optional[List[str]] = None,
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Identify potential risks or concerns; use composite and acoustic when available."""
        risks = []
        
        if level in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            risks.append("Risk of losing the meeting partner's attention completely")
            risks.append("Important information may not be received or understood")
            risks.append("Decision-making may be compromised due to low engagement")
            risks.append("Relationship may be negatively impacted if disengagement persists")
        
        if metrics.attention < METRIC_CRITICAL:
            risks.append("Very low attention - partner may be missing critical information")
        if metrics.eye_contact < METRIC_CRITICAL:
            risks.append("Minimal eye contact - partner may be multitasking or distracted")
        if metrics.head_movement < METRIC_LOW:
            risks.append("Excessive movement may indicate stress, impatience, or distraction")
        if level == EngagementLevel.MEDIUM and score < 50:
            risks.append("Engagement is declining - may drop to low if not addressed")
        comp = composite_metrics or {}
        ac = set(acoustic_tags or [])
        if comp.get("cognitive_load_multimodal", 0) >= COMPOSITE_RISK_HIGH or comp.get("confusion_multimodal", 0) >= COMPOSITE_RISK:
            risks.append("Elevated cognitive load or confusion (furrowed brow, gaze aversion, uncertain content)")
        if comp.get("skepticism_objection_strength", 0) >= COMPOSITE_RISK or comp.get("tension_objection_multimodal", 0) >= COMPOSITE_RISK:
            risks.append("Resistance or objection cues (lip compression, averted gaze, tense expression)")
        if comp.get("disengagement_risk_multimodal", 0) >= COMPOSITE_RISK_HIGH or comp.get("loss_of_interest_multimodal", 0) >= COMPOSITE_RISK:
            risks.append("Disengagement or loss of interest (flat affect, gaze drifting, low commitment language)")
        if "acoustic_uncertainty" in ac:
            risks.append("Vocal uncertainty or questioning tone—consider clarifying")
        if "acoustic_tension" in ac:
            risks.append("Vocal tension may signal objection or stress—acknowledge and address")
        if "acoustic_disengagement_risk" in ac:
            risks.append("Low vocal energy and flat pitch suggest possible withdrawal—re-engage")
        # Soft risk: MEDIUM level without strong positive signals—invite participation
        if level == EngagementLevel.MEDIUM and metrics.eye_contact < METRIC_STRONG and metrics.facial_expressiveness < METRIC_HIGH:
            risks.append("Limited positive engagement signals; consider inviting participation or checking in")
        
        return risks
    
    def _identify_opportunities(
        self,
        score: float,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Identify opportunities to capitalize on."""
        opportunities = []
        
        if level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            opportunities.append("Capitalize now: present strongest proposals and seek commitments (state-of-the-art meeting research: peak receptivity)")
            opportunities.append("Advance to concrete next steps—ideal moment to close or get verbal commitment (Cialdini: strike while commitment cues are high)")
            opportunities.append("Build on interest: ask for referrals, testimonials, or expand scope while engagement is high")
        
        if metrics.eye_contact >= METRIC_STRONG:
            opportunities.append("Strong eye contact = full attention—deliver your most important ask now")
        if metrics.facial_expressiveness >= METRIC_HIGH:
            opportunities.append("Active expressions signal interest—capitalize with a clear ask or next step")
        if metrics.mouth_activity >= METRIC_HIGH:
            opportunities.append("Partner ready to participate—invite input then steer toward commitment")
        
        if level == EngagementLevel.MEDIUM and score > 55:
            opportunities.append("Engagement building—continue value-focused approach and invite participation")
        
        if not opportunities:
            return ["Continue monitoring engagement levels"]
        return opportunities
    
    def format_for_ai(self, context: EngagementContext) -> str:
        """
        Format context as a rich, detailed string for AI consumption.
        
        This method formats the engagement context in a comprehensive way that
        provides the AI coach with all necessary information to provide
        actionable, business-meeting specific recommendations.
        
        Args:
            context: EngagementContext object
        
        Returns:
            Formatted string ready for AI prompt with rich business context
        """
        lines = [
            "=== REAL-TIME MEETING PARTNER ENGAGEMENT ANALYSIS ===",
            "",
            f"📊 ENGAGEMENT STATE: {context.summary}",
            f"📈 DETAILED ASSESSMENT: {context.level_description}",
            "",
            "🔍 KEY BEHAVIORAL INDICATORS:",
            *[f"  • {indicator}" for indicator in context.key_indicators],
            "",
            "💡 RECOMMENDED ACTIONS FOR OPTIMIZING THIS MOMENT:",
            *[f"  → {action}" for action in context.suggested_actions],
        ]
        
        if context.risk_factors:
            lines.extend([
                "",
                "⚠️ POTENTIAL RISKS & CONCERNS:",
                *[f"  ⚠ {risk}" for risk in context.risk_factors]
            ])
        
        if context.opportunities:
            lines.extend([
                "",
                "🎯 STRATEGIC OPPORTUNITIES:",
                *[f"  ✓ {opp}" for opp in context.opportunities]
            ])
        
        lines.extend([
            "",
            "=== END ENGAGEMENT ANALYSIS ===",
            "",
            "Use this real-time engagement data (facial metrics, speech cues, acoustic) to provide specific, actionable coaching advice.",
            "Leverage the current engagement state to capitalize on opportunities (re-engage when low, build when medium, capitalize when high).",
            "Be concise, practical, and immediately applicable to the meeting context."
        ])
        
        return "\n".join(lines)

    def build_context_bundle_for_foundry(
        self,
        context: EngagementContext,
        acoustic_summary: str = "",
        acoustic_tags: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        persistently_low_line: Optional[str] = None,
        composite_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Build the full context string in ordered sections for Azure AI Foundry.
        Groups related chunks: meeting context (summary, level, indicators, actions),
        composite metrics (higher-level mental states), negative signals, persistently low,
        positive/opportunities, voice, user context.
        """
        sections = []
        # [MEETING CONTEXT] — summary, level, key indicators, suggested actions only
        meeting_lines = [
            "=== REAL-TIME MEETING PARTNER ENGAGEMENT ANALYSIS ===",
            "",
            f"ENGAGEMENT STATE: {context.summary}",
            f"DETAILED ASSESSMENT: {context.level_description}",
            "",
            "KEY BEHAVIORAL INDICATORS:",
            *[f"  • {indicator}" for indicator in context.key_indicators],
            "",
            "RECOMMENDED ACTIONS:",
            *[f"  → {action}" for action in context.suggested_actions],
            "",
            "=== END ENGAGEMENT ANALYSIS ===",
        ]
        sections.append(f"[MEETING CONTEXT]\n" + "\n".join(meeting_lines) + "\n[/MEETING CONTEXT]")
        # [COMPOSITE METRICS] — higher-level mental states from condition-based combinations
        if composite_metrics:
            notable = [
                (k, v) for k, v in composite_metrics.items()
                if isinstance(v, (int, float)) and 35 <= float(v) <= 100
            ]
            notable.sort(key=lambda x: -float(x[1]))
            if notable:
                comp_lines = [
                    "[COMPOSITE METRICS]",
                    "Higher-level mental states inferred from facial signifier combinations (0–100):",
                ]
                for k, v in notable[:12]:
                    label = COMPOSITE_LABELS.get(k, k)
                    comp_lines.append(f"  • {label}: {int(v)}")
                comp_lines.append("[/COMPOSITE METRICS]")
                sections.append("\n".join(comp_lines))
        # [NEGATIVE SIGNALS]
        if context.risk_factors:
            neg_lines = ["[NEGATIVE SIGNALS]", "Risks and concerns to address:"]
            for r in context.risk_factors:
                neg_lines.append(f"  - {r}")
            if acoustic_tags:
                neg_lines.append(f"Voice tags: {', '.join(acoustic_tags)}")
            neg_lines.append("Suggest how to relieve root causes (clarify, validate and address, re-engage).")
            neg_lines.append("[/NEGATIVE SIGNALS]")
            sections.append("\n".join(neg_lines))
        # [PERSISTENTLY LOW / ABSENCE OF POSITIVE]
        if persistently_low_line:
            sections.append(f"[PERSISTENTLY LOW / ABSENCE OF POSITIVE]\n{persistently_low_line}\n[/PERSISTENTLY LOW / ABSENCE OF POSITIVE]")
        # [POSITIVE / OPPORTUNITIES]
        if context.opportunities:
            opp_lines = ["[POSITIVE / OPPORTUNITIES]"] + [f"  - {o}" for o in context.opportunities] + ["[/POSITIVE / OPPORTUNITIES]"]
            sections.append("\n".join(opp_lines))
        # [VOICE ANALYSIS]
        if acoustic_summary or (acoustic_tags and len(acoustic_tags) > 0):
            voice_parts = ["[VOICE ANALYSIS]"]
            if acoustic_summary:
                voice_parts.append(acoustic_summary)
            if acoustic_tags:
                voice_parts.append(f"Tags: {', '.join(acoustic_tags)}")
            voice_parts.append("[/VOICE ANALYSIS]")
            sections.append("\n".join(voice_parts))
        # [ADDITIONAL CONTEXT FROM USER]
        if additional_context and additional_context.strip():
            sections.append(f"[ADDITIONAL CONTEXT FROM USER]\n{additional_context.strip()}\n[/ADDITIONAL CONTEXT FROM USER]")
        return "\n\n".join(sections)

    # Backward compatibility alias
    build_context_bundle_for_openai = build_context_bundle_for_foundry

# ============== 4. Face detection ==============
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# Interface and result type
# -----------------------------------------------------------------------------


@dataclass
class FaceDetectionResult:
    """Standardized face detection result (any backend)."""
    landmarks: np.ndarray  # (N, 2) or (N, 3)
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 1.0
    emotions: Optional[Dict[str, float]] = None
    head_pose: Optional[Dict[str, float]] = None
    attributes: Optional[Dict[str, Any]] = None


class FaceDetectorInterface(ABC):
    """Abstract interface for face detection backends."""

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces in BGR image. Returns list of FaceDetectionResult."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """True if this detector can be used."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """e.g. 'mediapipe', 'azure_face_api'."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


# -----------------------------------------------------------------------------
# Azure 27 → MediaPipe 468 landmark expansion
# -----------------------------------------------------------------------------


def expand_azure_landmarks_to_mediapipe(
    landmarks: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, ...],
) -> np.ndarray:
    """
    Expand Azure 27 landmarks to 468x3 MediaPipe-compatible array.
    bbox: (left, top, width, height) or None. frame_shape: (H, W, ...).
    """
    out = np.zeros((468, 3), dtype=np.float64)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    n = landmarks.shape[0]
    lm = landmarks[:, :2] if landmarks.shape[1] >= 2 else landmarks
    if lm.shape[1] == 2:
        lm = np.hstack([lm, np.zeros((n, 1), dtype=lm.dtype)])

    def get(azure_idx: int) -> Optional[np.ndarray]:
        if azure_idx < n:
            return np.array([float(lm[azure_idx, 0]), float(lm[azure_idx, 1]), 0.0], dtype=np.float64)
        return None

    def first(*candidates: Optional[np.ndarray]) -> np.ndarray:
        for c in candidates:
            if c is not None:
                return c
        return np.zeros(3, dtype=np.float64)

    def set_mp(mp_idx: int, pt: np.ndarray) -> None:
        if 0 <= mp_idx < 468:
            out[mp_idx, :] = pt[:3]

    NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
    MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
    MOUTH_LEFT, MOUTH_RIGHT = 61, 17
    UPPER_LIP, LOWER_LIP = 13, 14
    NOSE_TIP, CHIN = 4, 175
    FACE_LEFT, FACE_RIGHT = 234, 454

    for i in NOSE:
        set_mp(i, first(get(2), get(17), get(18)))
    set_mp(4, first(get(2), get(17)))
    set_mp(6, first(get(17), get(2)))
    set_mp(19, first(get(18), get(2)))
    set_mp(20, first(get(19), get(2)))
    set_mp(51, first(get(21), get(2)))
    set_mp(94, first(get(22), get(2)))

    set_mp(MOUTH_LEFT, first(get(3)))
    set_mp(MOUTH_RIGHT, first(get(4)))
    set_mp(UPPER_LIP, first(get(23), get(24)))
    set_mp(LOWER_LIP, first(get(26), get(25)))
    fallback_mouth = first(get(24), get(3), get(4))
    for i in MOUTH:
        if out[i, 0] == 0 and out[i, 1] == 0:
            set_mp(i, fallback_mouth)

    pts = [get(7), get(8), get(9), get(10)]
    p0 = first(pts[0], pts[1], pts[2], pts[3])
    for i, idx in enumerate(LEFT_EYE):
        set_mp(idx, first(pts[i % 4], p0))
    pts = [get(16), get(14), get(15), get(13)]
    p0 = first(pts[0], pts[1], pts[2], pts[3])
    for i, idx in enumerate(RIGHT_EYE):
        set_mp(idx, first(pts[i % 4], p0))

    for i, idx in enumerate(LEFT_EYEBROW):
        set_mp(idx, first(get(6), get(5)) if i < 3 else first(get(5), get(6)))
    for i, idx in enumerate(RIGHT_EYEBROW):
        set_mp(idx, first(get(11), get(12)) if i < 3 else first(get(12), get(11)))

    nose_pt = get(2)
    mouth_mid_x = (float(lm[3, 0]) + float(lm[4, 0])) / 2 if n > 4 else w / 2
    if n > 26:
        mouth_bottom_y = float(lm[26, 1])
    elif n > 24:
        mouth_bottom_y = (float(lm[24, 1]) + float(lm[25, 1])) / 2
    else:
        mouth_bottom_y = float(out[14, 1]) if (out[14, 0] != 0 or out[14, 1] != 0) else h * 0.55
    nose_y = float(nose_pt[1]) if nose_pt is not None else (float(lm[24, 1]) if n > 24 else mouth_bottom_y - 20.0)

    if bbox is not None and len(bbox) >= 4:
        left, top, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        set_mp(CHIN, np.array([mouth_mid_x, mouth_bottom_y + 0.15 * max(bh, 1), 0.0], dtype=np.float64))
        set_mp(FACE_LEFT, np.array([float(left), nose_y, 0.0], dtype=np.float64))
        set_mp(FACE_RIGHT, np.array([float(left + bw), nose_y, 0.0], dtype=np.float64))
    else:
        set_mp(CHIN, np.array([mouth_mid_x, mouth_bottom_y + 0.1 * h, 0.0], dtype=np.float64))
        set_mp(FACE_LEFT, np.array([max(0, mouth_mid_x - 0.3 * w), nose_y, 0.0], dtype=np.float64))
        set_mp(FACE_RIGHT, np.array([min(w, mouth_mid_x + 0.3 * w), nose_y, 0.0], dtype=np.float64))

    return out


# -----------------------------------------------------------------------------
# MediaPipe implementation
# -----------------------------------------------------------------------------


class MediaPipeFaceDetector(FaceDetectorInterface):
    """MediaPipe Face Mesh–based detector (468 landmarks)."""

    def __init__(self, min_detection_confidence: float = 0.15, min_tracking_confidence: float = 0.15):
        import mediapipe as mp
        self._mp = mp
        self._det_conf = max(0.01, min(0.99, float(min_detection_confidence)))
        self._track_conf = max(0.01, min(0.99, float(min_tracking_confidence)))
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        )
        self._face_mesh_static = None
        self._face_detection = None
        self._consecutive_tracking_failures = 0
        self._available = True

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        import cv2
        if image is None or image.size == 0:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            self._consecutive_tracking_failures = 0
            return self._extract_landmarks(results, width, height)
        self._consecutive_tracking_failures += 1
        static = self._get_static()
        results_static = static.process(rgb)
        if results_static.multi_face_landmarks:
            self._consecutive_tracking_failures = 0
            return self._extract_landmarks(results_static, width, height)
        if self._consecutive_tracking_failures >= 5:
            simple = self._get_simple_detection().process(rgb)
            if simple.detections:
                self._reset_tracking_mesh()
                self._consecutive_tracking_failures = 0
        return []

    def _get_static(self):
        if self._face_mesh_static is None:
            self._face_mesh_static = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.05, min_tracking_confidence=0.05,
            )
        return self._face_mesh_static

    def _get_simple_detection(self):
        if self._face_detection is None:
            self._face_detection = self._mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.3,
            )
        return self._face_detection

    def _extract_landmarks(self, results, width: int, height: int) -> List[FaceDetectionResult]:
        out = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x * width, lm.y * height, lm.z * width])
            arr = np.array(landmarks, dtype=np.float32)
            x, y = arr[:, 0], arr[:, 1]
            bbox = (int(np.min(x)), int(np.min(y)), int(np.max(x) - np.min(x)), int(np.max(y) - np.min(y)))
            out.append(FaceDetectionResult(landmarks=arr, bounding_box=bbox, confidence=1.0))
        return out

    def _reset_tracking_mesh(self) -> None:
        try:
            self.face_mesh.close()
        except Exception:
            pass
        self.face_mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=self._det_conf, min_tracking_confidence=self._track_conf,
        )

    def is_available(self) -> bool:
        return self._available

    def get_name(self) -> str:
        return "mediapipe"

    def close(self) -> None:
        for obj in (getattr(self, "face_mesh", None), self._face_mesh_static, self._face_detection):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# Azure Face API implementation
# -----------------------------------------------------------------------------


class AzureFaceAPIDetector(FaceDetectorInterface):
    """Azure Face API–based detector (emotions, head pose, 27 landmarks)."""

    def __init__(self):
        from services import get_azure_face_api_service
        self.service = get_azure_face_api_service()
        self._available = self.service is not None

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        if not self.service:
            return []
        try:
            import requests
            face_data_list = self.service.detect_faces(
                image, return_face_landmarks=True, return_face_attributes=True,
            )
            if not face_data_list:
                return []
            out = []
            for face_data in face_data_list:
                landmarks = self.service.extract_landmarks_from_face(face_data)
                if landmarks is None or landmarks.shape[0] < 10:
                    continue
                bbox = self.service.get_face_rectangle(face_data)
                emotions = self.service.extract_emotion_from_face(face_data)
                head_pose = self.service.extract_head_pose_from_face(face_data)
                if landmarks.shape[1] == 2:
                    landmarks = np.hstack([landmarks, np.zeros((landmarks.shape[0], 1), dtype=landmarks.dtype)])
                attributes = {}
                if "faceAttributes" in face_data:
                    attrs = face_data["faceAttributes"]
                    attributes = {k: attrs.get(k) for k in ("age", "gender", "smile", "glasses", "facialHair")}
                out.append(FaceDetectionResult(
                    landmarks=landmarks, bounding_box=bbox, confidence=1.0,
                    emotions=emotions, head_pose=head_pose, attributes=attributes,
                ))
            return out
        except Exception:
            return []

    def is_available(self) -> bool:
        return self._available

    def get_name(self) -> str:
        return "azure_face_api"

    def close(self) -> None:
        pass

# ============== 5. Expression signifiers ==============
import json
import os
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import config


# MediaPipe-style indices (use safe access for Azure's 27-point set)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
# Mouth corners: 61 (L), 291 (L inner), 17 (R), 78 (R inner) – 291,78 may be out of MOUTH
MOUTH_LEFT, MOUTH_RIGHT = 61, 17
NOSE_TIP, CHIN = 4, 175
FACE_LEFT, FACE_RIGHT = 234, 454
# Inner brows for furrow
INNER_BROW_L, INNER_BROW_R = 70, 300  # fallback to first/last of eyebrow if OOB

# All 44 signifier keys (single shared list; no per-call allocation).
# Original 30 + 14 research-based additions (Waller 2024, Ekman FACS, Edmondson, B2B negotiation).
SIGNIFIER_KEYS: List[str] = [
    "g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
    "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
    "g1_micro_smile", "g1_brow_raise_sustained", "g1_mouth_open_receptive", "g1_eye_widening", "g1_nod_intensity",
    "g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
    "g2_stillness", "g2_lowered_brow",
    "g2_brow_furrow_deep", "g2_gaze_shift_frequency", "g2_mouth_tight_eval",
    "g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
    "g3_lip_corner_dip", "g3_brow_lower_sustained", "g3_eye_squeeze", "g3_head_shake",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition",
    "g4_mouth_relax", "g4_smile_sustain",
]


def _safe(landmarks: np.ndarray, indices: List[int], dim: int = 2) -> np.ndarray:
    n, d = landmarks.shape[0], min(dim, landmarks.shape[1])
    out = []
    for i in indices:
        if i < n:
            out.append(landmarks[i, :d])
        else:
            out.append(np.zeros(d, dtype=landmarks.dtype))
    return np.array(out) if out else np.zeros((0, d))


def _ear(pts: np.ndarray) -> float:
    """Eye aspect ratio; robust to degenerate/outlier points."""
    if len(pts) < 4:
        return 0.2
    x, y = pts[:, 0], pts[:, 1]
    v = np.abs(np.max(y) - np.min(y))
    h = np.max(x) - np.min(x) + 1e-6
    ear = v / max(h, 2.0)  # Floor h to avoid noise-driven extremes
    return float(np.clip(ear, 0.05, 0.8))  # Plausible range


def _normalize_lm(landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
    lm = np.array(landmarks, dtype=np.float64)
    if lm.shape[1] == 2:
        lm = np.hstack([lm, np.zeros((lm.shape[0], 1))])
    if np.max(lm[:, :2]) <= 1.0:
        lm[:, 0] *= w
        lm[:, 1] *= h
        if lm.shape[1] > 2:
            lm[:, 2] *= max(w, h)
    return lm


# In-code defaults for signifier threshold params (no external file).
_DEFAULT_SIGNIFIER_PARAMS: Dict[str, Any] = {
    "contempt_abs_threshold": 0.11,
    "contempt_above_baseline": 0.05,
    "contempt_temporal_min": 2,
    "contempt_warmup_frames": 6,
    "gaze_aversion_deg": 8.0,
    "gaze_aversion_s8": 8,
    "gaze_aversion_s6": 6,
    "gaze_aversion_s4": 4,
    "duchenne_au6_lo": 0.75,
    "duchenne_au6_hi": 0.98,
    "duchenne_au12_threshold": 0.006,
    "lip_compression_below_threshold": 0.8,
    "lip_compression_sustained_min": 4,
    "lip_compression_sustained_all_min": 5,
    "narrowed_pupils_ratio_threshold": 0.88,
    "narrowed_pupils_sustained_min": 2,
    "baseline_warmup_alpha": 0.2,
    "baseline_warmup_frames": 60,
    "baseline_alpha_slow": 0.08,
    "smooth_alpha": 1.0,
}


class ExpressionSignifierEngine:
    """30 signifier scores (0-100) from landmarks; temporal buffer for blinks, nodding, stillness."""

    _FPS_REF = 30.0  # Reference FPS for scaling time-based frame thresholds (30–60 fps)

    def __init__(
        self,
        buffer_frames: int = 22,
        weights_provider: Optional[Callable[[], Dict[str, List[float]]]] = None,
    ):
        self.buffer_frames = max(10, buffer_frames)
        self._buf: deque = deque(maxlen=self.buffer_frames)
        self._weights_provider = weights_provider
        self._fps: float = self._FPS_REF
        self._landmarks: Optional[np.ndarray] = None
        self._face_result: Optional[FaceDetectionResult] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        # Baseline for dilation proxy and Z
        self._baseline_eye_area: float = 0.0
        self._baseline_z: float = 0.0
        self._baseline_ear: float = 0.0  # For Duchenne squinch detection
        self._baseline_mar: float = 0.0  # For Duchenne mouth-opening comparison
        self._pupil_dilation_history: deque = deque(maxlen=3)  # Minimal smoothing for real-time
        self._last_pupil_dilation_score: float = 0.0  # Last valid score (for blink frames); default 0
        self._blink_start_frames: int = 0
        self._blinks_in_window: int = 0
        self._last_blink_reset: float = 0.0
        # Phase 4: Configurable params from JSON (load first so smooth_alpha can use it)
        self._params: Dict[str, Any] = self._load_signifier_params()
        # Feature smoothing: 1.0 = no smoothing (real-time); was 0.75 for blend with previous frame
        self._smooth_alpha: float = float(self._params.get("smooth_alpha", 1.0))
        # Contempt: baseline asymmetry (person-specific); history for temporal consistency
        self._contempt_asymmetry_history: deque = deque(maxlen=24)
        # Previous-frame landmarks for temporal movement (relaxed exhale: stillness after release)
        self._prev_landmarks: Optional[np.ndarray] = None
        # Baseline warmup: faster adaptation in first N frames (Phase 2)
        self._baseline_warmup_frames: int = 0
        self._BASELINE_WARMUP_MAX: int = 60  # ~2 s at 30 fps
        self._BASELINE_ALPHA_SLOW: float = 0.08   # EMA alpha after warmup
        self._BASELINE_ALPHA_FAST: float = 0.20   # EMA alpha during warmup

    def _load_signifier_params(self) -> Dict[str, Any]:
        """Return signifier threshold params (in-code defaults)."""
        return dict(_DEFAULT_SIGNIFIER_PARAMS)

    def set_fps(self, fps: float) -> None:
        """Set current processing FPS so time-based frame thresholds scale (30–60 fps)."""
        self._fps = max(15.0, min(120.0, float(fps)))

    def reset(self) -> None:
        """Clear temporal buffer and baselines (e.g. on detection start)."""
        self._buf.clear()
        self._baseline_eye_area = 0.0
        self._baseline_z = 0.0
        self._baseline_ear = 0.0
        self._baseline_mar = 0.0
        self._pupil_dilation_history.clear()
        self._last_pupil_dilation_score = 0.0  # default 0; output scale 0–100
        self._blink_start_frames = 0
        self._blinks_in_window = 0
        self._contempt_asymmetry_history.clear()
        self._prev_landmarks = None
        self._diag_frame_count = 0
        self._baseline_warmup_frames = 0

    def update(
        self,
        landmarks: np.ndarray,
        face_result: Optional[FaceDetectionResult],
        frame_shape: Tuple[int, int, int],
    ) -> None:
        h, w = frame_shape[:2]
        lm = _normalize_lm(landmarks, w, h)
        self._landmarks = lm
        self._face_result = face_result
        self._shape = frame_shape

        # Snapshot for buffer
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        left_ear = _ear(le) if len(le) >= 4 else 0.2
        right_ear = _ear(re) if len(re) >= 4 else 0.2
        ear = (left_ear + right_ear) / 2.0
        eye_area = 0.0
        if len(le) >= 3 and len(re) >= 3:
            def _area(p):
                if len(p) < 3: return 0.0
                x, y = p[:, 0], p[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            eye_area = _area(le[:, :2]) + _area(re[:, :2])

        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        ley = np.mean(lb[:, 1]) if len(lb) else 0.0
        rey = np.mean(rb[:, 1]) if len(rb) else 0.0
        leye_y = np.mean(le[:, 1]) if len(le) else 0.0
        reye_y = np.mean(re[:, 1]) if len(re) else 0.0
        eyebrow_l = leye_y - ley if (ley != 0 or leye_y != 0) else 15.0
        eyebrow_r = reye_y - rey if (rey != 0 or reye_y != 0) else 15.0

        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 4:
            mar, mw, mh = 0.2, 40.0, 8.0
        else:
            mw = float(np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0]) + 1e-6)
            mh = float(np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1]))
            mar = float(np.clip(mh / mw, 0.02, 0.65))  # Plausible range; rejects noise extremes

        # Head pose from participation-style logic
        # Try to use Azure head_pose if available (more accurate)
        if face_result and face_result.head_pose:
            head_pitch = float(face_result.head_pose.get('pitch', 0.0))
            head_yaw = float(face_result.head_pose.get('yaw', 0.0))
            head_roll = float(abs(face_result.head_pose.get('roll', 0.0)))
        else:
            # MediaPipe: estimate from landmarks
            nose = lm[NOSE_TIP, :2] if lm.shape[0] > NOSE_TIP else lm[0, :2]
            chin = lm[CHIN, :2] if lm.shape[0] > CHIN else lm[min(16, lm.shape[0] - 1), :2]
            lf = lm[FACE_LEFT, 0] if lm.shape[0] > FACE_LEFT else np.mean(lm[:, 0]) - 50
            rf = lm[FACE_RIGHT, 0] if lm.shape[0] > FACE_RIGHT else np.mean(lm[:, 0]) + 50
            face_cx = (lf + rf) / 2
            face_hw = max(1e-6, abs(rf - lf) / 2)
            head_yaw = np.clip((nose[0] - face_cx) / face_hw, -1.5, 1.5) * 45.0
            
            # For pitch: use nose position relative to eye center (better than chin-nose)
            # Azure convention: positive pitch = looking up, negative = looking down
            # Looking up: nose moves up (lower Y) relative to eyes
            # Looking down: nose moves down (higher Y) relative to eyes
            eye_center_y = (np.mean(le[:, 1]) + np.mean(re[:, 1])) / 2 if (len(le) and len(re)) else nose[1]
            nose_offset_y = eye_center_y - nose[1]  # Positive = nose above eyes (looking up), Negative = nose below (looking down)
            # Normalize by face scale for pitch estimation
            face_scale_for_pitch = max(1e-6, face_hw * 2.0)  # Use face width as proxy for scale
            head_pitch = np.clip(nose_offset_y / face_scale_for_pitch * 45.0, -30.0, 30.0)  # Scale to degrees, match Azure convention
            
            head_roll = 0.0
            if lm.shape[0] > max(FACE_LEFT, FACE_RIGHT):
                ly = float(lm[FACE_LEFT, 1])
                ry = float(lm[FACE_RIGHT, 1])
                head_roll = abs(np.degrees(np.arctan2(ry - ly, rf - lf)))

        face_z = float(np.mean(lm[:, 2])) if lm.shape[1] > 2 and np.any(np.isfinite(lm[:, 2])) else 0.0
        frame_c = np.array([w / 2, h / 2])
        eye_c = (np.mean(le[:, :2], axis=0) + np.mean(re[:, :2], axis=0)) / 2 if (len(le) and len(re)) else frame_c
        gaze_x = float(eye_c[0] - frame_c[0])
        gaze_y = float(eye_c[1] - frame_c[1])
        face_var = float(np.var(lm[:, :2])) if lm.size >= 4 else 0.0
        nose_arr = [lm[i, 1] for i in NOSE if i < lm.shape[0]]
        nose_std = float(np.std(nose_arr)) if nose_arr else 0.0
        nose_height = float(max(nose_arr) - min(nose_arr)) if nose_arr else 0.0
        
        # Face scale factor for size/position invariance (inter-ocular distance)
        # This normalizes all pixel-based thresholds to be relative to face size
        if len(le) >= 2 and len(re) >= 2:
            left_eye_center = np.mean(le[:, :2], axis=0)
            right_eye_center = np.mean(re[:, :2], axis=0)
            face_scale = float(np.linalg.norm(left_eye_center - right_eye_center))
        elif mw > 0:
            face_scale = mw * 1.5  # Fallback: mouth width * 1.5 (typical ratio)
        else:
            # Last resort: estimate from landmark spread
            if lm.shape[0] > 10:
                face_scale = float(np.max(lm[:min(100, lm.shape[0]), 0]) - np.min(lm[:min(100, lm.shape[0]), 0])) * 0.4
            else:
                face_scale = 50.0
        face_scale = max(20.0, face_scale)  # Minimum scale to avoid division issues
        # Proper MAR: V=d(13,14) / H=d(61,17). Better for lip compression than bbox MAR.
        mar_inner = mar
        if lm.shape[0] > 61:
            p13 = lm[13, :2]
            p14 = lm[14, :2]
            p61 = lm[61, :2]
            p17 = lm[17, :2]
            v = float(np.linalg.norm(p13 - p14))
            h = float(np.linalg.norm(p61 - p17)) + 1e-6
            if h >= 5.0:
                mar_inner = v / h
        is_blink = 1.0 if ear < 0.16 else 0.0

        # Mouth corner asymmetry ratio (for contempt baseline): |ly - ry| / face_scale
        mouth_corner_asymmetry_ratio = 0.0
        if lm.shape[0] > max(MOUTH_LEFT, MOUTH_RIGHT):
            ly = float(lm[MOUTH_LEFT, 1])
            ry = float(lm[MOUTH_RIGHT, 1])
            mouth_corner_asymmetry_ratio = abs(ly - ry) / max(face_scale, 1e-6)

        # Baselines: faster alpha during warmup (first ~2 s), then slower for noise invariance
        warmup_max = self._params.get("baseline_warmup_frames", self._BASELINE_WARMUP_MAX)
        alpha_fast = self._params.get("baseline_warmup_alpha", self._BASELINE_ALPHA_FAST)
        alpha_slow = self._params.get("baseline_alpha_slow", self._BASELINE_ALPHA_SLOW)
        alpha = alpha_fast if self._baseline_warmup_frames < warmup_max else alpha_slow
        decay = 1.0 - alpha
        if is_blink < 0.5:
            if self._baseline_eye_area <= 0 and eye_area > 0:
                self._baseline_eye_area = eye_area
            elif eye_area > 0:
                self._baseline_eye_area = decay * self._baseline_eye_area + alpha * eye_area
        if self._baseline_z == 0 and face_z != 0:
            self._baseline_z = face_z
        elif face_z != 0:
            self._baseline_z = decay * self._baseline_z + alpha * face_z
        if is_blink < 0.5:
            if self._baseline_ear <= 0 and ear > 0:
                self._baseline_ear = ear
            elif ear > 0:
                self._baseline_ear = decay * self._baseline_ear + alpha * ear
        if self._baseline_mar <= 0 and mar_inner > 0:
            self._baseline_mar = mar_inner
        elif mar_inner > 0:
            self._baseline_mar = decay * self._baseline_mar + alpha * mar_inner
        self._baseline_warmup_frames = min(self._baseline_warmup_frames + 1, warmup_max + 1)

        # Feature-level temporal smoothing: blend with previous frame to reduce landmark jitter
        prev = self._buf[-1] if len(self._buf) > 0 else None
        a = self._smooth_alpha
        if prev is not None:
            ear = a * ear + (1 - a) * prev.get("ear", ear)
            eye_area = a * eye_area + (1 - a) * prev.get("eye_area", eye_area)
            mar = a * mar + (1 - a) * prev.get("mar", mar)
            mar_inner = a * mar_inner + (1 - a) * prev.get("mar_inner", mar_inner)
            eyebrow_l = a * eyebrow_l + (1 - a) * prev.get("eyebrow_l", eyebrow_l)
            eyebrow_r = a * eyebrow_r + (1 - a) * prev.get("eyebrow_r", eyebrow_r)
            gaze_x = a * gaze_x + (1 - a) * prev.get("gaze_x", gaze_x)
            gaze_y = a * gaze_y + (1 - a) * prev.get("gaze_y", gaze_y)
            head_pitch = a * head_pitch + (1 - a) * prev.get("pitch", head_pitch)
            head_yaw = a * head_yaw + (1 - a) * prev.get("yaw", head_yaw)
            face_z = a * face_z + (1 - a) * prev.get("face_z", face_z)
            face_var = a * face_var + (1 - a) * prev.get("face_var", face_var)
            nose_std = a * nose_std + (1 - a) * prev.get("nose_std", nose_std)

        # Per-frame face movement (mean squared displacement vs previous frame) for relaxed exhale
        face_movement = 0.0
        if self._prev_landmarks is not None and self._prev_landmarks.shape[0] == lm.shape[0] and lm.shape[0] >= 4:
            n = min(lm.shape[0], self._prev_landmarks.shape[0])
            d = lm[:n, :2].astype(np.float64) - self._prev_landmarks[:n, :2].astype(np.float64)
            face_movement = float(np.mean(np.sum(d ** 2, axis=1)))

        t = time.time()
        # Blink counting (reset every 2s)
        if t - self._last_blink_reset > 2.0:
            self._blinks_in_window = 0
            self._last_blink_reset = t
        if is_blink > 0.5:
            self._blink_start_frames += 1
        else:
            min_b = max(1, int(2 * self._fps / self._FPS_REF))
            max_b = max(min_b, int(8 * self._fps / self._FPS_REF))
            if self._blink_start_frames >= min_b and self._blink_start_frames <= max_b:
                self._blinks_in_window += 1
            self._blink_start_frames = 0

        snap = {
            "ear": ear, "eye_area": eye_area, "eyebrow_l": eyebrow_l, "eyebrow_r": eyebrow_r,
            "pitch": head_pitch, "yaw": head_yaw, "roll": head_roll,
            "mar": mar, "mar_inner": mar_inner, "mouth_w": mw, "mouth_h": mh,
            "face_z": face_z, "gaze_x": gaze_x, "gaze_y": gaze_y,
            "face_var": face_var, "nose_std": nose_std, "nose_height": nose_height, "is_blink": is_blink,
            "face_scale": face_scale,
            "mouth_corner_asymmetry_ratio": mouth_corner_asymmetry_ratio,
            "face_movement": face_movement,
        }
        self._buf.append(snap)
        self._prev_landmarks = lm.copy()
        # Frame counter for diagnostic logging (Phase C1)
        self._diag_frame_count = getattr(self, "_diag_frame_count", 0) + 1

    def get_all_scores(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        lm = self._landmarks
        fr = self._face_result
        shp = self._shape
        buf = list(self._buf)
        if lm is None or shp is None or len(buf) < 1:
            for k in self._all_keys():
                out[k] = 0.0
            return out

        h, w = shp[0], shp[1]
        cur = buf[-1]
        mouth_pts = _safe(lm, MOUTH)
        ml = lm[MOUTH_LEFT, :2] if lm.shape[0] > MOUTH_LEFT else np.zeros(2)
        mr = lm[MOUTH_RIGHT, :2] if lm.shape[0] > MOUTH_RIGHT else np.zeros(2)

        # --- Group 1: Interest & Engagement ---
        out["g1_duchenne"] = self._g1_duchenne(lm, cur, buf)
        out["g1_pupil_dilation"] = self._g1_pupil_dilation(cur, buf)
        out["g1_eyebrow_flash"] = self._g1_eyebrow_flash(buf)
        out["g1_eye_contact"] = self._g1_eye_contact(cur, w, h, buf)
        out["g1_head_tilt"] = self._g1_head_tilt(cur, buf)
        out["g1_forward_lean"] = self._g1_forward_lean(cur, buf)
        out["g1_facial_symmetry"] = self._g1_facial_symmetry(lm, buf)
        out["g1_rhythmic_nodding"] = self._g1_rhythmic_nodding(buf)
        out["g1_parted_lips"] = self._g1_parted_lips(cur, buf)
        out["g1_softened_forehead"] = self._g1_softened_forehead(lm, cur, buf)
        out["g1_micro_smile"] = self._g1_micro_smile(lm, cur, buf)
        out["g1_brow_raise_sustained"] = self._g1_brow_raise_sustained(buf)
        out["g1_mouth_open_receptive"] = self._g1_mouth_open_receptive(cur, buf)
        out["g1_eye_widening"] = self._g1_eye_widening(cur, buf)
        out["g1_nod_intensity"] = self._g1_nod_intensity(buf)

        # --- Group 2: Cognitive Load ---
        out["g2_look_up_lr"] = self._g2_look_up_lr(cur, buf)
        out["g2_lip_pucker"] = self._g2_lip_pucker(lm, cur, buf)
        out["g2_eye_squint"] = self._g2_eye_squint(cur, buf)
        out["g2_thinking_brow"] = self._g2_thinking_brow(lm, buf)
        out["g2_chin_stroke"] = 0.0  # no hand detection
        out["g2_stillness"] = self._g2_stillness(buf)
        out["g2_lowered_brow"] = self._g2_lowered_brow(lm, buf)
        out["g2_brow_furrow_deep"] = self._g2_brow_furrow_deep(lm, buf)
        out["g2_gaze_shift_frequency"] = self._g2_gaze_shift_frequency(buf)
        out["g2_mouth_tight_eval"] = self._g2_mouth_tight_eval(cur, buf)

        # --- Group 3: Resistance (store as-is; composite uses 100 - x) ---
        out["g3_contempt"] = self._g3_contempt(lm, ml, mr, fr, cur, buf)
        out["g3_nose_crinkle"] = self._g3_nose_crinkle(cur, buf)
        out["g3_lip_compression"] = self._g3_lip_compression(cur, buf)
        out["g3_eye_block"] = self._g3_eye_block(buf)
        out["g3_jaw_clench"] = self._g3_jaw_clench(lm, cur, fr, buf)
        out["g3_rapid_blink"] = self._g3_rapid_blink()
        out["g3_gaze_aversion"] = self._g3_gaze_aversion(cur, buf)
        out["g3_no_nod"] = self._g3_no_nod(buf)
        out["g3_narrowed_pupils"] = self._g3_narrowed_pupils(cur, buf)  # proxy: squint
        out["g3_mouth_cover"] = 0.0  # no hand detection
        out["g3_lip_corner_dip"] = self._g3_lip_corner_dip(lm, cur, buf)
        out["g3_brow_lower_sustained"] = self._g3_brow_lower_sustained(buf)
        out["g3_eye_squeeze"] = self._g3_eye_squeeze(cur, buf)
        out["g3_head_shake"] = self._g3_head_shake(buf)

        # --- Group 4: Decision-Ready ---
        out["g4_relaxed_exhale"] = self._g4_relaxed_exhale(buf)
        out["g4_fixed_gaze"] = self._g4_fixed_gaze(buf, w, h)
        out["g4_smile_transition"] = self._g4_smile_transition(buf, out.get("g1_duchenne", 0))
        out["g4_mouth_relax"] = self._g4_mouth_relax(buf)
        out["g4_smile_sustain"] = self._g4_smile_sustain(buf, out.get("g1_duchenne", 0))

        # Continuous 0-100: clamp raw scores; no binary/hysteresis
        for k in out:
            v = float(out[k])
            if not np.isfinite(v):
                out[k] = 0.0
            else:
                out[k] = float(np.clip(v, 0.0, 100.0))

        # Phase C1: configurable diagnostic logging (off by default)
        try:
            import config
            if getattr(config, "ENGAGEMENT_DIAGNOSTIC_LOGGING", False):
                interval = max(1, int(getattr(config, "ENGAGEMENT_DIAGNOSTIC_LOG_INTERVAL", 30)))
                fc = getattr(self, "_diag_frame_count", 0)
                if interval and fc % interval == 0:
                    cur = buf[-1] if buf else {}
                    diag = {
                        "raw": {
                            "ear": cur.get("ear"),
                            "mar": cur.get("mar"),
                            "yaw": cur.get("yaw"),
                            "pitch": cur.get("pitch"),
                            "mouth_corner_asymmetry_ratio": cur.get("mouth_corner_asymmetry_ratio"),
                            "face_scale": cur.get("face_scale"),
                            "baseline_ear": getattr(self, "_baseline_ear", 0.0),
                            "baseline_mar": getattr(self, "_baseline_mar", 0.0),
                        },
                        "scores": {
                            "g1_duchenne": out.get("g1_duchenne"),
                            "g1_eye_contact": out.get("g1_eye_contact"),
                            "g3_contempt": out.get("g3_contempt"),
                            "g4_smile_transition": out.get("g4_smile_transition"),
                        },
                    }
                    import logging
                    logging.getLogger(__name__).info("engagement_diagnostic %s", diag)
        except Exception:
            pass

        return out

    def _all_keys(self) -> List[str]:
        """Return the list of all signifier keys (cached at module level)."""
        return SIGNIFIER_KEYS

    def _get_weights(self) -> Dict[str, List[float]]:
        """Fetch weights once per frame; reuse via get_group_means(..., W) / get_composite_score(..., W)."""
        n = len(SIGNIFIER_KEYS)
        return self._weights_provider() if self._weights_provider else {"signifier": [1.0] * n, "group": [0.35, 0.15, 0.35, 0.15]}

    def get_group_means(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
        """Return group means only (g1..g4). g3 is inverted (high = low resistance). No composite. Fast path for spike detection."""
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if W is None:
            W = self._get_weights()
        n_keys = len(keys)
        sw = W.get("signifier", [1.0] * n_keys)
        if len(sw) != n_keys:
            sw = [1.0] * n_keys

        def wmean(grp_keys: List[str]) -> float:
            total_w, total_ws = 0.0, 0.0
            for k in grp_keys:
                if k not in scores:
                    continue
                i = keys.index(k) if k in keys else 0
                wi = float(sw[i]) if i < len(sw) else 1.0
                total_w += wi * float(scores[k])
                total_ws += wi
            if total_ws > 1e-9:
                return total_w / total_ws
            vals = [float(scores[k]) for k in grp_keys if k in scores]
            return float(np.mean(vals)) if vals else 0.0

        g1k = ["g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
               "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
               "g1_micro_smile", "g1_brow_raise_sustained", "g1_mouth_open_receptive", "g1_eye_widening", "g1_nod_intensity"]
        g2k = ["g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
               "g2_stillness", "g2_lowered_brow", "g2_brow_furrow_deep", "g2_gaze_shift_frequency", "g2_mouth_tight_eval"]
        g3k = ["g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
               "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
               "g3_lip_corner_dip", "g3_brow_lower_sustained", "g3_eye_squeeze", "g3_head_shake"]
        g4k = ["g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition", "g4_mouth_relax", "g4_smile_sustain"]

        g1 = wmean(g1k)
        g2 = wmean(g2k)
        g3_raw = wmean(g3k)
        g3 = 100.0 - g3_raw
        g4 = wmean(g4k)
        return {"g1": g1, "g2": g2, "g3": g3, "g4": g4}

    def _composite_from_group_means(self, g: Dict[str, float], gw: List[float]) -> float:
        """
        Composite engagement from group means. Psychology-informed adjustments:
        - High interest+decision-ready (G1+G4): bonus (convergent positive signals)
        - High resistance (G3_raw): penalty (contempt, gaze aversion, etc.)
        """
        composite_raw = gw[0] * g["g1"] + gw[1] * g["g2"] + gw[2] * g["g3"] + gw[3] * g["g4"]
        if (g["g1"] + g["g4"]) / 2.0 > 62.0:
            composite_raw = min(100.0, composite_raw + 8.0)
        if (100.0 - g["g3"]) > 38.0:
            composite_raw = max(0.0, composite_raw - 10.0)
        return float(max(0.0, min(100.0, composite_raw)))

    def get_composite_score(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> float:
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if all(float(scores.get(k, 0)) == 0.0 for k in keys):
            return 0.0
        if W is None:
            W = self._get_weights()
        gw = W.get("group", [0.35, 0.15, 0.35, 0.15])
        if len(gw) != 4:
            gw = [0.35, 0.15, 0.35, 0.15]
        g = self.get_group_means(scores, W)
        return self._composite_from_group_means(g, gw)

    def _get_composite_raw_for_breakdown(self, scores: Optional[Dict[str, float]], gw: List[float], g: Dict[str, float]) -> float:
        composite_raw = gw[0] * g["g1"] + gw[1] * g["g2"] + gw[2] * g["g3"] + gw[3] * g["g4"]
        if (g["g1"] + g["g4"]) / 2.0 > 62.0:
            composite_raw = min(100.0, composite_raw + 8.0)
        if (100.0 - g["g3"]) > 38.0:
            composite_raw = max(0.0, composite_raw - 10.0)
        return float(max(0.0, min(100.0, composite_raw)))

    def get_composite_breakdown(self, scores: Optional[Dict[str, float]] = None, W: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Return a step-by-step breakdown of how the composite engagement score is calculated.
        Used for frontend "How is the score calculated?" and transparency.
        """
        if scores is None:
            scores = self.get_all_scores()
        keys = self._all_keys()
        if W is None:
            W = self._get_weights()
        sw = W.get("signifier", [1.0] * 30)
        gw = W.get("group", [0.35, 0.15, 0.35, 0.15])
        if len(sw) != 30:
            sw = [1.0] * 30
        if len(gw) != 4:
            gw = [0.35, 0.15, 0.35, 0.15]

        def weighted_mean(grp_keys: List[str]) -> float:
            total_w, total_ws = 0.0, 0.0
            for k in grp_keys:
                if k not in scores:
                    continue
                i = keys.index(k) if k in keys else 0
                wi = float(sw[i]) if i < len(sw) else 1.0
                si = float(scores[k])
                total_w += wi * si
                total_ws += wi
            if total_ws > 1e-9:
                return total_w / total_ws
            return float(np.mean([scores[k] for k in grp_keys if k in scores])) if grp_keys else 0.0

        g1k = ["g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
               "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
               "g1_micro_smile", "g1_brow_raise_sustained", "g1_mouth_open_receptive", "g1_eye_widening", "g1_nod_intensity"]
        g2k = ["g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
               "g2_stillness", "g2_lowered_brow", "g2_brow_furrow_deep", "g2_gaze_shift_frequency", "g2_mouth_tight_eval"]
        g3k = ["g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
               "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
               "g3_lip_corner_dip", "g3_brow_lower_sustained", "g3_eye_squeeze", "g3_head_shake"]
        g4k = ["g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition", "g4_mouth_relax", "g4_smile_sustain"]

        g1 = weighted_mean(g1k)
        g2 = weighted_mean(g2k)
        g3_raw_wmean = weighted_mean(g3k)
        g3 = 100.0 - g3_raw_wmean
        g4 = weighted_mean(g4k)

        composite_before = gw[0] * g1 + gw[1] * g2 + gw[2] * g3 + gw[3] * g4
        adjustments: List[str] = []
        composite_after = composite_before
        if (g1 + g4) / 2.0 > 62.0:
            composite_after = min(100.0, composite_after + 8.0)
            adjustments.append("(G1+G4)/2 > 62: +8")
        if g3_raw_wmean > 38.0:
            composite_after = max(0.0, composite_after - 10.0)
            adjustments.append("G3_raw > 38: -10")
        score = float(max(0.0, min(100.0, composite_after)))

        return {
            "formula": "score = clip(G1*w1 + G2*w2 + G3*w3 + G4*w4 + adjustments, 0, 100)",
            "groupWeights": {"G1": gw[0], "G2": gw[1], "G3": gw[2], "G4": gw[3]},
            "groupMeans": {"G1": round(g1, 2), "G2": round(g2, 2), "G3_raw": round(g3_raw_wmean, 2), "G3": round(g3, 2), "G4": round(g4, 2)},
            "compositeBeforeAdjustments": round(composite_before, 2),
            "adjustments": adjustments if adjustments else ["none"],
            "score": round(score, 1),
            "signifierScores": {k: round(float(scores.get(k, 0)), 1) for k in keys if k in scores},
        }

    # ----- Group 1 -----
    def _g1_duchenne(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Duchenne smile detector based on FACS AU 6 (cheek raise/eye squinch) + AU 12 (lip corner pull).
        Per Ekman & Friesen (1982), AU6+AU12 together indicate genuine (felt) positive affect vs. social smile.
        
        Output spans 0-100 after scaling:
        - Low (0-25): Neither corner lift nor squinch present
        - Medium (25-60): Only corner lift OR only squinch
        - High (60-100): Both corner lift AND squinch (genuine Duchenne smile)
        
        Returns raw 50-100; scaling (v-50)*2 yields display 0-100.
        """
        ear = cur.get("ear", 0.2)
        face_scale = max(1e-6, cur.get("face_scale", 50.0))
        mouth_pts = _safe(lm, MOUTH)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0 else 0.25
        
        if len(mouth_pts) < 6:
            return 50.0
        
        # === AU 12: Lip Corner Pull (0-1 intensity) ===
        ly = float(mouth_pts[0, 1])
        ry = float(mouth_pts[5, 1]) if len(mouth_pts) > 5 else ly
        upper_pts = mouth_pts[1:min(5, len(mouth_pts)), :]
        upper_y = float(np.mean(upper_pts[:, 1])) if len(upper_pts) > 0 else ly
        corner_lift = upper_y - (ly + ry) / 2.0
        lift_ratio = corner_lift / face_scale
        
        AU12_threshold = float(self._params.get("duchenne_au12_threshold", 0.006))
        au12_intensity = 0.0
        if lift_ratio > AU12_threshold:
            au12_intensity = min(1.0, (lift_ratio - AU12_threshold) / 0.05)
        
        # === AU 6: Cheek Raise / Eye Squinch (0-1 intensity) ===
        au6_intensity = 0.0
        if baseline_ear > 0.05:
            ear_ratio = ear / baseline_ear
            au6_lo = float(self._params.get("duchenne_au6_lo", 0.75))
            au6_hi = float(self._params.get("duchenne_au6_hi", 0.98))
            if au6_lo <= ear_ratio <= au6_hi:
                center = (au6_lo + au6_hi) / 2.0
                span = (au6_hi - au6_lo) / 2.0
                au6_intensity = 1.0 - abs(ear_ratio - center) / max(span, 0.01)
                au6_intensity = max(0.0, min(1.0, au6_intensity))
        
        # === Combine: neither=low, one=medium, both=high ===
        # Each signal contributes up to 25 raw points; synergy adds up to 25 when both present
        corner_contrib = au12_intensity * 25.0
        squinch_contrib = au6_intensity * 25.0
        synergy = 0.0
        if au12_intensity > 0.2 and au6_intensity > 0.2:
            synergy = min(25.0, au12_intensity * au6_intensity * 30.0)
        
        raw_score = 50.0 + corner_contrib + squinch_contrib + synergy
        return float(max(50.0, min(100.0, raw_score)))

    def _g1_pupil_dilation(self, cur: dict, buf: list) -> float:
        """
        Pupil dilation proxy via eye openness vs baseline.
        Research: Hess (1965)—pupil dilation correlates with interest/arousal. We use
        eye-area ratio (wider eyes = higher) as proxy since pupil measurement unavailable.
        
        FALSE POSITIVE REDUCTION: Uses median of recent eye-area values (excluding blinks)
        to avoid single-frame noise. Requires sustained change from baseline.
        """
        is_blink = cur.get("is_blink", 0.0)
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        
        if is_blink > 0.5:
            self._last_pupil_dilation_score = 0.0
            return 0.0
        
        # TEMPORAL: Use median of recent eye areas (exclude blinks)
        if len(buf) >= 3:
            recent_areas = [b.get("eye_area", area) for b in buf[-4:]
                           if b.get("is_blink", 0) < 0.5]
            if recent_areas:
                area = float(np.median(recent_areas))
        
        r = area / base if base > 0 else 1.0
        
        # Stricter thresholds: require larger deviation from baseline
        if r >= 1.12:  # Strong dilation (raised from 1.08)
            score = 65.0 + min(30.0, (r - 1.12) * 250.0)
        elif r >= 1.06:
            score = 54.0 + (r - 1.06) / 0.06 * 10.0
        elif r >= 1.02:
            score = 50.0 + (r - 1.02) / 0.04 * 4.0
        elif r >= 0.98:
            score = 48.0 + (r - 0.98) / 0.04 * 2.0  # Near baseline = neutral
        elif r >= 0.92:
            score = 40.0 + (r - 0.92) / 0.06 * 8.0
        else:
            score = max(0.0, 40.0 + (r - 0.92) * 300.0)
        
        self._last_pupil_dilation_score = score
        return float(max(0.0, min(100.0, score)))

    def _g1_eyebrow_flash(self, buf: list) -> float:
        """
        Eyebrow flash: rapid bilateral raise (~200ms). Research: Ekman; cross-cultural
        signal of recognition, openness, interest. During conversation = nonverbal "yes."
        
        FALSE POSITIVE REDUCTION: Requires BOTH clear raise AND return pattern to
        distinguish from head movements or noise. Uses stricter threshold for raise.
        """
        if len(buf) < 10:
            return 50.0  # Warmup
        heights = [(b["eyebrow_l"] + b["eyebrow_r"]) / 2 for b in buf]
        face_scale = buf[-1].get("face_scale", 50.0)
        threshold = face_scale * 0.028  # Stricter (raised from 0.022)
        
        # Use first half as baseline (stable reference)
        baseline = float(np.median(heights[:max(2, len(heights) // 2)]))  # Median for robustness
        cur = heights[-1]
        recent_heights = heights[-5:]
        max_recent = float(np.max(recent_heights))
        
        raised = max_recent > baseline + threshold
        # Returning: current is clearly below peak (completed flash)
        returning = cur < max_recent - threshold * 0.5
        
        # Require BOTH raise AND return for high score (complete flash pattern)
        if raised and returning:
            flash_magnitude = (max_recent - baseline) / max(face_scale * 0.12, 1e-6)
            return 58.0 + min(32.0, flash_magnitude * 100.0)
        elif raised and cur > baseline + threshold * 0.6:  # Still raised
            raise_magnitude = (cur - baseline) / max(face_scale * 0.12, 1e-6)
            return 50.0 + min(20.0, raise_magnitude * 55.0)
        # Small sustained raise (not a flash, but mild interest)
        if cur > baseline + threshold * 0.6:
            return 50.0 + min(6.0, (cur - baseline) / max(face_scale * 0.10, 1e-6) * 20.0)
        return 50.0

    def _g1_eye_contact(self, cur: dict, w: int, h: int, buf: list) -> float:
        """
        Sustained eye contact: head orientation + face position.
        Research: eye-mind hypothesis; shared signal hypothesis; direct gaze = engagement.
        Social inclusion delays disengagement from direct gaze. Sustained facing = active listening.
        Uses median of recent yaw/pitch for robustness.
        """
        gx = cur.get("gaze_x", 0)
        gy = cur.get("gaze_y", 0)
        yaw = cur.get("yaw", 0)   # No median smoothing: real-time tracking
        pitch = cur.get("pitch", 0)   # No median smoothing: real-time tracking
        face_scale = cur.get("face_scale", 50.0)

        # Primary: head orientation toward camera (yaw/pitch near 0 = looking at camera)
        # Score 100 when head is forward; decay by angle (wider tolerance so normal movement still scores high)
        yaw_deg = abs(float(yaw))
        pitch_deg = abs(float(pitch))
        # 0° = 100; ~25° yaw or ~20° pitch = 50; 50°+ = 0
        head_score = 100.0 - min(100.0, (yaw_deg / 25.0) * 50.0 + (pitch_deg / 20.0) * 50.0)

        # Secondary: face position in frame (eye center vs frame center); cap effect so head dominates
        gaze_dist = np.sqrt(float(gx) * float(gx) + float(gy) * float(gy))
        gaze_normalized = min(1.0, gaze_dist / max(face_scale * 0.45, 1e-6))  # 45% = neutral; less strict
        gaze_bonus = (1.0 - gaze_normalized) * 15.0  # Up to +15 when face centered

        base_score = min(100.0, head_score + gaze_bonus)

        if len(buf) < 2:
            return float(max(0.0, min(100.0, base_score)))

        # Same formula for each frame in history (consistent with base_score)
        def _frame_eye_contact(frame: dict) -> float:
            gxf = frame.get("gaze_x", 0)
            gyf = frame.get("gaze_y", 0)
            yaw_f = abs(float(frame.get("yaw", 0)))
            pitch_f = abs(float(frame.get("pitch", 0)))
            fs = max(face_scale, frame.get("face_scale", 50.0), 20.0)
            head_s = 100.0 - min(100.0, (yaw_f / 25.0) * 50.0 + (pitch_f / 20.0) * 50.0)
            gd = np.sqrt(float(gxf) * float(gxf) + float(gyf) * float(gyf))
            gn = min(1.0, gd / max(fs * 0.45, 1e-6))
            g_bonus = (1.0 - gn) * 15.0
            return min(100.0, head_s + g_bonus)

        ref = self._FPS_REF
        win_eye = max(9, int(18 * self._fps / ref))
        recent_frames = buf[-win_eye:] if len(buf) >= win_eye else buf
        frame_scores = [_frame_eye_contact(f) for f in recent_frames]

        # Count consecutive frames with good eye contact from current backward (lower threshold = more sustained)
        good_threshold = 50.0
        sustained_period = 0
        for s in reversed(frame_scores):
            if s > good_threshold:
                sustained_period += 1
            else:
                break

        avg_recent = float(np.mean(frame_scores)) if frame_scores else base_score

        t_14 = max(7, int(14 * self._fps / ref))
        t_10 = max(5, int(10 * self._fps / ref))
        t_6 = max(3, int(6 * self._fps / ref))
        t_3 = max(2, int(3 * self._fps / ref))
        if sustained_period >= t_14:
            sustained_bonus = 28.0
        elif sustained_period >= t_10:
            sustained_bonus = 18.0 + (sustained_period - t_10) * 2.5
        elif sustained_period >= t_6:
            sustained_bonus = 8.0 + (sustained_period - t_6) * 2.5
        elif sustained_period >= t_3:
            sustained_bonus = 2.0 + (sustained_period - t_3) * 2.0
        else:
            sustained_bonus = 0.0

        consistency_bonus = min(8.0, max(0.0, (avg_recent - 55.0) / 45.0 * 8.0)) if avg_recent > 55.0 else 0.0

        final_score = base_score + sustained_bonus + consistency_bonus
        return float(max(0.0, min(100.0, final_score)))

    def _g1_head_tilt(self, cur: dict, buf: list) -> float:
        """
        Head tilt (lateral/roll): engagement signal from psychology research.
        
        Research basis (Davidenko et al., 2018; UC Santa Cruz / Perception):
        - Lateral tilt as small as 11° facilitates social engagement; makes faces more approachable.
        - Tilt exposes neck (trust/vulnerability); signals active listening vs passive.
        - Optimal band ~6-18°; extreme (>35°) can indicate confusion or unnatural pose.
        
        FALSE POSITIVE REDUCTION: Requires sustained tilt (checked via buffer) to avoid
        scoring high on momentary head movements or landmark noise.
        
        Returns raw 50-100; 50 = no tilt (display 0), high = tilt present (engagement).
        """
        roll = abs(float(cur.get("roll", 0)))
        
        # TEMPORAL CHECK: Use median of recent rolls to avoid single-frame spikes
        if len(buf) >= 4:
            recent_rolls = [abs(float(b.get("roll", 0))) for b in buf[-4:]]
            roll = float(np.median(recent_rolls))
        
        # No tilt (< 3.5°): neutral/passive; raw 50 -> display 0 (raised from 2.5)
        if roll < 3.5:
            return 50.0
        # Subtle tilt (3.5-7°): onset of interest
        if 3.5 <= roll < 7.0:
            return 50.0 + (roll - 3.5) / 3.5 * 14.0  # 50->64
        # Optimal band (7-18°): engagement/curiosity peak (Davidenko: 11° meaningful)
        if 7.0 <= roll <= 18.0:
            peak = 11.0
            dist = abs(roll - peak)
            return 68.0 + (1.0 - min(1.0, dist / 6.0)) * 24.0  # 68-92, peak at 11°
        # Strong tilt (18-28°): clear engagement, ramp down
        if 18.0 < roll <= 28.0:
            return 64.0 - (roll - 18.0) * 1.4  # 64->50
        # Very strong (28-40°): possible confusion; low-mid
        if 28.0 < roll <= 40.0:
            return 50.0 - (roll - 28.0) * 0.3  # 50->46
        # Extreme (> 40°): "too weird" per research
        return 45.0

    def _g1_forward_lean(self, cur: dict, buf: list) -> float:
        """
        Forward lean: approach motivation proxy. Research: Riskind & Gotay; embodied
        approach—leaning forward increases left frontal activation to appetitive stimuli,
        amplifies approach motivation. Lean toward = desire, interest, engagement.
        
        FALSE POSITIVE REDUCTION: Uses baseline and requires sustained lean (median of
        recent frames) to avoid scoring on momentary movements or noise.
        """
        z = cur.get("face_z", 0)
        base = self._baseline_z
        if base == 0 or z == 0:
            return 50.0
        
        # TEMPORAL: Use median of recent Z values for stability
        if len(buf) >= 4:
            recent_z = [b.get("face_z", z) for b in buf[-4:]]
            z = float(np.median(recent_z))
        
        # z < baseline = lean toward camera; smaller ratio = stronger lean
        # Require at least 2% change to avoid noise (raised from 1%)
        if z < base * 0.98:
            return 50.0 + min(48.0, (1.0 - z / base) * 180.0)  # Reduced sensitivity
        return 50.0

    def _g1_facial_symmetry(self, lm: np.ndarray, buf: list) -> float:
        """
        Facial Symmetry: Measures bilateral symmetry of facial features along vertical center line.
        
        High symmetry (balanced facial expression) indicates focused engagement.
        Low symmetry (asymmetric expression) may indicate distraction, discomfort, or lack of focus.
        
        Calculation:
        - Checks symmetry of eyes, eyebrows, mouth, and nose alignment
        - Mirrors right side features across face center (vertical midline) and compares to left
        - Measures both horizontal (75% weight) and vertical (25% weight) symmetry
        - Normalizes errors by face scale (inter-ocular distance) for size invariance
        - Uses realistic scoring curve: <3% error = 90-100, 3-6% = 70-90, 6-10% = 50-70, etc.
        - Accounts for natural facial asymmetry (most faces have 3-8% natural asymmetry)
        """
        if len(buf) < 2 or lm.shape[0] < 20:
            return 50.0
        
        face_scale = buf[-1].get("face_scale", 50.0)
        
        # Extract key facial features
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        
        if len(le) < 3 or len(re) < 3:
            return 50.0
        
        # Calculate feature centers
        left_eye_center = np.mean(le[:, :2], axis=0)
        right_eye_center = np.mean(re[:, :2], axis=0)
        
        # Eyebrow centers (if available)
        left_brow_center = np.mean(lb[:, :2], axis=0) if len(lb) >= 2 else left_eye_center + np.array([0, -10])
        right_brow_center = np.mean(rb[:, :2], axis=0) if len(rb) >= 2 else right_eye_center + np.array([0, -10])
        
        # Mouth corners
        ml = lm[MOUTH_LEFT, :2] if lm.shape[0] > MOUTH_LEFT else np.zeros(2)
        mr = lm[MOUTH_RIGHT, :2] if lm.shape[0] > MOUTH_RIGHT else np.zeros(2)
        
        # Nose tip for vertical alignment check
        nose_tip = lm[NOSE_TIP, :2] if lm.shape[0] > NOSE_TIP else (left_eye_center + right_eye_center) / 2.0
        
        # Calculate face center (vertical midline)
        face_center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        
        # Mirror right side features across center and compare to left
        # For perfect symmetry: left_x = 2*center_x - right_x
        right_eye_mirrored_x = 2 * face_center_x - right_eye_center[0]
        right_brow_mirrored_x = 2 * face_center_x - right_brow_center[0]
        right_mouth_mirrored_x = 2 * face_center_x - mr[0]
        
        # Calculate symmetry errors in pixels first
        eye_symmetry_error_x_px = abs(left_eye_center[0] - right_eye_mirrored_x)
        brow_symmetry_error_x_px = abs(left_brow_center[0] - right_brow_mirrored_x)
        mouth_symmetry_error_x_px = abs(ml[0] - right_mouth_mirrored_x)
        
        # Vertical alignment check (nose should be centered between eyes)
        eye_midpoint_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        nose_alignment_error_px = abs(nose_tip[0] - eye_midpoint_x)
        
        # Vertical symmetry: check if features are at similar heights
        eye_height_diff_px = abs(left_eye_center[1] - right_eye_center[1])
        brow_height_diff_px = abs(left_brow_center[1] - right_brow_center[1])
        mouth_height_diff_px = abs(ml[1] - mr[1])
        
        # Normalize errors by face scale (inter-ocular distance)
        # This makes the metric invariant to face size and distance from camera
        face_scale_safe = max(face_scale, 20.0)  # Ensure minimum scale
        
        eye_symmetry_error_x = eye_symmetry_error_x_px / face_scale_safe
        brow_symmetry_error_x = brow_symmetry_error_x_px / face_scale_safe
        mouth_symmetry_error_x = mouth_symmetry_error_x_px / face_scale_safe
        nose_alignment_error = nose_alignment_error_px / face_scale_safe
        
        eye_height_diff = eye_height_diff_px / face_scale_safe
        brow_height_diff = brow_height_diff_px / face_scale_safe
        mouth_height_diff = mouth_height_diff_px / face_scale_safe
        
        # Combine all symmetry errors with weights
        # Eyes are most important (40%), then mouth (30%), brows (20%), nose (10%)
        horizontal_error = (
            0.4 * eye_symmetry_error_x +
            0.3 * mouth_symmetry_error_x +
            0.2 * brow_symmetry_error_x +
            0.1 * nose_alignment_error
        )
        
        # Vertical symmetry (less weight, as natural faces have some vertical asymmetry)
        vertical_error = (
            0.5 * eye_height_diff +
            0.3 * mouth_height_diff +
            0.2 * brow_height_diff
        )
        
        # Combined error (horizontal is more important for symmetry perception)
        total_error = 0.75 * horizontal_error + 0.25 * vertical_error
        
        # Convert error to score using a realistic curve that spans 0-100
        # Real faces typically have 3-8% natural asymmetry
        # Perfect symmetry (0% error) = 100
        # Excellent symmetry (< 3% error) = 90-100
        # Good symmetry (3-6% error) = 70-90
        # Moderate symmetry (6-10% error) = 50-70
        # Poor symmetry (10-15% error) = 30-50
        # Very poor symmetry (> 15% error) = 0-30
        
        # Finer bands: small asymmetry changes produce larger score changes
        if total_error < 0.02:
            symmetry_score = 100.0 - (total_error / 0.02) * 8.0
        elif total_error < 0.045:
            symmetry_score = 92.0 - ((total_error - 0.02) / 0.025) * 22.0
        elif total_error < 0.08:
            symmetry_score = 70.0 - ((total_error - 0.045) / 0.035) * 22.0
        elif total_error < 0.12:
            symmetry_score = 48.0 - ((total_error - 0.08) / 0.04) * 20.0
        else:
            capped_error = min(total_error, 0.28)
            symmetry_score = max(0.0, 28.0 - ((capped_error - 0.12) / 0.16) * 28.0)
        
        return float(max(0.0, min(100.0, symmetry_score)))

    def _g1_micro_smile(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """Brief lip-corner raise; receptivity (Ekman micro-expressions). Lighter than Duchenne."""
        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 6:
            return 50.0
        face_scale = max(1e-6, cur.get("face_scale", 50.0))
        ly, ry = float(mouth_pts[0, 1]), float(mouth_pts[5, 1]) if len(mouth_pts) > 5 else float(mouth_pts[0, 1])
        upper_y = float(np.mean(mouth_pts[1:min(5, len(mouth_pts)), 1]))
        corner_lift = upper_y - (ly + ry) / 2.0
        lift_ratio = corner_lift / face_scale
        if lift_ratio > 0.006:
            return 50.0 + min(40.0, (lift_ratio - 0.006) * 800.0)
        return 50.0

    def _g1_brow_raise_sustained(self, buf: list) -> float:
        """Sustained brow raise = interest/curiosity (FACS AU2)."""
        if len(buf) < 8:
            return 50.0
        heights = [(b.get("eyebrow_l", 0) + b.get("eyebrow_r", 0)) / 2 for b in buf[-8:]]
        face_scale = buf[-1].get("face_scale", 50.0)
        baseline = float(np.median(heights[:4])) if len(heights) >= 4 else heights[0]
        cur_avg = float(np.mean(heights[-4:])) if len(heights) >= 4 else heights[-1]
        raise_amt = (cur_avg - baseline) / max(face_scale * 0.08, 1e-6)
        if raise_amt > 0.3:
            return 55.0 + min(35.0, raise_amt * 50.0)
        return 50.0

    def _g1_mouth_open_receptive(self, cur: dict, buf: list) -> float:
        """Slight jaw drop = receptivity (Waller 2024 expressiveness)."""
        mar = cur.get("mar", 0.2)
        if len(buf) >= 3:
            mar = float(np.median([b.get("mar", mar) for b in buf[-3:]]))
        if 0.10 <= mar <= 0.28:
            return 52.0 + (mar - 0.10) / 0.18 * 36.0
        return 48.0

    def _g1_eye_widening(self, cur: dict, buf: list) -> float:
        """Wider eyes = interest/arousal (Hess)."""
        if cur.get("is_blink", 0) > 0.5:
            return 0.0
        area = cur.get("eye_area", 0.0)
        base = max(1e-6, self._baseline_eye_area)
        if base <= 0:
            return 50.0
        r = area / base
        if r >= 1.08:
            return 55.0 + min(35.0, (r - 1.08) * 200.0)
        if r >= 1.02:
            return 50.0 + (r - 1.02) / 0.06 * 8.0
        return 48.0

    def _g1_nod_intensity(self, buf: list) -> float:
        """Nod magnitude; agreement strength (Wells & Petty)."""
        if len(buf) < 12:
            return 40.0
        pitches = [b["pitch"] for b in buf[-12:]]
        pitch_std = float(np.std(pitches))
        d = np.diff(pitches)
        crosses = int(np.sum((d[:-1] * d[1:]) < 0))
        if 1 <= crosses <= 5 and pitch_std > 0.5:
            return 52.0 + min(30.0, pitch_std * 15.0 + crosses * 3.0)
        return 42.0

    def _g1_rhythmic_nodding(self, buf: list) -> float:
        """
        Rhythmic nodding: vertical pitch oscillations. Research: Wells & Petty (1980)—
        nodding increases persuasion (self-validation). Nodding = agreement, engagement.
        We detect pitch oscillations (zero-crossings) as nodding rhythm.
        """
        if len(buf) < 12:
            return 40.0
        pitches = [b["pitch"] for b in buf[-12:]]
        pitch_std = float(np.std(pitches))
        d = np.diff(pitches)
        crosses = int(np.sum((d[:-1] * d[1:]) < 0))
        if pitch_std < 0.6:
            return 34.0 + pitch_std * 13.0
        nod_bonus = min(26.0, (crosses * 6.0) + (pitch_std - 0.6) * 20.0) if (1 <= crosses <= 4) else 0.0
        base = 42.0 + (pitch_std - 0.6) / 0.6 * 8.0 if pitch_std < 1.2 else 50.0 - (pitch_std - 1.2) * 5.0
        return float(max(34.0, min(100.0, base + nod_bonus)))

    def _g1_parted_lips(self, cur: dict, buf: list) -> float:
        """
        Parted lips: slight mouth opening. Research: open mouth = receptivity, listening,
        preparation to speak. Closed = withholding; parted = engaged, attentive.
        
        FALSE POSITIVE REDUCTION: Uses median of recent MAR values to avoid triggering
        on speech artifacts or single-frame noise.
        """
        mar = cur.get("mar", 0.2)
        
        # TEMPORAL: Use median of recent MAR for stability (avoid speech artifacts)
        if len(buf) >= 3:
            recent_mar = [b.get("mar", mar) for b in buf[-3:]]
            mar = float(np.median(recent_mar))
        
        # Narrower optimal band to avoid false positives
        if 0.12 <= mar <= 0.32:
            return 48.0 + (mar - 0.12) / 0.20 * 42.0  # 48-90
        if 0.08 <= mar < 0.12:
            return 40.0 + (mar - 0.08) / 0.04 * 8.0
        if 0.32 < mar <= 0.48:
            return 80.0 - (mar - 0.32) / 0.16 * 26.0  # Ramp down for wide open
        return 38.0 + min(8.0, mar * 20.0)

    def _g1_softened_forehead(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Softened forehead: relaxed brow (low tension). Research: FACS AU 4 (brow lowerer)
        absent = relaxed. Low brow variance + even inner/outer = not furrowed = relaxed engagement.
        
        FALSE POSITIVE REDUCTION: Uses median of recent brow measurements and requires
        sustained relaxation pattern to distinguish from momentary expressions.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 50.0
        face_scale = max(20.0, cur.get("face_scale", 50.0))

        # 1) Variance of brow points (low = flat = relaxed)
        v = float(np.var(np.vstack([lb[:, :2], rb[:, :2]])))
        v_normalized = v / max(face_scale * face_scale, 1e-6)

        # 2) Inner vs outer brow height: relaxed = similar height; furrowed = inner pulled down
        left_inner_y = float(lb[np.argmin(lb[:, 0]), 1])
        left_outer_y = float(lb[np.argmax(lb[:, 0]), 1])
        right_inner_y = float(rb[np.argmin(rb[:, 0]), 1])
        right_outer_y = float(rb[np.argmax(rb[:, 0]), 1])
        inner_outer_diff_left = left_inner_y - left_outer_y
        inner_outer_diff_right = right_inner_y - right_outer_y
        diff_normalized = (abs(inner_outer_diff_left) + abs(inner_outer_diff_right)) / max(face_scale * 0.15, 1e-6)
        evenness = max(0.0, 1.0 - min(1.0, diff_normalized))

        # Azure-friendly: when only 2 points per brow (4 unique), use flat-brow metric from evenness only
        brow_pts = np.vstack([lb[:, :2], rb[:, :2]])
        unique_rows = np.unique(np.round(brow_pts).astype(np.int32), axis=0)
        few_distinct_brow_points = len(unique_rows) <= 4

        # TEMPORAL: Check consistency over recent frames for stability
        sustained_relaxation = True
        if len(buf) >= 4:
            # Check if brow has been consistently even (not temporarily relaxed)
            recent_bl = [b.get("eyebrow_l", 0) for b in buf[-4:]]
            recent_br = [b.get("eyebrow_r", 0) for b in buf[-4:]]
            if recent_bl and recent_br:
                brow_std = float(np.std(recent_bl)) + float(np.std(recent_br))
                # Relaxed threshold so normal jitter doesn't disable sustained (was 0.04)
                if brow_std > face_scale * 0.08:
                    sustained_relaxation = False

        # Score from variance: low v_norm -> high (relaxed)
        # Azure path: 2 points per brow (4 unique) -> use evenness only so we don't penalize low point count
        if few_distinct_brow_points:
            var_score = 65.0 + 17.0 * evenness
        else:
            # Relaxed bands so typical/Azure-expanded brows get 60–82 range (was 0.0006/0.0025/0.006)
            if v_normalized < 0.002 and sustained_relaxation:
                var_score = 82.0
            elif v_normalized < 0.008:
                var_score = 68.0 + (0.008 - v_normalized) / 0.006 * 12.0
            elif v_normalized < 0.018:
                var_score = 52.0 + (0.018 - v_normalized) / 0.010 * 14.0
            else:
                # Gentler decay and higher floor so moderate variance still scores above 48
                var_score = max(44.0, 58.0 - (v_normalized - 0.018) * 400.0)

        # Combine: 60% variance (flat brow), 40% evenness
        raw = 50.0 + 0.5 * (var_score - 50.0) + 15.0 * evenness
        # Soft penalty when not sustained so jitter doesn't force score below 48 (was 0.85)
        if not sustained_relaxation:
            raw = raw * 0.92
        return float(max(32.0, min(90.0, raw)))

    # ----- Group 2 -----
    def _g2_look_up_lr(self, cur: dict, buf: list) -> float:
        """
        Look up/left/right: cognitive load cue. Research: gaze aversion during difficult
        tasks; look-up-left (NLU) = visual/constructed imagery. Eye-mind link: gaze
        reflects cognitive processing. Upward/lateral = thinking, accessing memory.
        
        FALSE POSITIVE REDUCTION: Uses median of recent pitch/yaw and requires
        sustained gaze shift (not momentary glance). Stricter thresholds.
        """
        p = float(cur.get("pitch", 0))
        y = float(cur.get("yaw", 0))
        
        # TEMPORAL: Use median of recent values for stability
        if len(buf) >= 3:
            recent_pitch = [b.get("pitch", p) for b in buf[-3:]]
            recent_yaw = [b.get("yaw", y) for b in buf[-3:]]
            p = float(np.median(recent_pitch))
            y = float(np.median(recent_yaw))
        
        # Stricter thresholds: require clearer gaze shifts
        looking_up = p > 5  # Raised from 3
        looking_lr = abs(y) > 8  # Raised from 6
        
        if looking_up and looking_lr:
            up_intensity = min(1.0, (p - 5) / 18.0)
            lr_intensity = min(1.0, (abs(y) - 8) / 28.0)
            combined = (up_intensity + lr_intensity) / 2.0
            return 52.0 + combined * 38.0
        elif looking_up:
            up_intensity = min(1.0, (p - 5) / 12.0)
            return 50.0 + up_intensity * 28.0
        elif looking_lr:
            lr_intensity = min(1.0, (abs(y) - 8) / 30.0)
            return 48.0 + lr_intensity * 16.0
        return 45.0

    def _g2_lip_pucker(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """
        Lip Pucker: Detects pursed lips (thinking, evaluating expression).
        
        Lip pucker is characterized by:
        - High MAR (mouth aspect ratio - vertical opening relative to width)
        - Narrow mouth width (lips pursed together)
        - Both conditions must be met for a pucker
        
        FALSE POSITIVE REDUCTION: Uses median of recent MAR/mouth width to avoid
        triggering on speech artifacts. Requires sustained pucker pattern.
        
        Returns 0-100 where:
        - 0-30: No pucker (normal mouth)
        - 30-60: Slight pucker
        - 60-80: Moderate pucker
        - 80-100: Strong pucker
        """
        mar = cur.get("mar", 0.2)
        face_scale = cur.get("face_scale", 50.0)
        mw = cur.get("mouth_w", 40.0)
        
        # TEMPORAL: Use median of recent values to avoid speech artifacts
        if len(buf) >= 3:
            recent_mar = [b.get("mar", mar) for b in buf[-3:]]
            recent_mw = [b.get("mouth_w", mw) for b in buf[-3:]]
            mar = float(np.median(recent_mar))
            mw = float(np.median(recent_mw))
        
        # Normalize mouth width by face scale for size invariance
        mw_normalized = mw / max(face_scale, 1e-6)
        
        # Normal mouth characteristics:
        # - MAR typically 0.15-0.25 for relaxed/neutral mouth
        # - Mouth width normalized: typically 0.20-0.35 of face scale
        
        # Lip pucker requires BOTH:
        # 1. High MAR (vertical opening) - indicates lips are pursed forward
        # 2. Narrow mouth width - indicates lips are compressed horizontally
        
        # No pucker: normal MAR and normal width (lower MAR threshold for sensitivity)
        if mar <= 0.22 and mw_normalized >= 0.19:
            return 10.0 + (mar / 0.22) * 12.0
        if mar <= 0.28:
            # MAR not high enough for pucker
            if mw_normalized < 0.15:
                # Narrow but not puckered (might be speaking or other expression)
                return 20.0 + (0.30 - mar) / 0.05 * 10.0  # 20-30
            else:
                # Normal mouth
                return 5.0 + (mar / 0.30) * 15.0  # 5-20
        
        if mw_normalized >= 0.20:
            return 15.0 + min(18.0, (mar - 0.28) / 0.08 * 18.0)
        
        # High MAR and narrow mouth = pucker (lower MAR threshold 0.28)
        if mar > 0.38 and mw_normalized < 0.12:
            pucker_intensity = min(1.0, (mar - 0.38) / 0.14)
            return 86.0 + pucker_intensity * 14.0
        elif mar > 0.32 and mw_normalized < 0.15:
            pucker_intensity = (mar - 0.32) / 0.06
            return 70.0 + pucker_intensity * 16.0
        elif mar > 0.28 and mw_normalized < 0.18:
            mar_factor = (mar - 0.28) / 0.04
            width_factor = (0.18 - mw_normalized) / 0.06
            pucker_intensity = (mar_factor + width_factor) / 2.0
            return 50.0 + pucker_intensity * 24.0
        else:
            mar_contribution = max(0.0, (mar - 0.22) / 0.10)
            width_contribution = max(0.0, (0.20 - mw_normalized) / 0.08)
            combined = (mar_contribution + width_contribution) / 2.0
            return 28.0 + combined * 24.0

    def _g2_eye_squint(self, cur: dict, buf: list) -> float:
        """
        Eye squint: narrowed eyes. Research: FACS AU 7; squinting = skepticism, distrust,
        evaluation. Combined with pursed lips = doubt. Cognitive load increases squinting.
        
        FALSE POSITIVE REDUCTION: Compares to baseline EAR and uses median of recent
        frames to avoid triggering on blinks or single-frame noise.
        """
        ear = cur.get("ear", 0.2)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0.05 else 0.22
        
        # TEMPORAL: Use median of recent EAR for stability
        if len(buf) >= 3:
            recent_ear = [b.get("ear", ear) for b in buf[-3:] if b.get("is_blink", 0) < 0.5]
            if recent_ear:
                ear = float(np.median(recent_ear))
        
        # Compare to baseline: squint = EAR below baseline
        ear_ratio = ear / baseline_ear if baseline_ear > 0 else 1.0
        
        # Only score high when clearly below baseline (not just naturally narrow eyes)
        if ear_ratio < 0.75 and ear < 0.15:  # Strong squint
            return 68.0 + min(24.0, (0.75 - ear_ratio) * 120.0)
        if ear_ratio < 0.85 and ear < 0.17:  # Moderate squint
            return 52.0 + (0.85 - ear_ratio) * 160.0
        if ear_ratio < 0.92:  # Slight squint
            return 42.0 + (0.92 - ear_ratio) * 100.0
        return 38.0  # Normal eyes

    def _g2_thinking_brow(self, lm: np.ndarray, buf: list) -> float:
        """
        Thinking brow: asymmetric brow raise. Research: one brow raised = curiosity,
        skepticism, or concentration. Different from bilateral flash; sustained asymmetry
        = evaluation, doubt.
        
        FALSE POSITIVE REDUCTION: Requires sustained asymmetry (not single-frame) and
        meaningful threshold relative to face scale. Natural brow asymmetry filtered.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        if len(lb) < 2 or len(rb) < 2:
            return 45.0
        ly, ry = float(np.mean(lb[:, 1])), float(np.mean(rb[:, 1]))
        d = abs(ly - ry)
        
        # Get face scale from buffer if available
        if len(buf) > 0:
            face_scale = buf[-1].get("face_scale", 50.0)
        else:
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        d_rel = d / face_scale
        
        # TEMPORAL: Check if asymmetry is sustained (not just single-frame noise)
        if len(buf) >= 3:
            recent_asymmetries = []
            for b in buf[-3:]:
                # Estimate brow asymmetry from eyebrow heights stored in buffer
                bl = b.get("eyebrow_l", 0)
                br = b.get("eyebrow_r", 0)
                fs = b.get("face_scale", 50.0)
                if fs > 0:
                    recent_asymmetries.append(abs(bl - br) / max(fs, 20.0))
            # Require sustained asymmetry (at least 2 of 3 frames)
            high_asymmetry_count = sum(1 for a in recent_asymmetries if a > 0.025)
            if high_asymmetry_count < 2:
                return 44.0
        
        # Stricter thresholds: require meaningful asymmetry
        if 0.04 <= d_rel <= 0.18:  # Clear asymmetry
            return 68.0 + min(14.0, (d_rel - 0.04) * 100.0)
        if 0.025 <= d_rel < 0.04:  # Slight asymmetry
            return 52.0 + (d_rel - 0.025) * 400.0
        return 44.0  # Natural/no asymmetry

    def _g2_stillness(self, buf: list) -> float:
        """
        Stillness: low facial movement. Research: can indicate focused attention or
        frozen/withdrawn state. Low face_var = minimal landmark movement. Context-dependent.
        
        FALSE POSITIVE REDUCTION: Stillness is ambiguous (could be focused listening
        or frozen state). Use more conservative scoring to avoid flagging attentive
        listeners. Require very low variance for extended period.
        """
        if len(buf) < 10:
            return 50.0  # Warmup
        vars_list = [b["face_var"] for b in buf[-10:]]
        scales = [b.get("face_scale", 50.0) for b in buf[-10:]]
        avg_scale = float(np.mean(scales)) if scales else 50.0
        avg_scale_sq = max(400.0, avg_scale * avg_scale)
        m = float(np.median(vars_list))  # Median: robust to single-frame movement spikes
        m_normalized = m / avg_scale_sq
        
        # Continuous: very low variance = high score; ramp down smoothly with movement
        if m_normalized < 0.0008:
            return 68.0 + (0.0008 - m_normalized) / 0.0008 * 12.0
        if m_normalized < 0.0025:
            return 55.0 + (0.0025 - m_normalized) / 0.0017 * 12.0
        if m_normalized < 0.005:
            return 46.0 + (0.005 - m_normalized) / 0.0025 * 8.0
        # Normal movement: smooth ramp 42–46 as variance increases
        return 42.0 + max(0.0, (0.012 - m_normalized)) / 0.007 * 4.0

    def _g2_lowered_brow(self, lm: np.ndarray, buf: list) -> float:
        """
        Lowered brow (furrowed): FACS AU 4. Research: corrugator activation = concentration,
        cognitive load, OR frustration/anger. CONTEXT-DEPENDENT: AU4 can signal (1) effortful
        thinking (G2), (2) problem understanding (communicative signal), or (3) negative
        affect (G3). This metric is in G2 (cognitive load); interpret in combination with
        other G2/G3 signals. Furrowed = thinking hard or negative affect.
        
        FALSE POSITIVE REDUCTION: Compares to recent baseline brow position (not absolute)
        and requires sustained lowering. Natural brow position varies per person.
        """
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        if len(lb) < 2 or len(le) < 2:
            return 46.0
        
        brow_y = (float(np.mean(lb[:, 1])) + float(np.mean(rb[:, 1]))) / 2
        eye_y = (float(np.mean(le[:, 1])) + float(np.mean(re[:, 1]))) / 2
        dist = eye_y - brow_y  # Smaller distance = furrowed (brow lowered)
        
        # Get face scale
        if len(buf) > 0:
            face_scale = buf[-1].get("face_scale", 50.0)
        else:
            face_scale = float(np.max(lm[:, 0]) - np.min(lm[:, 0])) * 0.5 if lm.shape[0] > 0 else 50.0
        face_scale = max(20.0, face_scale)
        dist_rel = dist / face_scale
        
        # BASELINE: Estimate person's typical brow-eye distance from history
        baseline_dist_rel = dist_rel
        if len(buf) >= 8:
            # Use median of first half of buffer as baseline
            half = max(4, len(buf) // 2)
            baseline_dists = []
            for b in buf[:half]:
                bl = b.get("eyebrow_l", 0)
                br = b.get("eyebrow_r", 0)
                fs = b.get("face_scale", 50.0)
                # Approximate: eyebrow height stored; lower = furrowed
                if fs > 0:
                    baseline_dists.append((bl + br) / (2 * max(fs, 20.0)))
            if baseline_dists:
                baseline_dist_rel = float(np.median(baseline_dists)) * 0.15 + 0.18  # Approximate
        
        # Compare to baseline: lower than typical = furrowed
        furrowed_amount = baseline_dist_rel - dist_rel
        
        # Continuous: furrowed amount and dist_rel drive score smoothly
        if dist_rel < 0.12 and furrowed_amount > 0.04:
            return 75.0 + min(15.0, furrowed_amount * 200.0)
        if dist_rel < 0.16 and furrowed_amount > 0.02:
            return 58.0 + (0.16 - dist_rel) / 0.04 * 14.0
        if dist_rel < 0.20:
            return 48.0 + (0.20 - dist_rel) / 0.04 * 8.0
        # Near neutral: smooth ramp so small changes in dist_rel still change score
        return 44.0 + max(0.0, 0.22 - dist_rel) * 25.0

    def _g2_brow_furrow_deep(self, lm: np.ndarray, buf: list) -> float:
        """Strong AU4 = cognitive load or frustration."""
        lb = _safe(lm, LEFT_EYEBROW)
        rb = _safe(lm, RIGHT_EYEBROW)
        le = _safe(lm, LEFT_EYE)
        re = _safe(lm, RIGHT_EYE)
        if len(lb) < 2 or len(le) < 2:
            return 46.0
        brow_y = (float(np.mean(lb[:, 1])) + float(np.mean(rb[:, 1]))) / 2
        eye_y = (float(np.mean(le[:, 1])) + float(np.mean(re[:, 1]))) / 2
        dist = eye_y - brow_y
        face_scale = buf[-1].get("face_scale", 50.0) if buf else 50.0
        face_scale = max(20.0, face_scale)
        dist_rel = dist / face_scale
        if dist_rel < 0.10:
            return 70.0 + min(20.0, (0.12 - dist_rel) * 200.0)
        if dist_rel < 0.14:
            return 55.0 + (0.14 - dist_rel) / 0.04 * 12.0
        return 46.0

    def _g2_gaze_shift_frequency(self, buf: list) -> float:
        """High shift rate = processing (Kahneman System 2)."""
        if len(buf) < 10:
            return 50.0
        yaws = [b.get("yaw", 0) for b in buf[-10:]]
        pitches = [b.get("pitch", 0) for b in buf[-10:]]
        d_yaw = np.abs(np.diff(yaws))
        d_pitch = np.abs(np.diff(pitches))
        shift_rate = float(np.mean(d_yaw) + np.mean(d_pitch))
        if shift_rate > 3.0:
            return 55.0 + min(35.0, shift_rate * 4.0)
        if shift_rate > 1.5:
            return 50.0 + (shift_rate - 1.5) * 5.0
        return 45.0

    def _g2_mouth_tight_eval(self, cur: dict, buf: list) -> float:
        """Tight mouth during evaluation."""
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        if len(buf) >= 3:
            mar = float(np.median([b.get("mar_inner", b.get("mar", mar)) for b in buf[-3:]]))
        baseline = max(0.08, self._baseline_mar) if self._baseline_mar > 0 else 0.18
        if mar < baseline * 0.75 and mar < 0.06:
            return 60.0 + min(25.0, (0.06 - mar) * 200.0)
        if mar < baseline * 0.85:
            return 50.0 + (0.85 - mar / baseline) * 30.0
        return 42.0

    # ----- Group 3 -----
    def _g3_contempt(
        self,
        lm: np.ndarray,
        ml: np.ndarray,
        mr: np.ndarray,
        fr: Optional[FaceDetectionResult],
        cur: dict,
        buf: list,
    ) -> float:
        """
        Contempt: unilateral lip curl (one corner raised). Research: Ekman & Friesen.
        CRITICAL: High false positive risk. Uses (1) baseline adaptation, (2) head-roll
        correction, (3) temporal consistency, (4) strict absolute thresholds.
        """
        # --- Azure path: require high confidence; scale gently ---
        if fr and getattr(fr, "emotions", None) and isinstance(fr.emotions, dict):
            c = float(fr.emotions.get("contempt", 0.0) or 0.0)
            # Only score above neutral when contempt > 0.55; scale gently to avoid persistent 100
            if c <= 0.45:
                return 35.0  # Below threshold = neutral/low
            if c <= 0.55:
                return 42.0 + (c - 0.45) * 80.0  # 42–50
            return min(82.0, 52.0 + (c - 0.55) * 80.0)  # 52–82 for c in [0.55, 0.92]

        # --- Landmark path: baseline + roll correction + temporal consistency ---
        if lm.shape[0] <= max(MOUTH_LEFT, MOUTH_RIGHT):
            return 35.0

        face_scale = cur.get("face_scale", 50.0)
        face_scale = max(20.0, face_scale)
        ly, ry = float(ml[1]), float(mr[1])
        raw_asymmetry_ratio = abs(ly - ry) / face_scale

        # Head-roll correction: tilt creates apparent asymmetry; subtract expected component
        roll = abs(float(cur.get("roll", 0)))
        roll_correction = 0.0
        if roll > 4.0:
            roll_correction = min(0.10, roll * 0.004)  # Up to 0.10 from ~25° roll
        corrected_ratio = max(0.0, raw_asymmetry_ratio - roll_correction)

        # Update history for baseline and temporal check
        self._contempt_asymmetry_history.append(corrected_ratio)

        # Baseline: median of last 18 frames (person's typical asymmetry when neutral)
        hist_list = list(self._contempt_asymmetry_history)
        contempt_warmup = int(self._params.get("contempt_warmup_frames", 6))
        if len(hist_list) < contempt_warmup:
            return 35.0  # Warmup: need enough history for stable baseline

        baseline = float(np.median(hist_list[-18:])) if len(hist_list) >= 18 else float(np.median(hist_list))

        # Require (a) absolute threshold, (b) above baseline, (c) temporal consistency
        abs_threshold = float(self._params.get("contempt_abs_threshold", 0.11))
        above_baseline_delta = float(self._params.get("contempt_above_baseline", 0.05))
        above_baseline = corrected_ratio > baseline + above_baseline_delta
        abs_met = corrected_ratio > abs_threshold

        temporal_min = int(self._params.get("contempt_temporal_min", 2))
        recent_ok = 0
        if len(buf) >= 4:
            for b in buf[-4:]:
                ar = b.get("mouth_corner_asymmetry_ratio", 0.0)
                roll_b = abs(float(b.get("roll", 0)))
                rc = min(0.10, roll_b * 0.004) if roll_b > 4.0 else 0.0
                cr = max(0.0, ar - rc)
                if cr > abs_threshold and cr > baseline + above_baseline_delta:
                    recent_ok += 1
        temporal_met = recent_ok >= temporal_min

        if not (abs_met and above_baseline and temporal_met):
            return 35.0

        # Score: how far above baseline; cap so we need pronounced contempt for high score
        excess = corrected_ratio - baseline - above_baseline_delta
        raw_score = 50.0 + min(32.0, excess * 250.0)  # 50–82 range
        return float(max(50.0, min(82.0, raw_score)))

    def _g3_nose_crinkle(self, cur: dict, buf: list) -> float:
        """
        Nose crinkle: nose shortening. Research: FACS AU 9; wrinkling nose = disgust,
        skepticism. Levator labii activates; nose shortens. CAUTION: Can also indicate
        concentration or reaction to strong stimulus. Require SUSTAINED shortening
        (3 of last 4 frames) and LARGE drop (median comparison) to reduce false positives.
        """
        nh = cur.get("nose_height", 0.0)
        if nh <= 0 or len(buf) < 6:
            return 8.0
        # Use median of last 6 nose_height for comparison (robust to jitter)
        recent = [b.get("nose_height", 0.0) for b in buf[-6:] if b.get("nose_height", 0) > 0]
        if not recent:
            return 8.0
        med = float(np.median(recent))
        if med <= 1e-6:
            return 8.0
        # Sustained shortening: current < median*0.88 in 3 of last 4 frames for score > 25
        short_count = sum(1 for b in buf[-4:] if (b.get("nose_height", 0) or 0) < med * 0.88)
        if nh < med * 0.88 and short_count >= 3:
            return 52.0 + min(36.0, (1.0 - nh / med) * 90.0)
        if nh < med * 0.88 and short_count >= 2:
            return 28.0 + (0.88 - nh / med) * 200.0
        if nh < med * 0.94:
            return 6.0 + (0.94 - nh / med) * 80.0  # Low score unless sustained
        return 6.0  # Default low

    def _g3_lip_compression(self, cur: dict, buf: list) -> float:
        """
        Lip compression: FACS AU 23/24. Research: pursed/compressed lips = disapproval,
        emotional restraint, withholding. "Preventing critical thoughts from being spoken."
        CAUTION: Can also indicate concentration, controlled speech, or processing in
        professional contexts.

        FALSE POSITIVE REDUCTION: (1) Compare to baseline MAR—only score high when
        current MAR is clearly BELOW person's typical mouth openness. (2) Use median
        of recent 4 frames to avoid single-frame/speech artifacts. (3) Require
        temporal consistency: at least 3 of last 4 frames below threshold. (4) Stricter
        absolute thresholds so neutral/resting lips rarely trigger.
        """
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        baseline_mar = max(0.08, self._baseline_mar) if self._baseline_mar > 0 else 0.20

        # WARMUP: Require enough history for stable baseline; otherwise return low
        buf_len = len(buf)
        if buf_len < 18:
            return 4.0
        if baseline_mar < 0.06:
            return 4.0

        # TEMPORAL: Use median of last 6 frames (longer window = fewer speech artifacts)
        if buf_len >= 6:
            recent_mar = [b.get("mar_inner", b.get("mar", mar)) for b in buf[-6:]]
            mar = float(np.median(recent_mar))
        mar_ratio = mar / baseline_mar if baseline_mar > 0 else 1.0

        below_threshold = float(self._params.get("lip_compression_below_threshold", 0.80))
        sustained_min = int(self._params.get("lip_compression_sustained_min", 4))
        sustained_all_min = int(self._params.get("lip_compression_sustained_all_min", 5))
        below_count = sum(
            1 for b in buf[-6:]
            if (b.get("mar_inner", b.get("mar", 0.2)) < baseline_mar * below_threshold)
        ) if buf_len >= 6 else 0
        sustained = below_count >= sustained_min
        sustained_all_6 = below_count >= sustained_all_min

        # Require mar_ratio < 0.72 for any score > 12 (avoids speech-artifact FPs)
        if mar_ratio >= 0.75:
            return 4.0
        if not sustained:
            if mar < 0.06 and mar_ratio < 0.72:
                return 8.0 + (0.75 - mar_ratio) * 20.0
            return 4.0

        # High band (score > 50): 5/6 sustained below baseline*0.80 (relaxed)
        if mar < 0.022 and mar_ratio < 0.68 and sustained_all_6:
            return 68.0 + min(18.0, (0.68 - mar_ratio) * 50.0)
        if mar < 0.032 and mar_ratio < 0.72 and sustained_all_6:  # sustained_all_6 = 5/6 now
            return 52.0 + (0.032 - mar) / 0.010 * 14.0
        if mar < 0.045 and mar_ratio < 0.72:
            return 38.0 + (0.045 - mar) / 0.013 * 10.0
        if mar < 0.058 and mar_ratio < 0.80:
            return 24.0 + (0.058 - mar) / 0.013 * 10.0
        if mar < 0.075 and mar_ratio < 0.85:
            return 12.0 + (0.075 - mar) / 0.017 * 8.0
        return 4.0

    def _g3_eye_block(self, buf: list) -> float:
        """
        Eye block: prolonged closure. Research: shutting out, aversion. Extended
        eye closure (EAR < 0.1) = blocking visual input, disengagement, or distress.
        CAUTION: Normal blinks last 1-3 frames. Only score high when PROLONGED
        (~400ms at 30fps, scaled for 30–60 fps) to avoid false positives from blinks.
        """
        if len(buf) < 6:
            return 6.0
        run = 0
        for b in reversed(buf):
            if b.get("ear", 0.2) < 0.10:
                run += 1
            else:
                break
        ref = self._FPS_REF
        t_600 = max(9, int(18 * self._fps / ref))
        t_400 = max(6, int(12 * self._fps / ref))
        t_14 = max(7, int(14 * self._fps / ref))
        t_8 = max(4, int(8 * self._fps / ref))
        if run >= t_600:
            return 88.0
        if run >= t_400:
            return 55.0 + (run - t_400) / max(1, (t_600 - t_400) / 2) * 28.0
        if run >= t_14:
            return 28.0 + (run - t_14) * 4.0
        if run >= t_8:
            return 12.0 + (run - t_8) * 2.0
        return 6.0  # Brief closure = likely blink

    def _g3_jaw_clench(self, lm: np.ndarray, cur: dict, fr: Optional[FaceDetectionResult], buf: list) -> float:
        """
        Jaw clench: tight jaw + mouth. Research: masseter tension = stress, resistance,
        suppressed aggression. Low MAR + corners down = clenched, tense.

        FALSE POSITIVE REDUCTION (minimal false detections): (1) Require BOTH tight
        lips AND pronounced corners-down (8% of face_scale). (2) Warmup 18+ frames;
        baseline comparison (mar_ratio < 0.72 for high score). (3) Sustained low MAR:
        5 of last 6 frames below baseline*0.72. (4) Compression alone (no corners down)
        capped at raw 36 so display never shows 100. Default 4.0.
        """
        mar = cur.get("mar_inner", cur.get("mar", 0.2))
        baseline_mar = max(0.08, self._baseline_mar) if self._baseline_mar > 0 else 0.20
        face_scale = cur.get("face_scale", 50.0)
        mouth_pts = _safe(lm, MOUTH)

        # WARMUP: Require enough history for stable baseline
        buf_len = len(buf)
        if buf_len < 18:
            return 4.0
        if baseline_mar < 0.06:
            return 4.0

        # TEMPORAL: Median of last 6 frames
        if buf_len >= 6:
            recent_mar = [b.get("mar_inner", b.get("mar", mar)) for b in buf[-6:]]
            mar = float(np.median(recent_mar))
        mar_ratio = mar / baseline_mar if baseline_mar > 0 else 1.0

        # Corners down: mouth corners noticeably BELOW lip midline (8% of face — very strict)
        corners_down = False
        if len(mouth_pts) >= 6:
            ly = float(mouth_pts[0, 1])
            ry = float(mouth_pts[5, 1])
            mid = float(np.mean(mouth_pts[:, 1]))
            threshold = face_scale * 0.08  # 8% — neutral mouths rarely qualify
            corners_down = (ly + ry) / 2 > mid + threshold

        # SUSTAINED: 5 of 6 for mid; 6/6 for highest band (68-82) for temporal consistency
        below_count = sum(
            1 for b in buf[-6:]
            if (b.get("mar_inner", b.get("mar", 0.2)) < baseline_mar * 0.72)
        ) if buf_len >= 6 else 0
        sustained_low_mar = below_count >= 5
        sustained_all_6 = below_count >= 6

        # Highest band (68-82): require 6/6 sustained_low_mar
        if mar < 0.038 and corners_down and mar_ratio < 0.72 and sustained_all_6:
            return min(82.0, 68.0 + (0.038 - mar) * 200.0)
        # Mid-high: 5 of 6 sustained
        if mar < 0.038 and corners_down and mar_ratio < 0.72 and sustained_low_mar:
            return min(67.0, 52.0 + (0.038 - mar) * 200.0)
        if mar < 0.048 and corners_down and mar_ratio < 0.78 and sustained_low_mar:
            return 52.0 + (0.048 - mar) / 0.010 * 18.0
        if mar < 0.058 and corners_down and mar_ratio < 0.82:
            return 38.0 + (0.058 - mar) / 0.010 * 12.0
        if mar < 0.070 and corners_down:
            return 24.0 + (0.070 - mar) / 0.012 * 10.0
        # Compression alone (no corners down): cap at raw 36 so display stays 0
        if mar < 0.038 and mar_ratio < 0.72:
            return 32.0 + (0.038 - mar) / 0.010 * 4.0
        if mar < 0.055 and mar_ratio < 0.80:
            return 18.0 + (0.055 - mar) / 0.017 * 10.0
        if mar < 0.075:
            return 8.0 + (0.075 - mar) / 0.020 * 6.0
        return 4.0

    def _g3_rapid_blink(self) -> float:
        """
        Rapid blinking: elevated blink rate. Research: blink rate increases with stress,
        cognitive load, anxiety. CAUTION: Normal blink rate varies (10-20/min); only
        score high when ELEVATED (5+ blinks in window). Context: rapid blinking can
        also indicate concentration or dry eyes. Use in combination with other G3 signals.
        """
        b = self._blinks_in_window
        # Score > 25 only when b >= 5 (avoid normal variation 4 in 2s as resistance); 6+ for 82+
        if b >= 6:
            return 82.0 + min(18.0, (b - 6) * 6.0)
        if b >= 5:
            return 48.0 + (b - 5) * 30.0  # 48-78 for b=5
        return 6.0 + min(b * 4.0, 18.0)  # 6, 10, 14, 18 for b=0,1,2,3; b=4 -> 18

    def _g3_gaze_aversion(self, cur: dict, buf: list) -> float:
        """
        Gaze aversion: looking away. Research: CRITICAL CONTEXT DEPENDENCY—gaze aversion
        serves DUAL functions: (1) internal cognitive processing/memory retrieval (positive),
        (2) disengagement/discomfort (negative). Gaze aversion duration: Glenberg et al. (1998),
        Doherty-Sneddon & Phelps (2005) on memory retrieval; sustained aversion in professional
        contexts signals disengagement (Mehrabian). Gaze-away during retrieval ~1s, lasting ~6s.
        
        Professional context: off-camera gaze is judged negatively despite cognitive function.
        To avoid false positives: only score high when SUSTAINED (>8 frames) AND combined
        with other resistance signals. Brief gaze shifts (<5 frames) are likely cognitive
        processing, NOT resistance.
        """
        if len(buf) < 5:
            return 8.0
        
        p = cur.get("pitch", 0)
        y = cur.get("yaw", 0)
        
        # Combine pitch and yaw for total gaze deviation
        pitch_dev = abs(p)
        yaw_dev = abs(y)
        total_dev = np.sqrt(pitch_dev * pitch_dev + yaw_dev * yaw_dev)
        
        n_recent = max(4, int(8 * self._fps / self._FPS_REF))
        recent_devs = []
        for b in buf[-n_recent:]:
            bp = abs(b.get("pitch", 0))
            by = abs(b.get("yaw", 0))
            recent_devs.append(np.sqrt(bp * bp + by * by))
        
        aversion_deg = float(self._params.get("gaze_aversion_deg", 8.0))
        sustained_aversion = 0
        for d in reversed(recent_devs):
            if d > aversion_deg:
                sustained_aversion += 1
            else:
                break
        
        ref = self._FPS_REF
        s_8 = max(4, int(8 * self._fps / ref))
        s_6 = max(3, int(6 * self._fps / ref))
        s_4 = max(2, int(4 * self._fps / ref))
        # FPS-scaled: params can override multiplier (e.g. 8, 6, 4)
        # Highest band (55+): require sustained_aversion >= s_8 (relaxed from s_10)
        if sustained_aversion >= s_8 and total_dev > 18.0:
            return 55.0 + min(40.0, (total_dev - 18.0) / 24.0 * 40.0)
        if sustained_aversion >= s_6 and total_dev > 18.0:
            return 48.0 + min(10.0, (sustained_aversion - s_6) * 4.0)
        if sustained_aversion >= s_4 and total_dev > 14.0:
            return 38.0 + min(22.0, sustained_aversion * 2.5)
        if sustained_aversion >= 2 and total_dev > 20.0:  # Relaxed from s_4
            return 28.0 + min(18.0, (total_dev - 20.0) / 15.0 * 18.0)
        if total_dev > 25.0:
            return 18.0 + min(12.0, (total_dev - 25.0) / 20.0 * 12.0)
        # Smooth floor: slight deviation still raises score a little (6–16)
        return 6.0 + min(10.0, total_dev * 0.35)

    def _g3_no_nod(self, buf: list) -> float:
        """
        No-nod: absence of vertical head movement. Research: nodding = agreement;
        absence of nodding = disengagement, resistance, or passive listening.
        
        FALSE POSITIVE REDUCTION: time-based window (scaled 30–60 fps); stricter thresholds for score >= 65;
        cap max at 58 so "no nod" does not dominate G3_raw unless combined with other cues.
        """
        win = max(12, int(24 * self._fps / self._FPS_REF))
        if len(buf) < win:
            return 12.0  # Warmup: longer window for sustained stillness
        pitches = [b["pitch"] for b in buf[-win:]]
        pitch_std = float(np.std(pitches))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        d = np.diff(pitches)
        zero_crossings = int(np.sum((d[:-1] * d[1:]) < 0))
        
        # Score >= 65 only with pitch_std < 0.4 and pitch_range < 1.5; cap max at 58
        if pitch_std < 0.4 and pitch_range < 1.5 and zero_crossings < 1:
            return min(58.0, 65.0 + (0.4 - pitch_std) * 25.0)
        if pitch_std < 0.5 and pitch_range < 1.8 and zero_crossings < 1:
            return 48.0 + (0.5 - pitch_std) * 20.0
        if pitch_std < 0.8 and pitch_range < 2.5 and zero_crossings < 2:
            return 38.0 + (0.8 - pitch_std) / 0.3 * 12.0
        if pitch_std < 1.0 and zero_crossings < 1:
            return 28.0 + (1.0 - pitch_std) * 10.0
        return 12.0 + max(0.0, (2.0 - pitch_std)) / 2.0 * 20.0

    def _g3_narrowed_pupils(self, cur: dict, buf: list) -> float:
        """
        Narrowed pupils proxy via eye squint (EAR). Research: pupil constriction
        correlates with negative arousal; we proxy via narrowed eyes (lower EAR).
        
        FALSE POSITIVE REDUCTION: Require 3 of last 4 frames with ear_ratio < 0.85 before
        score > 25; strong band ear_ratio < 0.68. Reduces overlap with G2 eye_squint.
        """
        ear = cur.get("ear", 0.2)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0.05 else 0.22
        
        # TEMPORAL: Use median of recent EAR (exclude blinks)
        if len(buf) >= 3:
            recent_ear = [b.get("ear", ear) for b in buf[-3:] if b.get("is_blink", 0) < 0.5]
            if recent_ear:
                ear = float(np.median(recent_ear))
        
        ear_ratio = ear / baseline_ear if baseline_ear > 0 else 1.0
        
        ratio_threshold = float(self._params.get("narrowed_pupils_ratio_threshold", 0.88))
        sustained_min = int(self._params.get("narrowed_pupils_sustained_min", 2))
        if len(buf) >= 4:
            recent_ratios = []
            for b in buf[-4:]:
                eb = b.get("ear", 0.2)
                if b.get("is_blink", 0) < 0.5 and baseline_ear > 0:
                    recent_ratios.append(eb / baseline_ear)
            sustained = sum(1 for r in recent_ratios if r < ratio_threshold) >= sustained_min if recent_ratios else False
        else:
            sustained = ear_ratio < ratio_threshold
        
        if not sustained and ear_ratio >= ratio_threshold:
            return 15.0
        if not sustained:
            return min(25.0, 15.0 + (ratio_threshold - ear_ratio) * 40.0)  # Cap at 25 until sustained
        
        # Strong band: ear_ratio < 0.68 (was 0.70)
        if ear_ratio < 0.68 and ear < 0.13:
            return 55.0 + min(35.0, (0.68 - ear_ratio) * 140.0)
        if ear_ratio < 0.82 and ear < 0.16:
            return 38.0 + (0.82 - ear_ratio) * 120.0
        if ear_ratio < 0.90:
            return 22.0 + (0.90 - ear_ratio) * 150.0
        return 15.0

    def _g3_lip_corner_dip(self, lm: np.ndarray, cur: dict, buf: list) -> float:
        """Sadness/disappointment leakage (FACS AU15). Mouth corners down."""
        mouth_pts = _safe(lm, MOUTH)
        if len(mouth_pts) < 6:
            return 35.0
        face_scale = max(1e-6, cur.get("face_scale", 50.0))
        ly, ry = float(mouth_pts[0, 1]), float(mouth_pts[5, 1]) if len(mouth_pts) > 5 else float(mouth_pts[0, 1])
        mid_y = float(np.mean(mouth_pts[:, 1]))
        corners_down = (ly + ry) / 2 > mid_y + face_scale * 0.03
        if corners_down:
            dip_amt = ((ly + ry) / 2 - mid_y) / face_scale
            return 50.0 + min(40.0, dip_amt * 300.0)
        return 35.0

    def _g3_brow_lower_sustained(self, buf: list) -> float:
        """Sustained frown = negative affect."""
        if len(buf) < 8:
            return 35.0
        heights = [(b.get("eyebrow_l", 0) + b.get("eyebrow_r", 0)) / 2 for b in buf[-8:]]
        face_scale = buf[-1].get("face_scale", 50.0)
        baseline = float(np.median(heights[:4])) if len(heights) >= 4 else heights[0]
        cur_avg = float(np.mean(heights[-4:])) if len(heights) >= 4 else heights[-1]
        lower_amt = (baseline - cur_avg) / max(face_scale * 0.06, 1e-6)
        if lower_amt > 0.4:
            return 52.0 + min(35.0, lower_amt * 40.0)
        return 38.0

    def _g3_eye_squeeze(self, cur: dict, buf: list) -> float:
        """Tight lid closure = distress (FACS AU7). Stronger than eye_block."""
        ear = cur.get("ear", 0.2)
        baseline_ear = self._baseline_ear if self._baseline_ear > 0.05 else 0.22
        if ear < 0.08 and (len(buf) < 4 or sum(1 for b in buf[-4:] if b.get("ear", 0.2) < 0.10) >= 2):
            return 65.0 + min(25.0, (0.08 - ear) * 200.0)
        if ear < 0.12:
            return 48.0 + (0.12 - ear) * 100.0
        return 38.0

    def _g3_head_shake(self, buf: list) -> float:
        """Lateral head movement = disagreement."""
        if len(buf) < 10:
            return 35.0
        yaws = [b.get("yaw", 0) for b in buf[-10:]]
        d = np.diff(yaws)
        crosses = int(np.sum((d[:-1] * d[1:]) < 0))
        yaw_std = float(np.std(yaws))
        if crosses >= 2 and yaw_std > 2.0:
            return 55.0 + min(30.0, crosses * 8.0 + yaw_std * 2.0)
        return 38.0

    # ----- Group 4 -----
    def _g4_relaxed_exhale(self, buf: list) -> float:
        """
        Relaxed exhale: release of tension. Research: tension release = movement drop
        (stillness after release) + mouth opening (MAR). Uses temporal movement instead
        of spatial face_var (which increases when mouth opens). Fallback: nose_std drop
        or no variance increase when MAR increased.
        """
        if len(buf) < 8:
            return 48.0  # Warmup
        window_before = buf[-8:-4]
        window_now = buf[-3:]
        scales_now = [b.get("face_scale", 50.0) for b in window_now]
        scales_before = [b.get("face_scale", 50.0) for b in window_before]
        avg_scale_now = float(np.mean(scales_now)) if scales_now else 50.0
        avg_scale_before = float(np.mean(scales_before)) if scales_before else 50.0
        var_now_raw = np.mean([b["face_var"] for b in window_now])
        var_before_raw = np.mean([b["face_var"] for b in window_before])
        var_now = var_now_raw / max(avg_scale_now * avg_scale_now, 1e-6)
        var_before = var_before_raw / max(avg_scale_before * avg_scale_before, 1e-6)
        mar_now = float(np.mean([b["mar"] for b in window_now]))
        mar_before = float(np.mean([b["mar"] for b in window_before]))
        recent_mars = [b["mar"] for b in buf[-12:]]
        min_recent_mar = float(np.min(recent_mars)) if recent_mars else mar_before
        baseline_mar = max(self._baseline_mar, 0.04)
        nose_std_now = float(np.mean([b.get("nose_std", 0.0) for b in window_now]))
        nose_std_before = float(np.mean([b.get("nose_std", 0.0) for b in window_before]))

        # 1) Tension drop: prefer movement (stillness after release); fallback when movement missing
        movement_now = float(np.mean([b.get("face_movement", 0.0) for b in window_now]))
        movement_before = float(np.mean([b.get("face_movement", 0.0) for b in window_before]))
        movement_available = movement_before > 1e-9
        if movement_available:
            tension_drop = movement_now < movement_before * 0.78
            strong_tension_drop = movement_before > 0 and movement_now < movement_before * 0.60
        else:
            # Fallback: nose_std drop (>15%) or no variance increase when MAR increased
            mar_increased = mar_before > 0.04 and mar_now > mar_before * 1.03
            nose_drop = nose_std_before > 1e-9 and nose_std_now < nose_std_before * 0.85
            no_var_spike = var_before > 0 and var_now <= var_before * 1.05
            tension_drop = nose_drop or (mar_increased and no_var_spike)
            strong_tension_drop = nose_drop

        # 2) Mouth opening: MAR increase (relaxed 1.03) or above baseline / above recent min
        mouth_opening = (
            (mar_before > 0.04 and mar_now > mar_before * 1.03)
            or (baseline_mar > 0.04 and mar_now > baseline_mar * 1.05)
            or (min_recent_mar > 0.04 and mar_now > min_recent_mar * 1.05)
        )

        # Scoring: both -> 70+; strong tension drop alone -> 58+; moderate tension / mouth with mild tension -> 55-62
        if tension_drop and mouth_opening:
            return 70.0 + min(18.0, (mar_now / max(mar_before, 0.04) - 1.03) * 90.0)
        if strong_tension_drop and mouth_opening:
            return 65.0
        if strong_tension_drop:
            return 58.0 + min(12.0, (1.0 - movement_now / movement_before) * 25.0) if movement_available else 58.0
        if tension_drop:
            return 48.0 + min(10.0, (1.0 - movement_now / movement_before) * 20.0) if movement_available else 52.0
        # Mouth opening with mild tension (no big variance spike) can reach 55-62
        if mouth_opening and var_before > 0 and var_now <= var_before * 1.08:
            return 55.0 + min(7.0, (mar_now / max(baseline_mar, 0.04) - 1.0) * 35.0)
        # Baseline: smooth increase with MAR so small mouth opening still raises score slightly
        return 42.0 + min(8.0, mar_now * 40.0)

    def _g4_fixed_gaze(self, buf: list, w: int, h: int) -> float:
        """
        Fixed gaze: stable head orientation = looking at a fixed region (camera or elsewhere).
        Uses yaw/pitch variance (head rotation); low variance = fixated, high = looking around.
        
        FALSE POSITIVE REDUCTION: Use longer window for stability check and require
        sustained fixation (not just momentary stillness). Distinguish genuine focus
        from brief pauses.
        """
        if len(buf) < 12:
            return 50.0  # Warmup
        # Use head orientation (yaw, pitch) with longer window for stability
        window = buf[-14:]
        yaws = [float(b.get("yaw", 0)) for b in window]
        pitches = [float(b.get("pitch", 0)) for b in window]
        std_yaw = float(np.std(yaws))
        std_pitch = float(np.std(pitches))
        head_std = np.sqrt(std_yaw * std_yaw + std_pitch * std_pitch)
        
        # Also check range (not just std) to catch steady drift
        yaw_range = float(np.max(yaws) - np.min(yaws))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        head_range = np.sqrt(yaw_range * yaw_range + pitch_range * pitch_range)
        
        # Continuous: low head_std/range = high (fixed gaze); smooth decay as movement increases
        if head_std < 1.2 and head_range < 3.5:
            return 82.0
        if head_std < 2.5 and head_range < 6.0:
            return 68.0 + (2.5 - head_std) / 1.3 * 10.0
        if head_std < 4.0 and head_range < 10.0:
            return 55.0 + (4.0 - head_std) / 1.5 * 10.0
        if head_std < 6.0:
            return 45.0 + (6.0 - head_std) / 2.0 * 8.0
        return 42.0 + max(0.0, (8.0 - head_std)) / 8.0 * 5.0  # Smooth floor 42-47

    def _g4_smile_transition(self, buf: list, duchenne: float) -> float:
        """
        Smile transition: sustained genuine smile. Research: sustained Duchenne
        + stable mouth (MAR) = authentic positive affect, decision-ready.
        
        FALSE POSITIVE REDUCTION: Require sustained Duchenne score (not just current
        frame) and stable mouth pattern to distinguish genuine smiles from brief
        expressions or noise.
        """
        if len(buf) < 12:
            return 45.0
        mars = [b["mar"] for b in buf[-12:]]
        mar_mean = float(np.mean(mars[-6:]))
        mar_std = float(np.std(mars[-6:]))
        sustained = mar_mean > 0.14 and mar_std < 0.04
        # Continuous in duchenne: base 46 at 52, ramp up with duchenne and sustained
        base = 46.0 + max(0.0, duchenne - 52.0) * 0.5  # 46-53 for duchenne 52-58
        if sustained:
            base = base + 12.0 + max(0.0, duchenne - 56.0) * 0.8  # sustained adds 12+ and duchenne bonus
        if duchenne >= 62 and sustained:
            base = 78.0 + min(12.0, (duchenne - 62) * 0.6)
        elif duchenne >= 58:
            base = max(base, 55.0 + min(8.0, (duchenne - 58) * 0.5))
        return float(max(45.0, min(100.0, base)))

    def _g4_mouth_relax(self, buf: list) -> float:
        """Jaw/mouth relaxation = release (Cialdini)."""
        if len(buf) < 8:
            return 48.0
        mar_now = float(np.mean([b["mar"] for b in buf[-3:]]))
        mar_before = float(np.mean([b["mar"] for b in buf[-8:-4]]))
        baseline = max(0.06, self._baseline_mar)
        if mar_before > 0.03 and mar_now > mar_before * 1.08 and mar_now > baseline * 1.05:
            return 60.0 + min(30.0, (mar_now / mar_before - 1.0) * 150.0)
        if mar_now > baseline * 1.05:
            return 52.0 + min(10.0, (mar_now / baseline - 1.0) * 50.0)
        return 45.0

    def _g4_smile_sustain(self, buf: list, duchenne: float) -> float:
        """Smile duration = authentic positive affect. Sustained Duchenne/MAR over frames."""
        if len(buf) < 12:
            return 45.0
        mar_mean = float(np.mean([b["mar"] for b in buf[-6:]]))
        mar_std = float(np.std([b["mar"] for b in buf[-6:]]))
        sustained_open = mar_mean > 0.14 and mar_std < 0.04
        if duchenne >= 60 and sustained_open:
            return 72.0 + min(18.0, (duchenne - 60) * 0.6)
        if duchenne >= 56 and sustained_open:
            return 62.0 + min(12.0, (duchenne - 56) * 2.0)
        if duchenne >= 52:
            return 50.0 + (duchenne - 52) * 1.5
        return 45.0


# -----------------------------------------------------------------------------
# Signifier weights (in-code defaults; used by engine and engagement detector)
# -----------------------------------------------------------------------------
DEFAULT_SIGNIFIER_WEIGHTS: List[float] = [
    1.05, 0.55, 1.00, 1.55, 1.20, 1.25, 1.00, 1.35, 1.10, 1.00,
    1.05, 1.00, 1.00, 1.00, 1.10, 1.00, 1.00, 1.15, 1.30, 0.50,
    1.10, 1.00, 1.15, 1.00, 1.05, 1.40, 1.10, 1.50, 1.00, 1.25,
    1.00, 1.55, 1.05, 0.85, 0.50, 1.10, 1.10, 1.00, 1.00,
    1.40, 1.50, 1.45, 1.15, 1.20,
]
DEFAULT_GROUP_WEIGHTS: List[float] = [0.30, 0.20, 0.30, 0.20]
DEFAULT_FUSION_AZURE: float = 0.5
DEFAULT_FUSION_MEDIAPIPE: float = 0.5

_signifier_current: Dict[str, List[float]] = {
    "signifier": list(DEFAULT_SIGNIFIER_WEIGHTS),
    "group": list(DEFAULT_GROUP_WEIGHTS),
}
_signifier_fusion: Dict[str, float] = {"azure": DEFAULT_FUSION_AZURE, "mediapipe": DEFAULT_FUSION_MEDIAPIPE}


def get_weights() -> Dict[str, List[float]]:
    """Return current signifier and group weights. Safe to modify the returned dict."""
    return {"signifier": list(_signifier_current["signifier"]), "group": list(_signifier_current["group"])}


def get_fusion_weights() -> Tuple[float, float]:
    """Return (azure_weight, mediapipe_weight) for unified score fusion. Sum to 1.0."""
    return (_signifier_fusion["azure"], _signifier_fusion["mediapipe"])


def load_weights() -> Dict[str, List[float]]:
    """Set signifier and fusion weights to in-code defaults (no file or backend)."""
    _signifier_current["signifier"] = list(DEFAULT_SIGNIFIER_WEIGHTS)
    _signifier_current["group"] = list(DEFAULT_GROUP_WEIGHTS)
    _signifier_fusion["azure"] = getattr(config, "FUSION_AZURE_WEIGHT", DEFAULT_FUSION_AZURE)
    _signifier_fusion["mediapipe"] = getattr(config, "FUSION_MEDIAPIPE_WEIGHT", DEFAULT_FUSION_MEDIAPIPE)
    total = _signifier_fusion["azure"] + _signifier_fusion["mediapipe"]
    if total > 0:
        _signifier_fusion["azure"] /= total
        _signifier_fusion["mediapipe"] /= total
    return get_weights()


def build_weights_provider() -> Callable[[], Dict[str, List[float]]]:
    """Return a callable that returns the current weights (for ExpressionSignifierEngine)."""
    return get_weights

# ============== 6. Engagement composites ==============
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Speech tag: {"category": str, "phrase": str, "time": float, "discourse_boost": bool?}
SpeechTag = Dict[str, Any]


def _signifier(scores: Dict[str, float], key: str) -> float:
    """Safe lookup; returns 0 if missing."""
    return float(scores.get(key, 0.0))


def _condition_met(
    scores: Dict[str, float],
    conditions: List[Tuple[str, Optional[float], Optional[float]]],
) -> Tuple[bool, float]:
    """
    Check if conditions are met. Each condition: (key, min_inclusive, max_inclusive).
    Use (key, min, None) for lower-bound only; (key, None, max) for upper-bound only.
    Returns (all_met, fulfillment_ratio 0-1). Fulfillment = fraction of conditions met.
    """
    if not scores or not conditions:
        return (False, 0.0)
    met_list = []
    for key, lo, hi in conditions:
        v = _signifier(scores, key)
        if lo is not None and hi is not None:
            met_list.append(1.0 if lo <= v <= hi else 0.0)
        elif hi is None:
            met_list.append(1.0 if v >= lo else 0.0)
        else:
            met_list.append(1.0 if v <= hi else 0.0)
    fulfillment = float(np.mean(met_list)) if met_list else 0.0
    all_met = fulfillment >= 0.99  # All conditions met (allow float tolerance)
    return (all_met, fulfillment)


def _condition_composite(
    scores: Dict[str, float],
    conditions: List[Tuple[str, Optional[float], Optional[float]]],
    base_when_met: float = 75.0,
    base_when_not: float = 20.0,
) -> float:
    """
    Score 0-100: high when all conditions met, low when not. Partial fulfillment
    scales between base_when_not and base_when_met.
    """
    _, fulfillment = _condition_met(scores, conditions)
    return base_when_not + fulfillment * (base_when_met - base_when_not)


def _has_category(tags: List[SpeechTag], categories: List[str]) -> bool:
    if not tags or not categories:
        return False
    cats = set(categories)
    return any((t.get("category") or "") in cats for t in tags)


def _category_strength(tags: List[SpeechTag], categories: List[str], window_sec: float = 12.0) -> float:
    """Return 0-1 strength from recency and count of matching tags in window. discourse_boost adds +0.1."""
    if not tags or not categories:
        return 0.0
    now = time.time()
    cutoff = now - window_sec
    cats = set(categories)
    matches = [t for t in tags if t.get("time", 0) >= cutoff and (t.get("category") or "") in cats]
    if not matches:
        return 0.0
    # More recent + more matches = higher strength
    recency = sum(1.0 - (now - t.get("time", 0)) / window_sec for t in matches) / max(len(matches), 1)
    count_norm = min(1.0, len(matches) / 3.0)
    base = min(1.0, 0.5 * recency + 0.5 * count_norm)
    # discourse_boost: align discourse marker with category -> +0.1 strength
    boost = 0.1 if any(t.get("discourse_boost") for t in matches) else 0.0
    return min(1.0, base + boost)


def compute_composite_metrics(
    group_means: Dict[str, float],
    signifier_scores: Optional[Dict[str, float]] = None,
    speech_tags: Optional[List[SpeechTag]] = None,
    composite_weights: Optional[Dict[str, Dict[str, float]]] = None,
    acoustic_tags: Optional[List[str]] = None,
    acoustic_negative_strength: float = 0.0,
) -> Dict[str, float]:
    """
    Compute 0-100 composite metrics from facial (G1-G4), speech tags, and optional acoustic.

    acoustic_tags: e.g. from get_recent_acoustic_tags(); used to boost confusion, tension, disengagement.
    acoustic_negative_strength: 0-1 from get_acoustic_negative_strength() (optional).

    Returns dict with keys including verbal_nonverbal_alignment, cognitive_load_multimodal,
    rapport_engagement, skepticism_objection_strength, decision_readiness_multimodal,
    opportunity_strength, trust_rapport, disengagement_risk_multimodal,
    confusion_multimodal, tension_objection_multimodal, loss_of_interest_multimodal,
    decision_plus_voice.
    """
    out: Dict[str, float] = {}
    g1 = group_means.get("g1", 50.0)
    g2 = group_means.get("g2", 50.0)
    g3 = group_means.get("g3", 50.0)  # high = low resistance
    g4 = group_means.get("g4", 50.0)
    tags = speech_tags or []
    cw = composite_weights or {}
    ac_tags = set(acoustic_tags or [])

    # Speech strengths (0-1)
    commit_interest = _category_strength(tags, ["commitment", "interest"])
    confusion_concern = _category_strength(tags, ["confusion", "concern"])
    objection_concern = _category_strength(tags, ["objection", "concern"])
    interest_realization = _category_strength(tags, ["interest", "realization"])
    interest_confirmation = _category_strength(tags, ["interest", "confirmation"])
    timeline_budget = _category_strength(tags, ["timeline", "budget"])
    skepticism_speech = _category_strength(tags, ["skepticism"])
    enthusiasm_speech = _category_strength(tags, ["enthusiasm"])
    hesitation_speech = _category_strength(tags, ["hesitation"])
    authority_speech = _category_strength(tags, ["authority"])

    # Acoustic boosts (0-1) for blending into composites
    has_uncertainty = 0.25 if "acoustic_uncertainty" in ac_tags else 0.0
    has_tension = 0.25 if "acoustic_tension" in ac_tags else 0.0
    has_disengagement = 0.25 if "acoustic_disengagement_risk" in ac_tags else 0.0
    has_roughness = 0.2 if "acoustic_roughness_proxy" in ac_tags else 0.0
    has_falling = 0.15 if "acoustic_falling_contour" in ac_tags else 0.0
    has_arousal_high = 0.15 if "acoustic_arousal_high" in ac_tags else 0.0
    has_monotone = 0.2 if "acoustic_monotone" in ac_tags else 0.0

    # Condition-based composites (core feature combinations → higher-level mental states)
    sig = signifier_scores or {}

    out["topic_interest_facial"] = float(np.clip(_condition_composite(sig, [("g1_forward_lean", 55.0, 90.0), ("g1_eye_contact", 70.0, None)], 80.0, 25.0), 0.0, 100.0))
    out["active_listening_facial"] = float(np.clip(_condition_composite(sig, [("g1_rhythmic_nodding", 45.0, None), ("g1_parted_lips", 45.0, None), ("g1_eye_contact", 60.0, None)], 78.0, 20.0), 0.0, 100.0))
    out["agreement_signals_facial"] = float(np.clip(_condition_composite(sig, [("g1_rhythmic_nodding", 50.0, None), ("g1_duchenne", 55.0, None), ("g1_eye_contact", 65.0, None)], 82.0, 22.0), 0.0, 100.0))
    out["closing_window_facial"] = float(np.clip(_condition_composite(sig, [("g4_smile_transition", 58.0, None), ("g4_fixed_gaze", 60.0, None), ("g4_mouth_relax", 52.0, None)], 85.0, 25.0), 0.0, 100.0))
    out["receptivity_facial"] = float(np.clip(_condition_composite(sig, [("g1_parted_lips", 48.0, None), ("g1_softened_forehead", 55.0, None), ("g1_duchenne", 52.0, None)], 76.0, 22.0), 0.0, 100.0))
    _topic_interest = out["topic_interest_facial"]
    _active_listening = out["active_listening_facial"]
    _closing_window = out["closing_window_facial"]
    _receptivity = out["receptivity_facial"]

    # Verbal-nonverbal alignment: words + face agree (commitment/interest + G4/G1)
    # Refined: +8 when topic_interest (lean + eye contact) supports nonverbal engagement
    wa = cw.get("verbal_nonverbal_alignment", {})
    w_align_speech = wa.get("speech", 0.6)
    w_align_face = wa.get("face", 0.4)
    face_positive = (g4 * 0.5 + g1 * 0.5) / 100.0
    align_raw = (commit_interest * w_align_speech + face_positive * w_align_face) * 100.0
    if _topic_interest >= 65.0:
        align_raw = min(100.0, align_raw + 8.0)  # Lean + eye contact reinforces alignment
    out["verbal_nonverbal_alignment"] = float(np.clip(align_raw, 0.0, 100.0))

    # Cognitive load (multimodal): G2 + confusion/concern speech + acoustic_uncertainty
    w_load = cw.get("cognitive_load_multimodal", {})
    w_g2 = w_load.get("g2", 0.55)
    w_conf = w_load.get("speech", 0.45)
    g2_norm = g2 / 100.0
    load_raw = (g2_norm * w_g2 + confusion_concern * w_conf) * 100.0
    load_raw = min(100.0, load_raw + has_uncertainty * 100.0)
    out["cognitive_load_multimodal"] = float(np.clip(load_raw, 0.0, 100.0))

    # Rapport: G1 + interest/realization + low resistance (G3 high = good)
    w_rapport = cw.get("rapport_engagement", {})
    w_g1_r = w_rapport.get("g1", 0.4)
    w_ir = w_rapport.get("speech", 0.35)
    w_g3_r = w_rapport.get("g3", 0.25)
    g3_norm = g3 / 100.0
    rapport_raw = (g1 / 100.0 * w_g1_r + interest_realization * w_ir + g3_norm * w_g3_r) * 100.0
    if _active_listening >= 60.0:
        rapport_raw = min(100.0, rapport_raw + 6.0)  # Nodding + parted lips + eye contact = rapport
    out["rapport_engagement"] = float(np.clip(rapport_raw, 0.0, 100.0))

    # Skepticism/objection: objection/concern speech + resistance + acoustic_tension + acoustic_roughness
    w_sk = cw.get("skepticism_objection_strength", {})
    w_obj = w_sk.get("speech", 0.6)
    w_res = w_sk.get("resistance", 0.4)
    g3_resistance = 100.0 - g3  # high = more resistance
    skept_raw = (objection_concern * w_obj + (g3_resistance / 100.0) * w_res) * 100.0
    skept_raw = min(100.0, skept_raw + (has_tension + has_roughness) * 100.0)
    out["skepticism_objection_strength"] = float(np.clip(skept_raw, 0.0, 100.0))

    # Decision readiness (multimodal): G4 + commitment/interest speech
    w_ready = cw.get("decision_readiness_multimodal", {})
    w_g4 = w_ready.get("g4", 0.55)
    w_ci = w_ready.get("speech", 0.45)
    ready_raw = (g4 / 100.0 * w_g4 + commit_interest * w_ci) * 100.0
    if _closing_window >= 70.0:
        ready_raw = min(100.0, ready_raw + 10.0)  # Smile + fixed gaze + mouth relax = closing window
    out["decision_readiness_multimodal"] = float(np.clip(ready_raw, 0.0, 100.0))

    # Opportunity strength: for closing moments; combines decision readiness + verbal-nonverbal alignment
    dr = out["decision_readiness_multimodal"] / 100.0
    vn = out["verbal_nonverbal_alignment"] / 100.0
    w_opp = cw.get("opportunity_strength", {})
    w_dr = w_opp.get("decision_readiness", 0.55)
    w_vn = w_opp.get("verbal_nonverbal", 0.45)
    opp_raw = (dr * w_dr + vn * w_vn) * 100.0
    out["opportunity_strength"] = float(np.clip(opp_raw, 0.0, 100.0))

    # Trust/rapport: G1 (interest face) + interest/realization speech + low resistance (G3 high)
    w_tr = cw.get("trust_rapport", {})
    w_g1_tr = w_tr.get("g1", 0.5)
    w_ir_tr = w_tr.get("speech", 0.3)
    w_g3_tr = w_tr.get("g3", 0.2)
    trust_raw = (g1 / 100.0 * w_g1_tr + interest_realization * w_ir_tr + g3_norm * w_g3_tr) * 100.0
    if _receptivity >= 65.0:
        trust_raw = min(100.0, trust_raw + 7.0)  # Open lips + relaxed brow + smile = receptivity
    out["trust_rapport"] = float(np.clip(trust_raw, 0.0, 100.0))

    # Disengagement risk (multimodal): low G1 + no positive speech + resistance + acoustic_disengagement_risk
    g3_res = (100.0 - g3) / 100.0
    no_commit = 1.0 - commit_interest
    w_dis = cw.get("disengagement_risk_multimodal", {})
    w_g1_dis = w_dis.get("g1_low", 0.35)
    w_nocommit = w_dis.get("no_commit", 0.35)
    w_res_dis = w_dis.get("resistance", 0.30)
    dis_raw = ((100.0 - g1) / 100.0 * w_g1_dis + no_commit * w_nocommit + g3_res * w_res_dis) * 100.0
    dis_raw = min(100.0, dis_raw + has_disengagement * 100.0)
    out["disengagement_risk_multimodal"] = float(np.clip(dis_raw, 0.0, 100.0))

    # Confusion (multimodal): G2 + confusion/concern speech + acoustic_uncertainty (Kahneman + vocal affect)
    conf_raw = (g2 / 100.0 * 0.45 + confusion_concern * 0.40) * 100.0 + has_uncertainty * 25.0
    out["confusion_multimodal"] = float(np.clip(conf_raw, 0.0, 100.0))

    # Tension/objection (multimodal): resistance + objection/concern + acoustic_tension + roughness (Ekman, Cialdini, Gobl)
    g3_res_norm = (100.0 - g3) / 100.0
    tension_raw = (g3_res_norm * 0.45 + objection_concern * 0.40) * 100.0 + (has_tension + has_roughness) * 25.0
    out["tension_objection_multimodal"] = float(np.clip(tension_raw, 0.0, 100.0))

    # Loss of interest (multimodal): low G1 + no commitment language + acoustic withdrawal (Mehrabian, Driskell)
    loss_g1 = (100.0 - g1) / 100.0
    loss_raw = (loss_g1 * 0.45 + no_commit * 0.40) * 100.0 + has_disengagement * 25.0 + acoustic_negative_strength * 15.0
    out["loss_of_interest_multimodal"] = float(np.clip(loss_raw, 0.0, 100.0))

    # Decision plus voice: decision_readiness + acoustic closure/arousal (Cialdini + vocal readiness)
    dr_val = out["decision_readiness_multimodal"] / 100.0
    voice_boost = min(0.2, has_falling + has_arousal_high)
    out["decision_plus_voice"] = float(np.clip((dr_val + voice_boost) * 100.0, 0.0, 100.0))

    # --- New composites (Part 5 of plan) ---

    # Psychological safety proxy (Edmondson): Low G3 + interest/confirmation + low tension
    low_tension = 1.0 - (has_tension + has_roughness)
    safety_raw = (g3 / 100.0 * 0.45 + interest_confirmation * 0.35 + low_tension * 0.2) * 100.0
    out["psychological_safety_proxy"] = float(np.clip(safety_raw, 0.0, 100.0))

    # Urgency sensitivity: timeline/budget speech + arousal + G4
    urgency_raw = (timeline_budget * 0.4 + (has_arousal_high * 4.0) * 0.3 + g4 / 100.0 * 0.3) * 100.0
    out["urgency_sensitivity"] = float(np.clip(urgency_raw, 0.0, 100.0))

    # Skepticism strength: skepticism speech + resistance + tension/roughness
    skept2_raw = (skepticism_speech * 0.5 + g3_resistance / 100.0 * 0.3) * 100.0
    skept2_raw = min(100.0, skept2_raw + (has_tension + has_roughness) * 50.0)
    out["skepticism_strength"] = float(np.clip(skept2_raw, 0.0, 100.0))

    # Enthusiasm multimodal: enthusiasm speech + G1 + acoustic_arousal_high
    enth_raw = (enthusiasm_speech * 0.4 + g1 / 100.0 * 0.4 + has_arousal_high * 2.0 * 0.2) * 100.0
    out["enthusiasm_multimodal"] = float(np.clip(enth_raw, 0.0, 100.0))

    # Hesitation multimodal: hesitation speech + G2 + acoustic_uncertainty
    hes_raw = (hesitation_speech * 0.4 + g2 / 100.0 * 0.4 + has_uncertainty * 2.0 * 0.2) * 100.0
    out["hesitation_multimodal"] = float(np.clip(hes_raw, 0.0, 100.0))

    # Authority deferral: authority speech + G2 + acoustic_monotone
    auth_raw = (authority_speech * 0.5 + g2 / 100.0 * 0.3 + has_monotone * 2.0 * 0.2) * 100.0
    out["authority_deferral"] = float(np.clip(auth_raw, 0.0, 100.0))

    # Rapport depth: G1 + interest/confirmation + acoustic_falling_contour
    rap_raw = (g1 / 100.0 * 0.45 + interest_confirmation * 0.35 + has_falling * 2.0 * 0.2) * 100.0
    out["rapport_depth"] = float(np.clip(rap_raw, 0.0, 100.0))

    # Cognitive overload proxy: G2 + confusion speech + acoustic_uncertainty + stillness
    still_norm = (signifier_scores or {}).get("g2_stillness", 50.0) / 100.0
    overload_raw = (g2 / 100.0 * 0.35 + confusion_concern * 0.35 + has_uncertainty * 2.0 * 0.15 + still_norm * 0.15) * 100.0
    out["cognitive_overload_proxy"] = float(np.clip(overload_raw, 0.0, 100.0))

    # Additional condition-based composites
    out["evaluating_thinking_facial"] = float(np.clip(_condition_composite(sig, [("g2_thinking_brow", 50.0, None), ("g2_lip_pucker", 40.0, None), ("g2_look_up_lr", 45.0, None)], 72.0, 25.0), 0.0, 100.0))
    out["cognitive_processing_facial"] = float(np.clip(_condition_composite(sig, [("g2_stillness", 50.0, None), ("g2_gaze_shift_frequency", 48.0, None), ("g2_lowered_brow", 45.0, None)], 70.0, 20.0), 0.0, 100.0))
    resistance_cluster = max(
        _condition_composite(sig, [("g3_contempt", 50.0, None), ("g3_gaze_aversion", 45.0, None)], 75.0, 15.0),
        _condition_composite(sig, [("g3_lip_compression", 45.0, None), ("g3_gaze_aversion", 45.0, None)], 72.0, 15.0),
    )
    out["resistance_cluster_facial"] = float(np.clip(resistance_cluster, 0.0, 100.0))
    out["withdrawal_facial"] = float(np.clip(_condition_composite(sig, [("g3_lip_compression", 45.0, None), ("g3_gaze_aversion", 40.0, None), ("g3_brow_lower_sustained", 45.0, None)], 78.0, 15.0), 0.0, 100.0))
    out["disagreement_facial"] = float(np.clip(_condition_composite(sig, [("g3_head_shake", 50.0, None), ("g3_contempt", 45.0, None)], 80.0, 18.0), 0.0, 100.0))
    out["passive_listening_facial"] = float(np.clip(_condition_composite(sig, [("g1_eye_contact", None, 45.0), ("g3_no_nod", 45.0, None), ("g2_stillness", 50.0, None)], 75.0, 15.0), 0.0, 100.0))
    out["trust_openness_facial"] = float(np.clip(_condition_composite(sig, [("g1_facial_symmetry", 65.0, None), ("g1_softened_forehead", 55.0, None), ("g1_eye_contact", 65.0, None)], 72.0, 25.0), 0.0, 100.0))
    out["curious_engaged_facial"] = float(np.clip(_condition_composite(sig, [("g1_brow_raise_sustained", 50.0, None), ("g1_eye_contact", 65.0, None), ("g1_mouth_open_receptive", 50.0, None)], 74.0, 22.0), 0.0, 100.0))

    return out

# ============== 7. Capability ==============
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None
try:
    import psutil
except ImportError:
    psutil = None

import config

# -----------------------------------------------------------------------------
# Detection capability (device tier, Azure latency, method recommendation)
# -----------------------------------------------------------------------------
DEVICE_TIER_LOW = "low"
DEVICE_TIER_MEDIUM = "medium"
DEVICE_TIER_HIGH = "high"
DEFAULT_AZURE_LATENCY_THRESHOLD_MS = 500.0
HIGH_TIER_CPU_COUNT = 4
HIGH_TIER_MEMORY_GB = 8.0


def get_cpu_count() -> int:
    """Return number of CPUs (logical)."""
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2


def get_memory_gb() -> Optional[float]:
    """Return total system memory in GB if psutil available, else None."""
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def get_device_tier() -> str:
    """Estimate device capability: low, medium, or high."""
    cpus = get_cpu_count()
    mem_gb = get_memory_gb()
    if cpus >= HIGH_TIER_CPU_COUNT and (mem_gb is None or mem_gb >= HIGH_TIER_MEMORY_GB):
        return DEVICE_TIER_HIGH
    if cpus >= 2 and (mem_gb is None or mem_gb >= 4.0):
        return DEVICE_TIER_MEDIUM
    return DEVICE_TIER_LOW


def get_azure_latency_ms(base_url: Optional[str] = None, timeout_sec: float = 5.0) -> Optional[float]:
    """Measure round-trip latency (ms) to backend config endpoint. Returns None on failure."""
    if requests is None:
        return None
    url = (base_url or getattr(config, "BACKEND_BASE_URL", None) or "http://localhost:5000").rstrip("/")
    try:
        start = time.perf_counter()
        r = requests.get(url + "/config/face-detection", timeout=timeout_sec)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return round(elapsed_ms, 2) if r.status_code == 200 else None
    except Exception:
        return None


def recommend_detection_method(
    azure_available: bool = True,
    device_tier: Optional[str] = None,
    azure_latency_ms: Optional[float] = None,
    latency_threshold_ms: Optional[float] = None,
    prefer_local: bool = False,
    return_unified_when_viable: bool = True,
) -> str:
    """Recommend 'mediapipe', 'azure_face_api', or 'unified' based on tier and latency."""
    if not azure_available or prefer_local:
        return "mediapipe"
    tier = device_tier or get_device_tier()
    threshold = latency_threshold_ms if latency_threshold_ms is not None else getattr(
        config, "AZURE_LATENCY_THRESHOLD_MS", DEFAULT_AZURE_LATENCY_THRESHOLD_MS
    )
    if tier == DEVICE_TIER_LOW:
        return "mediapipe"
    if azure_latency_ms is not None and azure_latency_ms > threshold:
        return "mediapipe"
    if tier == DEVICE_TIER_HIGH:
        return "unified" if return_unified_when_viable else "azure_face_api"
    if azure_latency_ms is not None and azure_latency_ms > threshold * 0.6:
        return "mediapipe"
    return "unified" if return_unified_when_viable else "azure_face_api"


def evaluate_capability(base_url: Optional[str] = None) -> Tuple[str, str, Optional[float], str]:
    """Returns (device_tier, recommended_method, azure_latency_ms, reason)."""
    tier = get_device_tier()
    try:
        cfg = config.get_face_detection_config()
        azure_available = cfg.get("azureFaceApiAvailable", False)
    except Exception:
        azure_available = False
    latency_ms = get_azure_latency_ms(base_url=base_url) if requests else None
    threshold = getattr(config, "AZURE_LATENCY_THRESHOLD_MS", DEFAULT_AZURE_LATENCY_THRESHOLD_MS)
    method = recommend_detection_method(
        azure_available=bool(azure_available), device_tier=tier,
        azure_latency_ms=latency_ms, latency_threshold_ms=threshold,
        return_unified_when_viable=True,
    )
    if not azure_available:
        reason = "Azure Face API not configured or unavailable"
    elif tier == DEVICE_TIER_LOW:
        reason = "Device tier is low; using local MediaPipe"
    elif latency_ms is not None and latency_ms > threshold:
        reason = f"Azure latency {latency_ms:.0f} ms above threshold; using MediaPipe"
    elif method == "unified":
        reason = f"Device tier {tier}, latency {latency_ms} ms; using unified (MediaPipe + Azure)"
    else:
        reason = f"Device tier {tier}, latency {latency_ms} ms; using Azure Face API"
    return tier, method, latency_ms, reason


# -----------------------------------------------------------------------------
# Metric selection (active signifiers/composites per tier)
# -----------------------------------------------------------------------------
def _get_full_signifier_keys() -> List[str]:
    return list(SIGNIFIER_KEYS)


FULL_SPEECH_CATEGORIES: List[str] = [
    "objection", "interest", "confusion", "commitment", "concern",
    "timeline", "budget", "realization",
    "urgency", "skepticism", "enthusiasm", "authority", "hesitation", "confirmation",
]
FULL_ACOUSTIC_TAGS: List[str] = [
    "acoustic_disengagement_risk", "acoustic_arousal_high", "acoustic_uncertainty",
    "acoustic_tension", "acoustic_roughness_proxy", "acoustic_falling_contour",
    "acoustic_monotone", "acoustic_emphasis_proxy", "acoustic_creakiness_proxy",
    "acoustic_breathiness_proxy", "acoustic_speech_rate_high", "acoustic_speech_rate_low",
]
FULL_COMPOSITE_KEYS: List[str] = [
    "verbal_nonverbal_alignment", "cognitive_load_multimodal", "rapport_engagement",
    "skepticism_objection_strength", "decision_readiness_multimodal", "opportunity_strength",
    "trust_rapport", "disengagement_risk_multimodal", "confusion_multimodal",
    "tension_objection_multimodal", "loss_of_interest_multimodal", "decision_plus_voice",
    "psychological_safety_proxy", "urgency_sensitivity", "skepticism_strength",
    "enthusiasm_multimodal", "hesitation_multimodal", "authority_deferral",
    "rapport_depth", "cognitive_overload_proxy",
    "topic_interest_facial", "active_listening_facial", "agreement_signals_facial",
    "evaluating_thinking_facial", "cognitive_processing_facial", "resistance_cluster_facial",
    "receptivity_facial", "withdrawal_facial", "disagreement_facial", "closing_window_facial",
    "passive_listening_facial", "trust_openness_facial", "curious_engaged_facial",
]
MEDIUM_DROP_SIGNIFIERS: List[str] = [
    "g1_pupil_dilation", "g1_eyebrow_flash", "g1_facial_symmetry", "g1_softened_forehead",
    "g1_micro_smile", "g1_brow_raise_sustained", "g1_eye_widening",
    "g2_chin_stroke", "g2_stillness", "g2_brow_furrow_deep", "g2_gaze_shift_frequency",
    "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover", "g3_lip_corner_dip", "g3_brow_lower_sustained",
]
LOW_KEEP_SIGNIFIERS: List[str] = [
    "g1_duchenne", "g1_eye_contact", "g1_head_tilt", "g1_forward_lean",
    "g1_rhythmic_nodding", "g1_parted_lips", "g1_mouth_open_receptive", "g1_nod_intensity",
    "g2_look_up_lr", "g2_eye_squint", "g2_thinking_brow", "g2_lowered_brow", "g2_mouth_tight_eval",
    "g3_contempt", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_eye_squeeze", "g3_head_shake",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition", "g4_mouth_relax", "g4_smile_sustain",
]
MEDIUM_DROP_COMPOSITES: List[str] = [
    "opportunity_strength", "trust_rapport", "urgency_sensitivity", "enthusiasm_multimodal",
    "rapport_depth", "psychological_safety_proxy",
]
LOW_KEEP_COMPOSITES: List[str] = [
    "verbal_nonverbal_alignment", "cognitive_load_multimodal", "rapport_engagement",
    "skepticism_objection_strength", "decision_readiness_multimodal",
    "disengagement_risk_multimodal", "confusion_multimodal", "tension_objection_multimodal",
    "loss_of_interest_multimodal", "decision_plus_voice",
    "skepticism_strength", "hesitation_multimodal", "authority_deferral", "cognitive_overload_proxy",
    "topic_interest_facial", "active_listening_facial", "agreement_signals_facial",
    "closing_window_facial", "resistance_cluster_facial", "passive_listening_facial",
]


@dataclass
class MetricConfig:
    """Active metric configuration based on system tier."""
    tier: str
    signifier_keys: List[str]
    speech_categories: List[str]
    acoustic_tags: List[str]
    composite_keys: List[str]


def _get_system_resources() -> Tuple[int, Optional[float]]:
    """Return (cpu_cores, ram_gb)."""
    return (get_cpu_count(), get_memory_gb())


def _determine_tier(cpu_cores: int, ram_gb: Optional[float], override: Optional[str]) -> str:
    if override and override.lower() in ("high", "medium", "low"):
        return override.lower()
    if cpu_cores >= 4 and (ram_gb is None or ram_gb >= 8.0):
        return "high"
    if cpu_cores <= 2 or (ram_gb is not None and ram_gb < 4.0):
        return "low"
    return "medium"


def _build_config(tier: str) -> MetricConfig:
    full_signifiers = _get_full_signifier_keys()
    if tier == "high":
        return MetricConfig(
            tier="high",
            signifier_keys=list(full_signifiers),
            speech_categories=list(FULL_SPEECH_CATEGORIES),
            acoustic_tags=list(FULL_ACOUSTIC_TAGS),
            composite_keys=list(FULL_COMPOSITE_KEYS),
        )
    if tier == "medium":
        signifiers = [k for k in full_signifiers if k not in MEDIUM_DROP_SIGNIFIERS]
        composites = [k for k in FULL_COMPOSITE_KEYS if k not in MEDIUM_DROP_COMPOSITES]
        return MetricConfig(
            tier="medium",
            signifier_keys=signifiers,
            speech_categories=list(FULL_SPEECH_CATEGORIES),
            acoustic_tags=list(FULL_ACOUSTIC_TAGS),
            composite_keys=composites,
        )
    signifiers = [k for k in full_signifiers if k in LOW_KEEP_SIGNIFIERS]
    composites = [k for k in FULL_COMPOSITE_KEYS if k in LOW_KEEP_COMPOSITES]
    return MetricConfig(
        tier="low",
        signifier_keys=signifiers,
        speech_categories=list(FULL_SPEECH_CATEGORIES)[:8],
        acoustic_tags=list(FULL_ACOUSTIC_TAGS)[:6],
        composite_keys=composites,
    )


def get_active_metrics(
    cpu_cores: Optional[int] = None,
    ram_gb: Optional[float] = None,
    network_latency_ms: Optional[float] = None,
    override: Optional[str] = None,
) -> MetricConfig:
    """Return active metric configuration based on system resources."""
    if cpu_cores is None or ram_gb is None:
        detected_cores, detected_ram = _get_system_resources()
        cpu_cores = cpu_cores if cpu_cores is not None else detected_cores
        ram_gb = ram_gb if ram_gb is not None else detected_ram
    tier = _determine_tier(cpu_cores, ram_gb, override)
    return _build_config(tier)


def get_active_metrics_with_config() -> MetricConfig:
    """Get active metrics using config (METRIC_SELECTOR_ENABLED, METRIC_SELECTOR_OVERRIDE)."""
    try:
        if not getattr(config, "METRIC_SELECTOR_ENABLED", True):
            return _build_config("high")
        override = getattr(config, "METRIC_SELECTOR_OVERRIDE", None)
        return get_active_metrics(override=override)
    except Exception:
        return _build_config("high")

# ============== 8. B2B opportunity detector ==============
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Type for recent speech tags (from insight_generator.get_recent_speech_tags)
SpeechTag = Dict[str, Any]  # {"category": str, "phrase": str, "time": float}

# Cooldowns by polarity (config overrides if present)
NEGATIVE_OPPORTUNITY_COOLDOWN_SEC = float(getattr(config, "NEGATIVE_OPPORTUNITY_COOLDOWN_SEC", 14))
POSITIVE_OPPORTUNITY_COOLDOWN_SEC = float(getattr(config, "POSITIVE_OPPORTUNITY_COOLDOWN_SEC", 50))
_HISTORY_LEN = 12

# Negative = risk to relationship/deal; surface more readily (shorter cooldown, lower thresholds)
NEGATIVE_OPPORTUNITY_IDS: set = {
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
    "cognitive_overload_risk",
    "confusion_moment",
    "need_clarity",
    "skepticism_surface",
    "objection_moment",
    "resistance_peak",
    "hesitation_moment",
    "disengagement_risk",
}
# Positive = moments to capitalize; require stronger evidence, longer cooldown
POSITIVE_OPPORTUNITY_IDS: set = {
    "decision_readiness_multimodal",
    "aha_insight_multimodal",
    "closing_window",
    "decision_ready",
    "ready_to_sign",
    "buying_signal",
    "commitment_cue",
    "objection_fading",
    "aha_moment",
    "re_engagement_opportunity",
    "alignment_cue",
    "genuine_interest",
    "listening_active",
    "trust_building_moment",
    "urgency_sensitive",
    "processing_deep",
    "attention_peak",
    "rapport_moment",
}

# Priority order: all NEGATIVE first so we don't miss concerns, then positive
OPPORTUNITY_PRIORITY: List[str] = [
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
    "cognitive_overload_risk",
    "confusion_moment",
    "need_clarity",
    "skepticism_surface",
    "objection_moment",
    "resistance_peak",
    "hesitation_moment",
    "disengagement_risk",
    "decision_readiness_multimodal",
    "aha_insight_multimodal",
    "closing_window",
    "decision_ready",
    "ready_to_sign",
    "buying_signal",
    "commitment_cue",
    "objection_fading",
    "aha_moment",
    "re_engagement_opportunity",
    "alignment_cue",
    "genuine_interest",
    "listening_active",
    "trust_building_moment",
    "urgency_sensitive",
    "processing_deep",
    "attention_peak",
    "rapport_moment",
]

# Opportunity IDs that require recent_speech_tags (multimodal)
MULTIMODAL_OPPORTUNITY_IDS: set = {
    "decision_readiness_multimodal",
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "aha_insight_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
}

# Opportunity IDs that need composite_metrics and/or acoustic_tags (passed to evaluator)
COMPOSITE_ACOUSTIC_OPPORTUNITY_IDS: set = {
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
}

_last_fire_time: Dict[str, float] = {}
_history_means: Dict[str, deque] = {k: deque(maxlen=_HISTORY_LEN) for k in ("g1", "g2", "g3", "g4")}


def _g3_raw(g3: float) -> float:
    """Resistance raw: high = more resistance. G3 in API is 100 - resistance."""
    return 100.0 - float(g3)


def _update_history(group_means: Dict[str, float]) -> None:
    for k in ("g1", "g2", "g3", "g4"):
        v = group_means.get(k, 0.0)
        _history_means[k].append(float(v))


def _hist_list(key: str) -> List[float]:
    return list(_history_means[key])


def _check_cooldown(opportunity_id: str, now: float) -> bool:
    """Use shorter cooldown for negative (retention), longer for positive (reduce frequency)."""
    t = _last_fire_time.get(opportunity_id, 0.0)
    sec = NEGATIVE_OPPORTUNITY_COOLDOWN_SEC if opportunity_id in NEGATIVE_OPPORTUNITY_IDS else POSITIVE_OPPORTUNITY_COOLDOWN_SEC
    return (now - t) >= sec


def _fire(opportunity_id: str, now: float) -> None:
    _last_fire_time[opportunity_id] = now


def _has_recent_category(speech_tags: List[SpeechTag], categories: List[str]) -> bool:
    """True if any recent speech tag has category in categories."""
    if not speech_tags or not categories:
        return False
    cats = set(categories)
    return any((t.get("category") or "") in cats for t in speech_tags)


def _eval_closing_window(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: higher threshold to reduce frequency; only clear closing signals."""
    if len(hist["g4"]) < 6:
        return False
    h4 = hist["g4"]
    if g4 >= 65 and (g4 - min(h4)) >= 18 and g3 >= 55:
        return True
    return False


def _eval_decision_ready(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Decision-ready = high commitment (G4) + positive engagement (G1) + low resistance.
    Positive: higher threshold so positive insights only when multiple strong signals.
    """
    return g4 >= 70 and g1 >= 62 and g3 >= 62


def _eval_ready_to_sign(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Ready to sign = high commitment (G4) + LOW cognitive load (G2 < 50) +
    low resistance. Positive: higher threshold to reduce frequency.
    """
    return g4 >= 70 and g2 < 50 and g3 >= 58


def _eval_buying_signal(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Buying signal = high interest (G1) + decision cues (G4) + low resistance.
    Positive: higher threshold so we only surface on clear multiple positive signals.
    """
    return g1 >= 66 and g4 >= 60 and g3 >= 62


def _eval_commitment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if len(hist["g4"]) < 4:
        return False
    return g4 >= 62 and np.mean(hist["g4"][-4:]) >= 58 and g3 >= 58


def _eval_cognitive_overload_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Cognitive overload = high G2 + LOW G1. Negative: lower threshold for early catch.
    """
    return g2 >= 54 and g1 < 54


def _eval_confusion_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Confusion = high G2 + resistance or G2 spike. Negative: lower threshold so we don't miss.
    """
    r = _g3_raw(g3)
    return g2 >= 53 and (r >= 46 or (len(hist["g2"]) >= 4 and g2 > np.mean(hist["g2"][:-2])))


def _eval_need_clarity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Need clarity = moderate G2 + low G1. Negative: lower threshold so we don't miss subtle need.
    """
    return g2 >= 50 and g1 < 60


def _eval_skepticism_surface(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Skepticism = resistance rising. Negative: lower threshold for early catch.
    """
    r = _g3_raw(g3)
    if len(hist["g3"]) < 4:
        return r >= 46
    mean_g3 = np.mean(hist["g3"][:-1])
    r_prev = 100.0 - mean_g3
    return r >= 46 and r > r_prev + 3


def _eval_objection_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Objection = moderate-high resistance. Negative: lower threshold for early catch.
    Temporal consistency: require G3_raw >= 48 in at least 2 of last 3 frames (group_history)
    so one-frame spikes do not fire.
    """
    r = _g3_raw(g3)
    if r < 48:
        return False
    if len(hist["g3"]) < 3:
        return True
    # 2 of last 3 frames with G3_raw >= 48 (i.e. G3 <= 52)
    recent = hist["g3"][-3:]
    count_high_r = sum(1 for g in recent if _g3_raw(g) >= 48)
    return count_high_r >= 2


def _eval_resistance_peak(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Resistance peak = high G3 raw. Negative: lower threshold for early catch.
    Temporal consistency: require G3_raw >= 52 in at least 2 of last 3 frames so one-frame
    spikes do not fire.
    """
    r = _g3_raw(g3)
    if r < 52:
        return False
    if len(hist["g3"]) < 3:
        return True
    recent = hist["g3"][-3:]
    count_high_r = sum(1 for g in recent if _g3_raw(g) >= 52)
    return count_high_r >= 2


def _eval_hesitation_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Hesitation = moderate G2 + moderate resistance. Negative: lower threshold.
    """
    return g2 >= 52 and _g3_raw(g3) >= 44


def _eval_disengagement_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Disengagement = LOW G1 + moderate resistance. Negative: lower threshold to catch slippage.
    """
    return g1 < 52 and _g3_raw(g3) >= 44


def _eval_objection_fading(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g3"]) < 5:
        return False
    mean_prev = np.mean(hist["g3"][:-2])
    return mean_prev < 52 and g3 >= mean_prev + 10


def _eval_aha_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Aha = was processing -> now engaged. Positive: modestly higher bar.
    """
    if len(hist["g1"]) < 3 or len(hist["g2"]) < 3:
        return False
    g2_ago = hist["g2"][-2]
    return g2_ago >= 56 and g1 >= 60 and g3 >= 56


def _eval_re_engagement_opportunity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Re-engagement = was disengaged -> attention returning. Positive: slightly higher g1.
    """
    if len(hist["g1"]) < 6:
        return False
    return min(hist["g1"]) < 44 and g1 >= 52


def _eval_alignment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Alignment = G1 and G4 rising together. Positive: require slightly larger rise.
    """
    if len(hist["g1"]) < 6 or len(hist["g4"]) < 6:
        return False
    return (g1 - hist["g1"][0]) >= 12 and (g4 - hist["g4"][0]) >= 12


def _eval_genuine_interest(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Psychology: Genuine interest = high G1 + low resistance. Positive: higher threshold for multi-signal."""
    return g1 >= 64 and g3 >= 60


def _eval_listening_active(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: slightly higher bar to reduce frequency."""
    return g1 >= 57 and g3 >= 56


def _eval_trust_building_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 55:
        return g1 >= 58 and g3 >= 58
    return g1 >= 58 and g3 >= 58


def _eval_urgency_sensitive(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g4 >= 56 and g2 >= 50 and g3 >= 54


def _eval_processing_deep(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 56 and g3 >= 58


def _eval_attention_peak(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_eye_contact", 0) >= 58:
        return g1 >= 64
    return g1 >= 66


def _eval_rapport_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 52:
        return g1 >= 58
    return g1 >= 60


# ----- Multimodal composites (facial + speech; require recent_speech_tags) -----


def _eval_decision_readiness_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Decision-readiness with speech. Positive: higher threshold."""
    if not _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    return g4 >= 64 and g1 >= 58 and g3 >= 58


def _eval_cognitive_overload_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Cognitive overload with speech. Negative: lower threshold for early catch."""
    if not _has_recent_category(speech_tags, ["confusion", "concern"]):
        return False
    return g2 >= 53 and g1 < 56


def _eval_skepticism_objection_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Skepticism/objection with speech. Negative: lower threshold for early catch."""
    if not _has_recent_category(speech_tags, ["objection", "concern"]):
        return False
    return _g3_raw(g3) >= 44


def _eval_aha_insight_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Aha/insight with speech. Positive: slightly higher bar."""
    if not _has_recent_category(speech_tags, ["interest", "realization"]):
        return False
    if len(hist["g1"]) < 3 or len(hist["g2"]) < 3:
        return False
    g2_ago = hist["g2"][-2]
    return g2_ago >= 53 and g1 >= 60 and g3 >= 55


def _eval_disengagement_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Disengagement with lack of positive speech. Negative: lower threshold for early catch."""
    if _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    return g1 < 48 and g4 < 52 and _g3_raw(g3) >= 42


# ----- Composite + acoustic opportunity evaluators (need composite_metrics, acoustic_tags) -----


def _eval_confusion_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Confusion: speech confusion/concern AND (G2 high or composite confusion_multimodal high). Negative."""
    if not _has_recent_category(speech_tags, ["confusion", "concern"]):
        return False
    comp = composite_metrics or {}
    if comp.get("confusion_multimodal", 0) >= 55:
        return True
    return g2 >= 50


def _eval_tension_objection_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Tension/objection: G3 raw elevated AND (recent objection/concern or composite high). Negative."""
    if _g3_raw(g3) < 42:
        return False
    if _has_recent_category(speech_tags, ["objection", "concern"]):
        return True
    comp = composite_metrics or {}
    return comp.get("tension_objection_multimodal", 0) >= 55


def _eval_loss_of_interest(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Loss of interest: composite loss_of_interest_multimodal high OR (G1 low + no commit/interest + acoustic withdrawal). Negative."""
    comp = composite_metrics or {}
    if comp.get("loss_of_interest_multimodal", 0) >= 58:
        return True
    if g1 >= 50:
        return False
    if _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    ac = set(acoustic_tags or [])
    if "acoustic_disengagement_risk" in ac:
        return True
    return g1 < 45 and not _has_recent_category(speech_tags, ["commitment", "interest"])


def _eval_acoustic_disengagement_risk(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice suggests disengagement and face not highly engaged. Negative: don't miss withdrawal."""
    ac = set(acoustic_tags or [])
    if "acoustic_disengagement_risk" not in ac:
        return False
    return g1 < 60


def _eval_acoustic_uncertainty(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice uncertainty + (elevated G2 or recent confusion/concern). Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_uncertainty" not in ac:
        return False
    if g2 >= 50:
        return True
    return _has_recent_category(speech_tags, ["confusion", "concern"])


def _eval_acoustic_tension(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice tension + resistance. Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_tension" not in ac:
        return False
    return _g3_raw(g3) >= 40


def _eval_acoustic_roughness_proxy(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice roughness proxy (strain/tension). Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_roughness_proxy" not in ac:
        return False
    return _g3_raw(g3) >= 35


_EVALUATORS: Dict[str, Any] = {
    "decision_readiness_multimodal": _eval_decision_readiness_multimodal,
    "cognitive_overload_multimodal": _eval_cognitive_overload_multimodal,
    "skepticism_objection_multimodal": _eval_skepticism_objection_multimodal,
    "aha_insight_multimodal": _eval_aha_insight_multimodal,
    "disengagement_multimodal": _eval_disengagement_multimodal,
    "confusion_multimodal": _eval_confusion_multimodal,
    "tension_objection_multimodal": _eval_tension_objection_multimodal,
    "loss_of_interest": _eval_loss_of_interest,
    "acoustic_disengagement_risk": _eval_acoustic_disengagement_risk,
    "acoustic_uncertainty": _eval_acoustic_uncertainty,
    "acoustic_tension": _eval_acoustic_tension,
    "acoustic_roughness_proxy": _eval_acoustic_roughness_proxy,
    "closing_window": _eval_closing_window,
    "decision_ready": _eval_decision_ready,
    "ready_to_sign": _eval_ready_to_sign,
    "buying_signal": _eval_buying_signal,
    "commitment_cue": _eval_commitment_cue,
    "cognitive_overload_risk": _eval_cognitive_overload_risk,
    "confusion_moment": _eval_confusion_moment,
    "need_clarity": _eval_need_clarity,
    "skepticism_surface": _eval_skepticism_surface,
    "objection_moment": _eval_objection_moment,
    "resistance_peak": _eval_resistance_peak,
    "hesitation_moment": _eval_hesitation_moment,
    "disengagement_risk": _eval_disengagement_risk,
    "objection_fading": _eval_objection_fading,
    "aha_moment": _eval_aha_moment,
    "re_engagement_opportunity": _eval_re_engagement_opportunity,
    "alignment_cue": _eval_alignment_cue,
    "genuine_interest": _eval_genuine_interest,
    "listening_active": _eval_listening_active,
    "trust_building_moment": _eval_trust_building_moment,
    "urgency_sensitive": _eval_urgency_sensitive,
    "processing_deep": _eval_processing_deep,
    "attention_peak": _eval_attention_peak,
    "rapport_moment": _eval_rapport_moment,
}


def detect_opportunity(
    group_means: Dict[str, float],
    group_history: Optional[Dict[str, deque]] = None,
    signifier_scores: Optional[Dict[str, float]] = None,
    now: Optional[float] = None,
    recent_speech_tags: Optional[List[SpeechTag]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Evaluate B2B opportunity features in priority order. Returns (opportunity_id, context) for the
    first opportunity that fires and has passed cooldown, else None.

    Args:
        group_means: Current G1, G2, G3, G4 (0–100).
        group_history: Optional dict of deques (g1, g2, g3, g4) for temporal logic.
        signifier_scores: Optional 30 signifier scores.
        now: Timestamp for cooldown; defaults to time.time().
        recent_speech_tags: Optional list of recent speech tags for multimodal composites.
        composite_metrics: Optional composite metrics (confusion_multimodal, tension_objection_multimodal, etc.).
        acoustic_tags: Optional list of acoustic tags (e.g. from get_recent_acoustic_tags()).

    Returns:
        (opportunity_id, context) or None.
    """
    now = now if now is not None else time.time()
    g1 = float(group_means.get("g1", 0))
    g2 = float(group_means.get("g2", 0))
    g3 = float(group_means.get("g3", 0))
    g4 = float(group_means.get("g4", 0))

    if group_history is not None:
        for k in ("g1", "g2", "g3", "g4"):
            q = group_history.get(k)
            if q is not None:
                _history_means[k] = deque(q, maxlen=_HISTORY_LEN)
    else:
        _update_history(group_means)

    hist = {k: _hist_list(k) for k in ("g1", "g2", "g3", "g4")}
    sig = signifier_scores
    speech_tags = recent_speech_tags if recent_speech_tags is not None else []
    comp = composite_metrics
    ac_tags = acoustic_tags if acoustic_tags is not None else []

    for oid in OPPORTUNITY_PRIORITY:
        if not _check_cooldown(oid, now):
            continue
        fn = _EVALUATORS.get(oid)
        if fn is None:
            continue
        try:
            if oid in COMPOSITE_ACOUSTIC_OPPORTUNITY_IDS:
                fired = fn(g1, g2, g3, g4, hist, speech_tags, comp, ac_tags)
            elif oid in MULTIMODAL_OPPORTUNITY_IDS:
                fired = fn(g1, g2, g3, g4, hist, speech_tags)
            elif oid in ("trust_building_moment", "attention_peak", "rapport_moment"):
                fired = fn(g1, g2, g3, g4, hist, sig)
            else:
                fired = fn(g1, g2, g3, g4, hist)
            if fired:
                _fire(oid, now)
                context = {
                    "g1": g1, "g2": g2, "g3": g3, "g4": g4,
                    "signifier_scores": sig,
                    "recent_speech_tags": speech_tags,
                    "composite_metrics": comp,
                    "acoustic_tags": ac_tags,
                }
                return (oid, context)
        except Exception:
            continue
    return None


def update_history_from_detector(group_means: Dict[str, float]) -> None:
    """Update internal history when caller uses external group_history (e.g. engagement_detector)."""
    _update_history(group_means)


def clear_opportunity_state() -> None:
    """Clear cooldowns and history (e.g. when engagement stops)."""
    global _last_fire_time, _history_means
    _last_fire_time.clear()
    for k in _history_means:
        _history_means[k].clear()
