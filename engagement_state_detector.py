"""
Engagement State Detector Module

This module provides real-time engagement state detection from video feeds.
It uses face detection (MediaPipe or Azure Face API) to extract facial features
and compute engagement scores and metrics.

The system:
- Captures video from various sources (webcam, files, streams)
- Detects faces and extracts landmarks
- Computes 100 key blendshape features
- Calculates engagement metrics and overall score (0-100)
- Generates contextual information for AI coaching
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Tuple, List
import numpy as np
import config
from utils.video_source_handler import VideoSourceHandler, VideoSourceType
from utils.engagement_scorer import EngagementScorer, EngagementMetrics
from utils.context_generator import ContextGenerator, EngagementContext
from utils.face_detection_interface import FaceDetectorInterface, FaceDetectionResult
from utils.mediapipe_detector import MediaPipeFaceDetector
from utils.azure_face_detector import AzureFaceAPIDetector
from utils.azure_landmark_mapper import expand_azure_landmarks_to_mediapipe
from utils.business_meeting_feature_extractor import BusinessMeetingFeatureExtractor
from utils.expression_signifiers import ExpressionSignifierEngine
from utils import signifier_weights


class EngagementLevel(Enum):
    """Enumeration of engagement levels based on score ranges."""
    VERY_LOW = "VERY_LOW"    # 0-25
    LOW = "LOW"              # 25-45
    MEDIUM = "MEDIUM"        # 45-70
    HIGH = "HIGH"            # 70-85
    VERY_HIGH = "VERY_HIGH"  # 85-100
    
    @classmethod
    def from_score(cls, score: float) -> 'EngagementLevel':
        """
        Determine engagement level from score.
        
        Args:
            score: Engagement score (0-100)
        
        Returns:
            EngagementLevel enum value
        """
        if score >= 85:
            return cls.VERY_HIGH
        elif score >= 70:
            return cls.HIGH
        elif score >= 45:
            return cls.MEDIUM
        elif score >= 25:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class EngagementState:
    """Data class representing the current engagement state."""
    score: float  # Overall engagement score (0-100)
    level: EngagementLevel  # Engagement level category
    metrics: EngagementMetrics  # Detailed metrics breakdown
    context: EngagementContext  # Contextual information for AI
    timestamp: float  # Unix timestamp of measurement
    face_detected: bool  # Whether a face was detected in current frame
    confidence: float  # Confidence in the measurement (0-1)
    signifier_scores: Optional[dict] = None  # 30 expression signifier scores (0-100 each)


class EngagementStateDetector:
    """
    Main engagement state detector class.
    
    This class orchestrates video capture, face detection, feature extraction,
    scoring, and context generation to provide real-time engagement analysis.
    
    Usage:
        detector = EngagementStateDetector()
        detector.start_detection(source_type=VideoSourceType.WEBCAM)
        
        # In a loop or callback:
        state = detector.get_current_state()
        print(f"Engagement: {state.score:.1f} ({state.level.name})")
    """
    
    def __init__(
        self,
        smoothing_window: int = 10,
        min_face_confidence: float = 0.5,
        update_callback: Optional[Callable[[EngagementState], None]] = None,
        detection_method: Optional[str] = None,
        lightweight_mode: bool = False,
    ):
        """
        Initialize the engagement state detector.
        
        Args:
            smoothing_window: Number of frames to average for smoothing (default: 10)
            min_face_confidence: Minimum confidence for face detection (default: 0.5)
            update_callback: Optional callback function called when state updates
            detection_method: Face detection method to use ("mediapipe" or "azure_face_api").
                             If None, uses config.FACE_DETECTION_METHOD
            lightweight_mode: If True, use MediaPipe only, smaller buffer, process every 2nd
                             frame, and skip 100-feature extractor (for low-power devices).
        """
        self.lightweight_mode = bool(lightweight_mode)
        self._min_face_confidence = min_face_confidence
        if self.lightweight_mode:
            detection_method = "mediapipe"

        if detection_method is None:
            # Default to MediaPipe if not specified
            detection_method = config.FACE_DETECTION_METHOD.lower() or "mediapipe"

        # Initialize face detector (lightweight forces MediaPipe)
        self.face_detector: Optional[FaceDetectorInterface] = None
        self._azure_consecutive_empty = 0
        self._azure_fallback_threshold = 10  # switch to MediaPipe after this many empty Azure results

        if detection_method == "azure_face_api":
            try:
                self.face_detector = AzureFaceAPIDetector()
                if not self.face_detector.is_available():
                    print("Warning: Azure Face API not available, falling back to MediaPipe")
                    self.face_detector = MediaPipeFaceDetector(min_detection_confidence=min_face_confidence)
            except Exception as e:
                print(f"Warning: Failed to initialize Azure Face API: {e}. Falling back to MediaPipe")
                self.face_detector = MediaPipeFaceDetector(min_detection_confidence=min_face_confidence)
        else:
            # Default to MediaPipe
            self.face_detector = MediaPipeFaceDetector(min_detection_confidence=min_face_confidence)
        
        self.detection_method = self.face_detector.get_name()
        
        # Component initialization
        self.video_handler = VideoSourceHandler()
        self.scorer = EngagementScorer()
        self.context_generator = ContextGenerator()
        self.feature_extractor = BusinessMeetingFeatureExtractor()
        buf = 15 if self.lightweight_mode else 45
        self.signifier_engine = ExpressionSignifierEngine(
            buffer_frames=buf,
            weights_provider=signifier_weights.get_weights,
        )

        # State management
        self.current_state: Optional[EngagementState] = None
        self.score_history: deque = deque(maxlen=smoothing_window)
        self.metrics_history: deque = deque(maxlen=smoothing_window)
        
        # Track consecutive frames without face
        self.consecutive_no_face_frames = 0
        self.max_no_face_frames = 30  # Reset after ~1 second at 30 FPS
        
        # Note: Using simple 10-frame moving average instead of exponential
        
        # Threading and control
        self.detection_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Callback for state updates
        self.update_callback = update_callback
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Alert detection: significant drop and plateau (sampled ~1/sec, 35 sec window)
        self._alert_history: deque = deque(maxlen=35)
        self._last_alert_push_time: float = 0.0
        self._pending_alert: Optional[dict] = None
        self._last_drop_alert_time: float = 0.0
        self._last_plateau_alert_time: float = 0.0
        self._alert_cooldown_sec: float = 60.0
        self._drop_threshold: float = 15.0   # points below baseline to trigger drop
        self._plateau_std_max: float = 3.0   # max std for "flat" plateau
        self._plateau_mean_max: float = 85.0 # plateau only if mean < this (below max engagement)
        self._plateau_mean_min: float = 25.0 # ignore very low plateaus (likely no face / noise)
        self._frame_count: int = 0

    def start_detection(
        self,
        source_type: VideoSourceType = VideoSourceType.WEBCAM,
        source_path: Optional[str] = None
    ) -> bool:
        """
        Start engagement detection from a video source.
        
        Args:
            source_type: Type of video source (WEBCAM, FILE, STREAM)
            source_path: Path to video file or stream URL (required for FILE/STREAM)
        
        Returns:
            bool: True if detection started successfully, False otherwise
        """
        if self.is_running:
            self.stop_detection()
        
        # Reset state tracking
        self.consecutive_no_face_frames = 0
        self._frame_count = 0
        self._azure_consecutive_empty = 0
        self.score_history.clear()
        self.metrics_history.clear()
        self.signifier_engine.reset()
        self._alert_history.clear()
        self._last_alert_push_time = 0.0
        self._pending_alert = None
        self._last_drop_alert_time = 0.0
        self._last_plateau_alert_time = 0.0
        with self.lock:
            self.current_state = None
        
        # Initialize video source
        if not self.video_handler.initialize_source(source_type, source_path):
            print(f"Error: Failed to initialize video source type {source_type}")
            if source_path:
                print(f"  Source path: {source_path}")
            return False
        
        # Verify we can read at least one frame
        ret, test_frame = self.video_handler.read_frame()
        if not ret or test_frame is None:
            print("Error: Video source initialized but cannot read frames")
            self.video_handler.release()
            return False
        
        print(f"Engagement detection started: source_type={source_type}, source_path={source_path}")
        
        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
        return True
    
    def stop_detection(self) -> None:
        """Stop engagement detection and release resources."""
        self.is_running = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        self.video_handler.release()
        if self.face_detector:
            self.face_detector.close()
    
    def get_current_state(self) -> Optional[EngagementState]:
        """
        Get the current engagement state (thread-safe).
        
        Returns:
            EngagementState if available, None otherwise
        """
        with self.lock:
            return self.current_state
    
    def get_fps(self) -> float:
        """
        Get the current processing FPS.
        
        Returns:
            float: Average FPS over last 30 frames
        """
        if not self.fps_counter:
            return 0.0
        return float(np.mean(self.fps_counter))
    
    def _detection_loop(self) -> None:
        """
        Main detection loop running in a separate thread.
        
        This method continuously processes video frames, detects faces,
        extracts features, computes scores, and updates the current state.
        """
        while self.is_running:
            try:
                # Read frame from video source
                ret, frame = self.video_handler.read_frame()
                
                if not ret:
                    # No frame available, wait a bit and continue
                    # But track this to detect if video source is stuck
                    self.consecutive_no_face_frames += 1
                    if self.consecutive_no_face_frames > 60:  # ~2 seconds at 30 FPS
                        print("Warning: Video source not providing frames, checking connection...")
                        # Try to reinitialize if possible
                        if self.video_handler.source_type == VideoSourceType.WEBCAM:
                            # For webcam, try to reinitialize
                            self.video_handler.initialize_source(
                                self.video_handler.source_type,
                                self.video_handler.source_path
                            )
                        self.consecutive_no_face_frames = 0
                    time.sleep(0.033)  # ~30 FPS
                    continue
                
                # Process frame and update state
                state = self._process_frame(frame)
                
                if state:
                    with self.lock:
                        self.current_state = state
                    
                    # Call update callback if provided
                    if self.update_callback:
                        try:
                            self.update_callback(state)
                        except Exception as e:
                            print(f"Error in update callback: {e}")
                
                # Track FPS
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time
                if frame_time > 0:
                    self.fps_counter.append(1.0 / frame_time)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[EngagementState]:
        """
        Process a single video frame to extract engagement state.
        
        Args:
            frame: BGR image frame from video source
        
        Returns:
            EngagementState if face detected, None otherwise
        """
        if not self.face_detector:
            return None

        self._frame_count += 1
        if self.lightweight_mode and (self._frame_count % 2) == 1:
            with self.lock:
                last = self.current_state
            return last if last is not None else self._make_no_face_state()
        
        # Detect faces using the configured detector
        face_results = self.face_detector.detect_faces(frame)

        # Runtime fallback: Azure repeatedly returning no faces -> switch to MediaPipe
        if self.face_detector.get_name() == "azure_face_api":
            if not face_results:
                self._azure_consecutive_empty += 1
                if self._azure_consecutive_empty >= self._azure_fallback_threshold:
                    print(
                        "Info: Azure Face API produced repeated empty results; "
                        "switching to MediaPipe for face detection."
                    )
                    self.face_detector = MediaPipeFaceDetector(
                        min_detection_confidence=self._min_face_confidence
                    )
                    self.detection_method = "mediapipe"
                    self._azure_consecutive_empty = 0
                    face_results = self.face_detector.detect_faces(frame)
            else:
                self._azure_consecutive_empty = 0

        if not face_results:
            self.consecutive_no_face_frames += 1
            if self.consecutive_no_face_frames >= self.max_no_face_frames:
                self.consecutive_no_face_frames = 0
            return self._make_no_face_state()

        face_result = face_results[0]
        landmarks = face_result.landmarks
        if landmarks is None or landmarks.size == 0:
            self.consecutive_no_face_frames += 1
            return self._make_no_face_state()

        # Azure 27-point landmarks: expand to MediaPipe 468 layout for signifiers/feature extractor
        if self.face_detector.get_name() == "azure_face_api" and landmarks.shape[0] <= 27:
            bbox = getattr(face_result, "bounding_box", None)
            landmarks = expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame.shape)

        if self.lightweight_mode:
            self.signifier_engine.update(landmarks, face_result, frame.shape)
            signifier_scores = self.signifier_engine.get_all_scores()
            score = self.signifier_engine.get_composite_score(signifier_scores)
            metrics = self._metrics_from_signifiers(signifier_scores)
        else:
            blendshape_features = self.feature_extractor.extract_features(
                landmarks, frame.shape, face_result
            )
            metrics = self.scorer.compute_metrics(blendshape_features, landmarks, frame.shape)
            self.signifier_engine.update(landmarks, face_result, frame.shape)
            signifier_scores = self.signifier_engine.get_all_scores()
            score = self.signifier_engine.get_composite_score(signifier_scores)

        if not np.isfinite(score) or score < 0 or score > 100:
            score = float(self.scorer.calculate_score(metrics)) if not self.lightweight_mode else (
                float(np.mean(list(signifier_scores.values()))) if signifier_scores else 50.0
            )
        score = max(0.0, min(100.0, score))
        
        # Use raw score for real-time, quick-actionable insights (no 10-frame averaging)
        self.score_history.append(score)
        self.metrics_history.append(metrics)
        
        # Real-time score: use current frame score directly
        realtime_score = score
        if not np.isfinite(realtime_score):
            realtime_score = 50.0
        realtime_score = max(0.0, min(100.0, realtime_score))
        smoothed_metrics = self._average_metrics() if self.metrics_history else metrics
        
        # Reset no-face counter since we detected a face
        self.consecutive_no_face_frames = 0
        
        # Sample score ~once per second for alert detection
        now = time.time()
        if now - self._last_alert_push_time >= 1.0:
            self._alert_history.append(float(realtime_score))
            self._last_alert_push_time = now
            self._check_alerts(realtime_score, now)
        
        # Determine engagement level from real-time score
        level = EngagementLevel.from_score(realtime_score)
        
        # Generate context for AI coaching
        context = self.context_generator.generate_context(
            realtime_score,
            smoothed_metrics,
            level
        )
        
        # Calculate confidence based on detection stability
        confidence = self._calculate_confidence()
        
        return EngagementState(
            score=float(realtime_score),
            level=level,
            metrics=smoothed_metrics,
            context=context,
            timestamp=time.time(),
            face_detected=True,
            confidence=confidence,
            signifier_scores=signifier_scores
        )
    
    def _metrics_from_signifiers(self, s: dict) -> EngagementMetrics:
        """Build EngagementMetrics from 30 signifier scores (lightweight path)."""
        def m(*keys: str) -> float:
            vals = [s.get(k, 50.0) for k in keys if k in s]
            return float(np.mean(vals)) if vals else 50.0
        return EngagementMetrics(
            attention=m("g1_duchenne", "g1_eye_contact", "g1_eyebrow_flash", "g1_pupil_dilation"),
            eye_contact=float(s.get("g1_eye_contact", 50.0)),
            facial_expressiveness=m("g1_duchenne", "g1_parted_lips", "g4_smile_transition", "g1_softened_forehead"),
            head_movement=m("g1_head_tilt", "g1_rhythmic_nodding", "g1_forward_lean"),
            symmetry=float(s.get("g1_facial_symmetry", 50.0)),
            mouth_activity=float(s.get("g1_parted_lips", 50.0)),
        )

    def _make_no_face_state(self) -> EngagementState:
        """Build an engagement state with score 0 when no face is detected."""
        zero_metrics = EngagementMetrics()
        context = self.context_generator.generate_context(
            0.0, zero_metrics, EngagementLevel.VERY_LOW
        )
        return EngagementState(
            score=0.0,
            level=EngagementLevel.VERY_LOW,
            metrics=zero_metrics,
            context=context,
            timestamp=time.time(),
            face_detected=False,
            confidence=0.0
        )
    
    def _check_alerts(self, current_score: float, now: float) -> None:
        """
        Check for significant engagement drop or plateau and set _pending_alert.
        Uses _alert_history (one sample per second). Cooldown per alert type.
        """
        arr = list(self._alert_history)
        if len(arr) < 20:
            return
        with self.lock:
            if self._pending_alert is not None:
                return
            # Significant drop: current (recent mean) vs baseline (older mean)
            baseline = float(np.mean(arr[:10]))
            recent = float(np.mean(arr[-5:]))
            if recent < baseline - self._drop_threshold and (now - self._last_drop_alert_time) >= self._alert_cooldown_sec:
                self._pending_alert = {
                    "type": "drop",
                    "message": "Engagement has dropped noticeably. Here are some ways to re-engage: ",
                    "suggestions": [
                        "Ask an open-ended question.",
                        "Pause and check for understanding.",
                        "Share a relevant example or story.",
                        "Invite them to contribute their view.",
                        "Consider a short break or change of pace."
                    ]
                }
                self._last_drop_alert_time = now
                return
            # Plateau: low variance, mean below max engagement, above noise floor
            last20 = arr[-20:]
            std = float(np.std(last20))
            mean = float(np.mean(last20))
            if (std < self._plateau_std_max and
                self._plateau_mean_min < mean < self._plateau_mean_max and
                (now - self._last_plateau_alert_time) >= self._alert_cooldown_sec):
                self._pending_alert = {
                    "type": "plateau",
                    "message": "Engagement has been steady but could be higher. Consider: ",
                    "suggestions": [
                        "Introduce a thought-provoking question.",
                        "Add more energy and variety to your delivery.",
                        "Use their name and invite direct input.",
                        "Share something novel or unexpected.",
                        "Propose a quick interactive element."
                    ]
                }
                self._last_plateau_alert_time = now
    
    def get_and_clear_pending_alert(self) -> Optional[dict]:
        """Thread-safe get and clear of pending alert. Returns None if none."""
        with self.lock:
            a = self._pending_alert
            self._pending_alert = None
            return a
    
    def _average_metrics(self) -> EngagementMetrics:
        """
        Calculate average metrics from history.
        
        Returns:
            EngagementMetrics: Averaged metrics
        """
        if not self.metrics_history:
            return EngagementMetrics()
        
        avg_attention = np.mean([m.attention for m in self.metrics_history])
        avg_eye_contact = np.mean([m.eye_contact for m in self.metrics_history])
        avg_expressiveness = np.mean([m.facial_expressiveness for m in self.metrics_history])
        avg_head_movement = np.mean([m.head_movement for m in self.metrics_history])
        avg_symmetry = np.mean([m.symmetry for m in self.metrics_history])
        avg_mouth_activity = np.mean([m.mouth_activity for m in self.metrics_history])
        
        return EngagementMetrics(
            attention=float(avg_attention),
            eye_contact=float(avg_eye_contact),
            facial_expressiveness=float(avg_expressiveness),
            head_movement=float(avg_head_movement),
            symmetry=float(avg_symmetry),
            mouth_activity=float(avg_mouth_activity)
        )
    
    def _calculate_confidence(self) -> float:
        """
        Calculate confidence in the current measurement based on stability.
        
        Returns:
            float: Confidence score (0-1)
        """
        if len(self.score_history) < 3:
            return 0.5
        
        # Confidence based on score variance (lower variance = higher confidence)
        score_std = np.std(list(self.score_history))
        max_std = 50.0  # Maximum expected standard deviation
        confidence = max(0.0, 1.0 - (score_std / max_std))
        
        return float(confidence)
