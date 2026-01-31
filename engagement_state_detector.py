"""
Engagement State Detector.

Orchestrates real-time engagement from video: face detection (MediaPipe or Azure),
30 signifiers → 4 group means (G1–G4), score (0–100), spike alerts, and B2B opportunity
detection. Alerts (spike / opportunity / aural) are consumed by GET /engagement/state
and turned into popup + TTS via the insight generator. See docs/DOCUMENTATION.md.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Tuple, List
import numpy as np
import cv2
import config
from utils.video_source_handler import VideoSourceHandler, VideoSourceType
from utils.engagement_scorer import EngagementScorer, EngagementMetrics
from utils.context_generator import ContextGenerator, EngagementContext
from utils.face_detection_interface import FaceDetectorInterface, FaceDetectionResult
from utils.mediapipe_detector import MediaPipeFaceDetector
from utils.azure_landmark_mapper import expand_azure_landmarks_to_mediapipe
from utils.business_meeting_feature_extractor import BusinessMeetingFeatureExtractor
from utils.expression_signifiers import ExpressionSignifierEngine
from utils import signifier_weights
from utils.detection_capability import evaluate_capability, recommend_detection_method
from utils.b2b_opportunity_detector import detect_opportunity


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
    detection_method: Optional[str] = None  # "mediapipe" | "azure_face_api"
    azure_metrics: Optional[dict] = None  # When Azure: { base, composite, score }; for frontend


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
        smoothing_window: int = 4,
        min_face_confidence: float = 0.2,
        update_callback: Optional[Callable[[EngagementState], None]] = None,
        detection_method: Optional[str] = None,
        lightweight_mode: bool = False,
    ):
        """
        Initialize the engagement state detector.
        
        Args:
            smoothing_window: Number of frames to average for smoothing (default: 4, low latency)
            min_face_confidence: Minimum confidence for face detection (default: 0.2; lower helps in suboptimal lighting)
            update_callback: Optional callback function called when state updates
            detection_method: Face detection method to use ("mediapipe" or "azure_face_api").
                             If None, uses config.FACE_DETECTION_METHOD
            lightweight_mode: If True, use MediaPipe only, smaller buffer, process every 2nd
                             frame, and skip 100-feature extractor (for low-power devices).
        """
        self.lightweight_mode = bool(lightweight_mode)
        default_conf = getattr(config, "MIN_FACE_CONFIDENCE", 0.15)
        self._min_face_confidence = max(0.1, min(0.9, float(min_face_confidence or default_conf)))
        if self.lightweight_mode:
            detection_method = "mediapipe"

        if detection_method is None:
            detection_method = config.FACE_DETECTION_METHOD.lower() or "mediapipe"

        # Resolve "auto" to mediapipe or azure_face_api based on device + network
        self._effective_method: Optional[str] = None
        if (detection_method or "").lower() == "auto" and getattr(config, "AUTO_DETECTION_SWITCHING", True):
            try:
                _tier, recommended, _latency_ms, reason = evaluate_capability()
                detection_method = recommended
                self._effective_method = recommended
                print(f"Auto detection: {reason} -> {recommended}")
            except Exception as e:
                print(f"Auto detection evaluation failed: {e}, using mediapipe")
                detection_method = "mediapipe"
                self._effective_method = "mediapipe"

        # Initialize face detector(s); Azure imports deferred until needed (faster startup for MediaPipe-only)
        self.face_detector: Optional[FaceDetectorInterface] = None
        self._azure_detector_secondary = None
        self._get_all_azure_metrics = None
        self._get_azure_score_breakdown = None
        self._azure_consecutive_empty = 0
        self._azure_fallback_threshold = 10

        if (detection_method or "").lower() == "unified":
            from utils.azure_face_detector import AzureFaceAPIDetector
            from utils.azure_engagement_metrics import get_all_azure_metrics, get_azure_score_breakdown
            self._get_all_azure_metrics = get_all_azure_metrics
            self._get_azure_score_breakdown = get_azure_score_breakdown
            self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            self.detection_method = "unified"
            try:
                azure_sec = AzureFaceAPIDetector()
                if azure_sec.is_available():
                    self._azure_detector_secondary = azure_sec
                    print("Unified mode: MediaPipe + Azure Face API (fusion enabled)")
                else:
                    print("Unified mode: Azure not available, using MediaPipe only")
            except Exception as e:
                print(f"Unified mode: Azure init failed ({e}), using MediaPipe only")
        elif (detection_method or "").lower() == "azure_face_api":
            from utils.azure_face_detector import AzureFaceAPIDetector
            from utils.azure_engagement_metrics import get_all_azure_metrics, get_azure_score_breakdown
            self._get_all_azure_metrics = get_all_azure_metrics
            self._get_azure_score_breakdown = get_azure_score_breakdown
            try:
                self.face_detector = AzureFaceAPIDetector()
                if not self.face_detector.is_available():
                    print("Warning: Azure Face API not available, falling back to MediaPipe")
                    self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            except Exception as e:
                print(f"Warning: Failed to initialize Azure Face API: {e}. Falling back to MediaPipe")
                self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            self.detection_method = self.face_detector.get_name()
        else:
            self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            self.detection_method = self.face_detector.get_name()
        
        # Component initialization
        self.video_handler = VideoSourceHandler()
        self.scorer = EngagementScorer()
        self.context_generator = ContextGenerator()
        self.feature_extractor = BusinessMeetingFeatureExtractor()
        buf = 12 if self.lightweight_mode else 22
        self.signifier_engine = ExpressionSignifierEngine(
            buffer_frames=buf,
            weights_provider=signifier_weights.get_weights,
        )

        # State management (minimal history for confidence only)
        self.current_state: Optional[EngagementState] = None
        self.score_history: deque = deque(maxlen=3)
        self.metrics_history: deque = deque(maxlen=3)
        
        # Track consecutive frames without face
        self.consecutive_no_face_frames = 0
        self.max_no_face_frames = 30  # Reset after ~1 second at 30 FPS
        
        # Spike detection per group (G1..G4): short window, cooldown per group
        self._group_history: dict = {k: deque(maxlen=12) for k in ("g1", "g2", "g3", "g4")}
        self._last_spike_time: dict = {k: 0.0 for k in ("g1", "g2", "g3", "g4")}
        self._spike_cooldown_sec: float = 45.0
        self._spike_delta_threshold: float = 22.0  # sudden rise above recent min
        self._spike_min_value: float = 42.0  # only fire if current group mean >= this
        
        # Threading and control
        self.detection_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Callback for state updates
        self.update_callback = update_callback
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Pending spike alert (popup message)
        self._pending_alert: Optional[dict] = None
        self._frame_count: int = 0
        
        # Stock messages for metric group spikes (B2B meeting context; overwritten by insight_generator when OpenAI succeeds)
        self._SPIKE_MESSAGES: dict = {
            "g1": "They're showing stronger interest—good moment to deepen the value proposition or ask for their view.",
            "g2": "They look like they're thinking hard—consider pausing or clarifying to avoid overload before your next ask.",
            "g3": "Signs of resistance or discomfort—try acknowledging concerns, addressing objections, or shifting approach.",
            "g4": "They appear ready to decide—offer a clear next step or ask for commitment.",
        }

        # Last frame for video-feed streaming (thread-safe)
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_lock = threading.Lock()

    def get_last_frame_jpeg(self) -> Optional[bytes]:
        """
        Return the most recent video frame as JPEG bytes for the engagement video feed.
        Thread-safe; returns None if no frame is available.
        """
        with self._last_frame_lock:
            if self._last_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self._last_frame)
            return buf.tobytes()

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
        for q in self._group_history.values():
            q.clear()
        for k in self._last_spike_time:
            self._last_spike_time[k] = 0.0
        with self.lock:
            self._pending_alert = None
            self.current_state = None
        
        if not self.video_handler.initialize_source(source_type, source_path, lightweight=self.lightweight_mode):
            print(f"Error: Failed to initialize video source type {source_type}")
            if source_path:
                print(f"  Source path: {source_path}")
            return False
        
        # Verify we can read frames; warm up webcam (first frames can be black/delayed)
        if source_type != VideoSourceType.PARTNER:
            warmup = 12
            for _ in range(warmup):
                r, _ = self.video_handler.read_frame()
                if not r:
                    time.sleep(0.04)
            ret, test_frame = self.video_handler.read_frame()
            if not ret or test_frame is None:
                print("Error: Video source initialized but cannot read frames")
                self.video_handler.release()
                return False
        
        print(f"Engagement detection started: source_type={source_type}, source_path={source_path}, method={self.detection_method}")
        
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
        if getattr(self, "_azure_detector_secondary", None):
            try:
                self._azure_detector_secondary.close()
            except Exception:
                pass
            self._azure_detector_secondary = None
    
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
                            self.video_handler.initialize_source(
                                self.video_handler.source_type,
                                self.video_handler.source_path,
                                lightweight=self.lightweight_mode,
                            )
                        self.consecutive_no_face_frames = 0
                    time.sleep(0.033)  # ~30 FPS
                    continue

                # Update last frame for video-feed streaming (thread-safe)
                with self._last_frame_lock:
                    self._last_frame = frame.copy()

                # Lightweight: process every 2nd frame to halve CPU
                if self.lightweight_mode:
                    self._frame_count += 1
                    if self._frame_count % 2 != 0:
                        time.sleep(0.016)
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
        # Skip invalid or too-small frames (detector needs usable resolution)
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        if frame.size == 0 or len(frame.shape) < 2:
            return None
        h, w = frame.shape[:2]
        if w < 64 or h < 64:
            return None

        self._frame_count += 1
        # Process every frame for real-time sensitivity (no frame skip)
        face_results = self.face_detector.detect_faces(frame)

        # Runtime fallback: Azure repeatedly returning no faces -> switch to MediaPipe (single-detector mode only)
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

        is_azure = self.face_detector.get_name() == "azure_face_api"
        is_unified = self.detection_method == "unified"
        azure_result = None
        signifier_scores = None

        if is_unified:
            # Unified: MediaPipe primary (landmarks + signifiers) + optional Azure (emotions); fuse scores
            if landmarks.shape[0] <= 27:
                bbox = getattr(face_result, "bounding_box", None)
                landmarks = expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame.shape)
            self.signifier_engine.update(landmarks, face_result, frame.shape)
            signifier_scores = self.signifier_engine.get_all_scores()
            mediapipe_score = self.signifier_engine.get_composite_score(signifier_scores)
            metrics = self._metrics_from_signifiers(signifier_scores)
            group_means = self.signifier_engine.get_group_means(signifier_scores)
            for k in ("g1", "g2", "g3", "g4"):
                self._group_history[k].append(group_means[k])
            self._check_spike_alerts(group_means, time.time())
            self._check_opportunity_alerts(group_means, signifier_scores, time.time())
            azure_score = None
            if getattr(self, "_azure_detector_secondary", None) and self._get_all_azure_metrics:
                try:
                    azure_face_results = self._azure_detector_secondary.detect_faces(frame)
                    if azure_face_results:
                        azure_result = self._get_all_azure_metrics(azure_face_results[0])
                        azure_score = azure_result["score"]
                        w_azure, w_mp = signifier_weights.get_fusion_weights()
                        score = float(np.clip(w_azure * azure_score + w_mp * mediapipe_score, 0.0, 100.0))
                        # Blend metrics for context: half Azure-derived, half signifier-derived
                        metrics_azure = self._metrics_from_azure(azure_result)
                        metrics = EngagementMetrics(
                            attention=(metrics.attention + metrics_azure.attention) / 2.0,
                            eye_contact=(metrics.eye_contact + metrics_azure.eye_contact) / 2.0,
                            facial_expressiveness=(metrics.facial_expressiveness + metrics_azure.facial_expressiveness) / 2.0,
                            head_movement=metrics.head_movement,
                            symmetry=metrics.symmetry,
                            mouth_activity=(metrics.mouth_activity + metrics_azure.mouth_activity) / 2.0,
                        )
                    else:
                        score = mediapipe_score
                except Exception:
                    score = mediapipe_score
            else:
                score = mediapipe_score
        elif is_azure and self._get_all_azure_metrics:
            azure_result = self._get_all_azure_metrics(face_result)
            score = azure_result["score"]
            metrics = self._metrics_from_azure(azure_result)
        else:
            # MediaPipe path: group means only (no composite); score = (g1+g4)/2 for context. No heavy extractor.
            if landmarks.shape[0] <= 27:
                bbox = getattr(face_result, "bounding_box", None)
                landmarks = expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame.shape)
            self.signifier_engine.update(landmarks, face_result, frame.shape)
            signifier_scores = self.signifier_engine.get_all_scores()
            group_means = self.signifier_engine.get_group_means(signifier_scores)
            g1, g4 = group_means["g1"], group_means["g4"]
            score = float((g1 + g4) / 2.0)
            metrics = self._metrics_from_signifiers(signifier_scores)
            for k in ("g1", "g2", "g3", "g4"):
                self._group_history[k].append(group_means[k])
            self._check_spike_alerts(group_means, time.time())
            self._check_opportunity_alerts(group_means, signifier_scores, time.time())

        if not np.isfinite(score) or score < 0 or score > 100:
            if is_azure and not is_unified:
                score = 0.0
            else:
                score = float(np.mean(list(signifier_scores.values()))) if signifier_scores else 0.0
        score = max(0.0, min(100.0, float(score)))
        
        self.score_history.append(score)
        self.metrics_history.append(metrics)
        raw_score = score
        self.consecutive_no_face_frames = 0
        
        # Determine engagement level from raw score
        level = EngagementLevel.from_score(raw_score)
        
        # Generate context for AI coaching from current-frame metrics (no smoothing)
        context = self.context_generator.generate_context(
            raw_score,
            metrics,
            level
        )
        
        # Confidence from recent score variance (no smoothing of score itself)
        confidence = self._calculate_confidence()
        
        return EngagementState(
            score=float(raw_score),
            level=level,
            metrics=metrics,
            context=context,
            timestamp=time.time(),
            face_detected=True,
            confidence=confidence,
            signifier_scores=signifier_scores,
            detection_method=self.detection_method,
            azure_metrics=azure_result if (is_azure or is_unified) else None,
        )
    
    def _metrics_from_signifiers(self, s: dict) -> EngagementMetrics:
        """Build EngagementMetrics from 30 signifier scores (lightweight path)."""
        def m(*keys: str) -> float:
            vals = [s.get(k, 0.0) for k in keys if k in s]
            return float(np.mean(vals)) if vals else 0.0
        return EngagementMetrics(
            attention=m("g1_duchenne", "g1_eye_contact", "g1_eyebrow_flash", "g1_pupil_dilation"),
            eye_contact=float(s.get("g1_eye_contact", 0.0)),
            facial_expressiveness=m("g1_duchenne", "g1_parted_lips", "g4_smile_transition", "g1_softened_forehead"),
            head_movement=m("g1_head_tilt", "g1_rhythmic_nodding", "g1_forward_lean"),
            symmetry=float(s.get("g1_facial_symmetry", 0.0)),
            mouth_activity=float(s.get("g1_parted_lips", 0.0)),
        )

    def _metrics_from_azure(self, azure_result: dict) -> EngagementMetrics:
        """Build EngagementMetrics from Azure emotion + composite metrics for context/coaching."""
        comp = azure_result.get("composite") or {}
        base = azure_result.get("base") or {}
        return EngagementMetrics(
            attention=float(comp.get("focused", 0.0)),
            eye_contact=float(comp.get("focused", 0.0)),
            facial_expressiveness=float(base.get("happiness", 0.0) * 0.5 + base.get("surprise", 0.0) * 0.5),
            head_movement=0.0,  # Azure emotions don't provide head movement
            symmetry=0.0,
            mouth_activity=float(base.get("happiness", 0.0)),
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
    
    def _check_spike_alerts(self, group_means: dict, now: float) -> None:
        """
        Detect sudden spike in any metric group (G1..G4) and set _pending_alert.
        These spikes represent "moments" in the meeting (e.g. aha, confusion, resistance,
        ready to close). The popup message is generated by Azure OpenAI in the API layer
        using this visual cue plus optional speech transcript (see services/insight_generator).
        Spike = current - min(recent) >= threshold and current >= min_value; cooldown per group.
        """
        with self.lock:
            if self._pending_alert is not None:
                return
            for gkey in ("g1", "g2", "g3", "g4"):
                hist = self._group_history[gkey]
                if len(hist) < 6:
                    continue
                cur = group_means.get(gkey, 0.0)
                if cur < self._spike_min_value:
                    continue
                recent_min = float(min(hist))
                delta = cur - recent_min
                if delta >= self._spike_delta_threshold and (now - self._last_spike_time[gkey]) >= self._spike_cooldown_sec:
                    self._pending_alert = {
                        "type": "spike",
                        "group": gkey,
                        "message": self._SPIKE_MESSAGES.get(gkey, "Notable change in this metric—consider adjusting your approach."),
                    }
                    self._last_spike_time[gkey] = now
                    return

    def _check_opportunity_alerts(
        self, group_means: dict, signifier_scores: Optional[dict], now: float
    ) -> None:
        """
        Detect B2B opportunity features (e.g. closing window, cognitive overload, skepticism).
        If none fired, do nothing. If one fires and no spike alert is pending, set _pending_alert
        so the API can generate an insight via Azure OpenAI (see services/insight_generator).
        """
        with self.lock:
            if self._pending_alert is not None:
                return
            result = detect_opportunity(
                group_means=group_means,
                group_history=self._group_history,
                signifier_scores=signifier_scores,
                now=now,
            )
            if result is not None:
                oid, context = result
                self._pending_alert = {
                    "type": "opportunity",
                    "opportunity_id": oid,
                    "context": context,
                    "message": "Opportunity detected—consider acting on it.",
                }
    
    def get_and_clear_pending_alert(self) -> Optional[dict]:
        """Thread-safe get and clear of pending alert. Returns None if none."""
        with self.lock:
            a = self._pending_alert
            self._pending_alert = None
            return a

    def get_score_breakdown(self) -> Optional[dict]:
        """
        Return a step-by-step breakdown of how the current engagement score was calculated.
        Used for frontend "How is the score calculated?" transparency.
        Returns None if detection not running or no state; otherwise dict with detectionMethod,
        breakdown (Azure or MediaPipe), score, level, levelBands.
        """
        with self.lock:
            state = self.current_state
        if not state:
            return None
        level_bands = [
            {"level": "VERY_LOW", "range": [0, 25]},
            {"level": "LOW", "range": [25, 45]},
            {"level": "MEDIUM", "range": [45, 70]},
            {"level": "HIGH", "range": [70, 85]},
            {"level": "VERY_HIGH", "range": [85, 100]},
        ]
        base_response = {
            "detectionMethod": state.detection_method or "mediapipe",
            "score": float(state.score),
            "level": state.level.name if state.level else "UNKNOWN",
            "levelBands": level_bands,
            "faceDetected": bool(state.face_detected),
        }
        if not state.face_detected:
            base_response["breakdown"] = {"message": "No face detected; score is 0."}
            return base_response
        if state.detection_method == "unified" and (state.signifier_scores or state.azure_metrics):
            # Fused breakdown: MediaPipe + optional Azure + fusion weights
            w_azure, w_mp = signifier_weights.get_fusion_weights()
            breakdown = {
                "type": "unified",
                "fusionWeights": {"azure": w_azure, "mediapipe": w_mp},
                "fusedScore": float(state.score),
                "formula": "fusedScore = azure_weight × azure_score + mediapipe_weight × mediapipe_score",
            }
            if state.signifier_scores and self.signifier_engine:
                breakdown["mediapipe"] = self.signifier_engine.get_composite_breakdown(state.signifier_scores)
            if state.azure_metrics and self._get_azure_score_breakdown:
                base = state.azure_metrics.get("base") or {}
                composite = state.azure_metrics.get("composite") or {}
                breakdown["azure"] = self._get_azure_score_breakdown(
                    None, base_metrics=base, composite_metrics=composite
                )
            base_response["breakdown"] = breakdown
            return base_response
        if state.azure_metrics and self._get_azure_score_breakdown:
            base = state.azure_metrics.get("base") or {}
            composite = state.azure_metrics.get("composite") or {}
            breakdown = self._get_azure_score_breakdown(None, base_metrics=base, composite_metrics=composite)
            base_response["breakdown"] = breakdown
            return base_response
        if state.signifier_scores and self.signifier_engine:
            breakdown = self.signifier_engine.get_composite_breakdown(state.signifier_scores)
            base_response["breakdown"] = breakdown
            return base_response
        base_response["breakdown"] = {"message": "Breakdown not available for current state."}
        return base_response

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
