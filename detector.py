"""
=============================================================================
ENGAGEMENT STATE DETECTOR (detector.py)
=============================================================================

WHAT THIS FILE DOES (in plain language):
----------------------------------------
This is the core of "engagement detection." It takes VIDEO (and optionally
transcript + voice data from services) and turns it into:

  - A single ENGAGEMENT SCORE (0–100) and a level (e.g. LOW, HIGH).
  - Forty-two SIGNIFIER SCORES (e.g. eye contact, smile, gaze aversion) that
    describe *how* the person appears (interested, thinking, resistant, etc.).
  - COMPOSITE METRICS that combine face + speech + voice (e.g. "decision
    readiness," "confusion").
  - ALERTS when something notable happens (e.g. a sudden spike in "interest"
    or a B2B "opportunity" like "closing window" or "confusion").

The frontend sends video frames (or we read from webcam/file/partner). We run
face detection (MediaPipe and/or Azure), compute signifiers and composites,
then update the "current state." When the frontend calls GET /engagement/state,
the route reads this state and (if there is an alert) asks the insight
generator for short coaching text.

PIPELINE (simplified):
  1. Get one frame from the video source.
  2. Detect face(s) (MediaPipe and/or Azure).
  3. From face landmarks (and Azure emotions if used), compute 42 signifiers.
  4. Average signifiers into 4 groups (G1–G4): interest, cognitive load,
     resistance, decision-ready.
  5. Compute overall score and composite metrics (using speech/voice tags from services).
  6. Check for spikes and B2B opportunities; if one fires and cooldown allows,
     set a "pending alert." The route will later turn that into popup text.

RESEARCH BASIS:
---------------
Engagement bands and when to intervene vs. when to seek commitment are based
on virtual/sales meeting research (Cialdini, Edmondson, etc.). Low engagement
→ re-engage (simplify, check understanding). High engagement → capitalize
(proposals, commitments). Composites and acoustic tags (Scherer, Bachorowski,
Ladd) refine state for decision readiness, confusion, resistance.

WEIGHTS:
--------
Signifier and fusion weights are fixed in code (helpers.py); there is no
external config file. Alerts are turned into text by the insight generator
in services.py.
=============================================================================
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
from helpers import (
    VideoSourceHandler,
    VideoSourceType,
    EngagementScorer,
    EngagementMetrics,
    ContextGenerator,
    EngagementContext,
    FaceDetectorInterface,
    FaceDetectionResult,
    expand_azure_landmarks_to_mediapipe,
    ExpressionSignifierEngine,
    get_weights,
    get_fusion_weights,
    evaluate_capability,
    detect_opportunity,
    NEGATIVE_OPPORTUNITY_IDS,
    POSITIVE_OPPORTUNITY_IDS,
    compute_composite_metrics,
    get_active_metrics_with_config,
    MetricConfig,
)
from services import get_recent_speech_tags, get_recent_acoustic_tags_and_negative_strength, is_idle

# --- Azure engagement metrics (inlined) ---
_AZURE_EMOTION_KEYS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]
_AZURE_COMPOSITE_DEFINITIONS = {
    "receptive": {"happiness": 0.50, "contempt": -0.28, "anger": -0.28},
    "focused": {"neutral": 0.52, "fear": -0.24, "disgust": -0.22},
    "interested": {"surprise": 0.48, "happiness": 0.48},
    "agreeable": {"happiness": 0.52, "contempt": -0.26, "disgust": -0.24},
    "open": {"happiness": 0.42, "surprise": 0.42, "contempt": -0.22},
    "skeptical": {"contempt": 0.58, "happiness": -0.42},
    "concerned": {"fear": 0.48, "sadness": 0.48},
    "disagreeing": {"contempt": 0.48, "anger": 0.48},
    "stressed": {"fear": 0.48, "anger": 0.48},
    "disengaged": {"neutral": 0.55, "happiness": -0.24, "surprise": -0.24},
}
_AZURE_POSITIVE_COMPOSITES = ["receptive", "focused", "interested", "agreeable", "open"]
_AZURE_NEGATIVE_COMPOSITES = ["skeptical", "concerned", "disagreeing", "stressed", "disengaged"]


def _azure_get_emotions(face_result: Optional[FaceDetectionResult]) -> dict:
    """
    Extract the eight emotion values (anger, contempt, disgust, fear, happiness,
    neutral, sadness, surprise) from an Azure face result. Returns a dict with
    keys from _AZURE_EMOTION_KEYS and values in 0.0–1.0. If no face or no
    emotions, returns all zeros.
    """
    out = {k: 0.0 for k in _AZURE_EMOTION_KEYS}
    if not face_result or not getattr(face_result, "emotions", None) or not isinstance(face_result.emotions, dict):
        return out
    for k in _AZURE_EMOTION_KEYS:
        v = face_result.emotions.get(k)
        if v is not None and isinstance(v, (int, float)):
            out[k] = float(np.clip(v, 0.0, 1.0))
    return out


def _azure_compute_base_metrics(face_result: Optional[FaceDetectionResult]) -> dict:
    """
    Turn Azure emotions (0–1) into "base" metrics (0–100) for the API. Each
    emotion becomes one metric (e.g. anger → 0–100). Used when we use Azure
    Face for detection so the frontend can show the same emotion bars.
    """
    emotions = _azure_get_emotions(face_result)
    return {k: round(float(emotions[k]) * 100.0, 1) for k in _AZURE_EMOTION_KEYS}


def _azure_score_composite(emotions: dict, weights: dict) -> float:
    """
    Score one "composite" (e.g. receptive, skeptical) from emotion values and
    predefined weights. Result is 0–100. Composites combine emotions (e.g.
    receptive = high happiness, low contempt/anger).
    """
    raw = sum(emotions.get(k, 0.0) * w for k, w in weights.items())
    return float(np.clip(raw * 200.0, 0.0, 100.0))


def _azure_compute_composite_metrics(face_result: Optional[FaceDetectionResult]) -> dict:
    emotions = _azure_get_emotions(face_result)
    return {name: round(_azure_score_composite(emotions, w), 1) for name, w in _AZURE_COMPOSITE_DEFINITIONS.items()}


def compute_azure_engagement_score(
    face_result: Optional[FaceDetectionResult],
    base_metrics: Optional[dict] = None,
    composite_metrics: Optional[dict] = None,
) -> float:
    """
    Compute a single 0–100 engagement score from Azure face result. We average
    "positive" composites (receptive, focused, interested, etc.) and "negative"
    ones (skeptical, stressed, disengaged), then score = positive − negative
    with small adjustments when either side is very high. Used when detection
    method is Azure or unified.
    """
    if base_metrics is None:
        base_metrics = _azure_compute_base_metrics(face_result)
    if composite_metrics is None:
        composite_metrics = _azure_compute_composite_metrics(face_result)
    pos_sum = sum(composite_metrics.get(k, 0.0) for k in _AZURE_POSITIVE_COMPOSITES) / len(_AZURE_POSITIVE_COMPOSITES)
    neg_sum = sum(composite_metrics.get(k, 0.0) for k in _AZURE_NEGATIVE_COMPOSITES) / len(_AZURE_NEGATIVE_COMPOSITES)
    raw = pos_sum - neg_sum
    if pos_sum > 70.0:
        raw = min(100.0, raw + 8.0)
    if neg_sum > 70.0:
        raw = max(0.0, raw - 12.0)
    return float(np.clip(raw, 0.0, 100.0))


def get_azure_score_breakdown(
    face_result: Optional[FaceDetectionResult],
    base_metrics: Optional[dict] = None,
    composite_metrics: Optional[dict] = None,
) -> dict:
    if base_metrics is None:
        base_metrics = _azure_compute_base_metrics(face_result)
    if composite_metrics is None:
        composite_metrics = _azure_compute_composite_metrics(face_result)
    pos_avg = sum(composite_metrics.get(k, 0.0) for k in _AZURE_POSITIVE_COMPOSITES) / len(_AZURE_POSITIVE_COMPOSITES)
    neg_avg = sum(composite_metrics.get(k, 0.0) for k in _AZURE_NEGATIVE_COMPOSITES) / len(_AZURE_NEGATIVE_COMPOSITES)
    raw = pos_avg - neg_avg
    adjustments = []
    if pos_avg > 70.0:
        raw = min(100.0, raw + 8.0)
        adjustments.append("pos_avg > 70: +8")
    if neg_avg > 70.0:
        raw = max(0.0, raw - 12.0)
        adjustments.append("neg_avg > 70: -12")
    score = float(np.clip(raw, 0.0, 100.0))
    return {
        "formula": "score = clip(pos_avg - neg_avg + adjustments, 0, 100)",
        "base": base_metrics,
        "composite": composite_metrics,
        "positiveComposites": _AZURE_POSITIVE_COMPOSITES,
        "negativeComposites": _AZURE_NEGATIVE_COMPOSITES,
        "posAvg": round(pos_avg, 2),
        "negAvg": round(neg_avg, 2),
        "rawBeforeAdjustments": round(pos_avg - neg_avg, 2),
        "adjustments": adjustments or ["none"],
        "score": round(score, 1),
    }


def get_all_azure_metrics(face_result: Optional[FaceDetectionResult]) -> dict:
    base = _azure_compute_base_metrics(face_result)
    composite = _azure_compute_composite_metrics(face_result)
    score = compute_azure_engagement_score(face_result, base_metrics=base, composite_metrics=composite)
    return {"base": base, "composite": composite, "score": round(score, 1)}


class EngagementLevel(Enum):
    """
    Engagement levels from score (0-100), aligned with state-of-the-art meeting psychology.
    Bands are set so coaching advice maps to evidence-based actions: re-engage (VERY_LOW/LOW),
    build (MEDIUM), capitalize (HIGH/VERY_HIGH). Boundaries reflect when-to-intervene vs.
    when-to-seek-commitment research (Cialdini; Edmondson; virtual/sales engagement studies).
    """
    VERY_LOW = "VERY_LOW"    # 0-25: immediate re-engagement (receptivity very low)
    LOW = "LOW"              # 25-45: re-engage, simplify, check understanding
    MEDIUM = "MEDIUM"        # 45-70: build interest, address concerns, invite participation
    HIGH = "HIGH"            # 70-85: capitalize—present proposals, seek commitments
    VERY_HIGH = "VERY_HIGH"  # 85-100: peak receptivity—advance to next steps, close
    
    @classmethod
    def from_score(cls, score: float, prev_level: Optional['EngagementLevel'] = None) -> 'EngagementLevel':
        """
        Map engagement score to level for coaching advice.
        Boundaries from meeting-psychology research. Optional prev_level enables
        hysteresis: downgrade only when score drops 2 pts below boundary (reduces flicker).
        """
        _HYSTERESIS = 2.0
        def _level_for(s: float) -> 'EngagementLevel':
            if s >= 85:
                return cls.VERY_HIGH
            elif s >= 70:
                return cls.HIGH
            elif s >= 45:
                return cls.MEDIUM
            elif s >= 25:
                return cls.LOW
            else:
                return cls.VERY_LOW

        level = _level_for(score)
        if prev_level is None:
            return level
        # Hysteresis: stay at higher level until score drops below boundary
        order = [cls.VERY_LOW, cls.LOW, cls.MEDIUM, cls.HIGH, cls.VERY_HIGH]
        prev_idx = order.index(prev_level)
        new_idx = order.index(level)
        if new_idx < prev_idx:  # Downgrading
            thresholds = [0, 25, 45, 70, 85]
            hold_threshold = thresholds[prev_idx] - _HYSTERESIS
            if score >= hold_threshold:
                return prev_level  # Hold current level
        return level


@dataclass
class EngagementState:
    """
    A snapshot of the current engagement state at one moment in time.

    The detector updates this every frame (or every few frames). The route
    GET /engagement/state reads it and sends it to the frontend. Fields:
      score — Single number 0–100 (overall engagement).
      level — Category: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH (for coaching).
      metrics — Six basic metrics (attention, eye_contact, etc.).
      context — Human-readable summary and suggested actions (for AI and UI).
      signifier_scores — All 42 signifier values (0–100 each).
      azure_metrics — When using Azure: base emotions and B2B composites.
      composite_metrics — Multimodal composites (face + speech + voice).
    """
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
    composite_metrics: Optional[dict] = None  # Facial+speech composites (0-100 each); see utils/engagement_composites


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
        min_face_confidence: float = 0.2,
        update_callback: Optional[Callable[[EngagementState], None]] = None,
        detection_method: Optional[str] = None,
        lightweight_mode: bool = False,
    ):
        """
        Initialize the engagement state detector.
        
        Args:
            min_face_confidence: Minimum confidence for face detection (default: 0.2; lower helps in suboptimal lighting)
            update_callback: Optional callback function called when state updates
            detection_method: Face detection method to use ("mediapipe" or "azure_face_api").
                             If None, uses config.FACE_DETECTION_METHOD
            lightweight_mode: If True, use MediaPipe only, smaller buffer, process every 2nd
                             frame, and skip 100-feature extractor (for low-power devices).
        """
        self.lightweight_mode = bool(lightweight_mode)
        default_conf = config.MIN_FACE_CONFIDENCE
        self._min_face_confidence = max(0.01, min(0.9, float(min_face_confidence or default_conf)))
        if self.lightweight_mode:
            detection_method = "mediapipe"

        if detection_method is None:
            detection_method = config.FACE_DETECTION_METHOD.lower() or "mediapipe"

        # Resolve "auto" to mediapipe or azure_face_api based on device + network
        self._effective_method: Optional[str] = None
        if (detection_method or "").lower() == "auto" and config.AUTO_DETECTION_SWITCHING:
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
            from helpers import AzureFaceAPIDetector, MediaPipeFaceDetector
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
            from helpers import AzureFaceAPIDetector
            self._get_all_azure_metrics = get_all_azure_metrics
            self._get_azure_score_breakdown = get_azure_score_breakdown
            try:
                self.face_detector = AzureFaceAPIDetector()
                if not self.face_detector.is_available():
                    print("Warning: Azure Face API not available, falling back to MediaPipe")
                    from helpers import MediaPipeFaceDetector
                    self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            except Exception as e:
                print(f"Warning: Failed to initialize Azure Face API: {e}. Falling back to MediaPipe")
                from helpers import MediaPipeFaceDetector
                self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            self.detection_method = self.face_detector.get_name()
        else:
            from helpers import MediaPipeFaceDetector
            self.face_detector = MediaPipeFaceDetector(min_detection_confidence=self._min_face_confidence)
            self.detection_method = self.face_detector.get_name()
        
        # Metric config for dynamic selection (tiers: high/medium/low)
        try:
            self._metric_config: Optional[MetricConfig] = get_active_metrics_with_config()
        except Exception:
            self._metric_config = None  # Fallback: no filtering

        # Component initialization
        self.video_handler = VideoSourceHandler()
        self.scorer = EngagementScorer()
        self.context_generator = ContextGenerator()
        buf = 12 if self.lightweight_mode else 22
        self.signifier_engine = ExpressionSignifierEngine(
            buffer_frames=buf,
            weights_provider=get_weights,
        )

        # State management (minimal history for confidence only)
        self.current_state: Optional[EngagementState] = None
        self.score_history: deque = deque(maxlen=3)
        self.metrics_history: deque = deque(maxlen=3)
        
        # Track consecutive frames without face (FPS-aware: ~1s at 60 fps, ~2s at 30 fps)
        self.consecutive_no_face_frames = 0
        self._target_fps_min = max(15, float(config.TARGET_FPS_MIN))
        self._target_fps_max = max(30, float(config.TARGET_FPS_MAX))
        self.max_no_face_frames = int(self._target_fps_max)
        
        # Spike detection per group (G1..G4): short window, cooldown per group
        self._group_history: dict = {k: deque(maxlen=12) for k in ("g1", "g2", "g3", "g4")}
        self._last_spike_time: dict = {k: 0.0 for k in ("g1", "g2", "g3", "g4")}
        self._spike_cooldown_sec: float = 45.0
        self._spike_delta_threshold: float = 22.0  # sudden rise above recent min
        self._spike_min_value: float = 42.0  # only fire if current group mean >= this
        self._enable_single_spike_alerts: bool = False  # Disabled: use combinations instead
        
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
        self._last_composite_100_time: float = 0.0
        self._composite_100_cooldown_sec: float = 22.0  # Positive-only; reduce frequency
        self._last_insight_popup_time: float = 0.0
        # Retention: shorter buffer for negative (losing focus, confused, distracted); longer for positive
        self._insight_buffer_sec_negative: float = float(config.INSIGHT_BUFFER_SEC_NEGATIVE)
        self._insight_buffer_sec_positive: float = float(config.INSIGHT_BUFFER_SEC_POSITIVE)
        self._min_concurrent_features_negative: int = int(config.MIN_CONCURRENT_FEATURES_NEGATIVE)
        self._min_concurrent_features_positive: int = int(config.MIN_CONCURRENT_FEATURES_POSITIVE)
        
        # Stock messages for metric group spikes (B2B meeting context; overwritten by insight_generator when Azure AI Foundry succeeds)
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
        Thread-safe; copies the frame only when requested (copy-on-read) to avoid copying every frame in the loop.
        Uses JPEG quality 85 to reduce payload size and encode time while keeping visual quality.
        """
        with self._last_frame_lock:
            if self._last_frame is None:
                return None
            frame_to_encode = self._last_frame.copy()
        _, buf = cv2.imencode(".jpg", frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        self._last_composite_100_time = 0.0
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
        """
        Stop the detection loop and release the video source and face detector.

        Sets is_running to False so the loop exits, waits for the thread to
        finish (up to 2 seconds), then releases the video handler and closes
        the face detector(s). Called when the user stops engagement or when
        switching source.
        """
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
        Get the latest engagement state (thread-safe).

        The detection loop updates current_state every frame. This method
        returns a snapshot so the route (GET /engagement/state) can send it
        to the frontend. Returns None if no state has been computed yet
        (e.g. right after start, or when no face has been detected).
        """
        with self.lock:
            return self.current_state
    
    def get_fps(self) -> float:
        """
        Get the current processing FPS (target 30–60).
        
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
                    if self.consecutive_no_face_frames > int(self._target_fps_max * 2):  # ~2s regardless of FPS
                        print("Warning: Video source not providing frames, checking connection...")
                        # Try to reinitialize if possible
                        if self.video_handler.source_type == VideoSourceType.WEBCAM:
                            self.video_handler.initialize_source(
                                self.video_handler.source_type,
                                self.video_handler.source_path,
                                lightweight=self.lightweight_mode,
                            )
                        self.consecutive_no_face_frames = 0
                    time.sleep(1.0 / self._target_fps_min)
                    continue

                with self._last_frame_lock:
                    self._last_frame = frame

                current_fps = self.get_fps()
                self.signifier_engine.set_fps(
                    min(self._target_fps_max, max(self._target_fps_min, current_fps)) if current_fps > 0 else self._target_fps_min
                )

                if self.lightweight_mode and current_fps >= self._target_fps_min:
                    self._frame_count += 1
                    if self._frame_count % 2 != 0:
                        time.sleep(1.0 / self._target_fps_max)
                        continue

                if is_idle(60.0):
                    self._frame_count += 1
                    if self._frame_count % 4 != 0:
                        time.sleep(1.0 / self._target_fps_min)
                        continue

                # No adaptive frame skip: process every frame for real-time tracking at 30 fps

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
                
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time
                if frame_time > 0:
                    self.fps_counter.append(1.0 / frame_time)
                frame_budget = 1.0 / self._target_fps_max
                if frame_time < frame_budget and frame_time > 0:
                    time.sleep(frame_budget - frame_time)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[EngagementState]:
        """
        Process one video frame: detect face(s), compute signifiers and score,
        check for spikes and B2B opportunities, return the new state.

        Steps: run face detection → get landmarks (and Azure emotions if used)
        → run signifier engine (42 scores) → compute group means and overall
        score → build context → run composite metrics and opportunity detector
        → if an alert fires and cooldown allows, set pending alert → return
        EngagementState. If no face is detected, returns a "no face" state
        (low score, no signifiers) so the UI still gets a response.
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
                    from helpers import MediaPipeFaceDetector
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
        composite_metrics = None
        acoustic_tags: List[str] = []

        if is_unified:
            # Unified: MediaPipe primary (landmarks + signifiers) + optional Azure (emotions); fuse scores
            if landmarks.shape[0] <= 27:
                bbox = getattr(face_result, "bounding_box", None)
                landmarks = expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame.shape)
            self.signifier_engine.update(landmarks, face_result, frame.shape)
            signifier_scores = self.signifier_engine.get_all_scores()
            weights = self.signifier_engine._get_weights()
            mediapipe_score = self.signifier_engine.get_composite_score(signifier_scores, weights)
            metrics = self._metrics_from_signifiers(signifier_scores, landmarks=landmarks, frame_shape=frame.shape)
            group_means = self.signifier_engine.get_group_means(signifier_scores, weights)
            for k in ("g1", "g2", "g3", "g4"):
                self._group_history[k].append(group_means[k])
            now = time.time()
            speech_tags = get_recent_speech_tags(12)
            acoustic_tags, acoustic_neg = get_recent_acoustic_tags_and_negative_strength()
            self._check_composite_at_100(
                group_means, mediapipe_score, now,
                signifier_scores=signifier_scores,
                has_recent_speech=len(speech_tags) > 0,
            )
            self._check_spike_alerts(group_means, now)
            acoustic_tags_filtered = [t for t in acoustic_tags if not self._metric_config or t in self._metric_config.acoustic_tags]
            composite_metrics_raw = compute_composite_metrics(
                group_means, signifier_scores, speech_tags,
                acoustic_tags=acoustic_tags_filtered,
                acoustic_negative_strength=acoustic_neg,
            )
            composite_keys = self._metric_config.composite_keys if self._metric_config else None
            composite_metrics = {k: v for k, v in composite_metrics_raw.items() if not composite_keys or k in composite_keys}
            self._check_opportunity_alerts(
                group_means, signifier_scores, now,
                composite_score=mediapipe_score,
                composite_metrics=composite_metrics,
                acoustic_tags=acoustic_tags,
            )
            azure_score = None
            if getattr(self, "_azure_detector_secondary", None) and self._get_all_azure_metrics:
                try:
                    azure_face_results = self._azure_detector_secondary.detect_faces(frame)
                    if azure_face_results:
                        azure_result = self._get_all_azure_metrics(azure_face_results[0])
                        azure_score = azure_result["score"]
                        w_azure, w_mp = get_fusion_weights()
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
            weights = self.signifier_engine._get_weights()
            group_means = self.signifier_engine.get_group_means(signifier_scores, weights)
            composite_score = self.signifier_engine.get_composite_score(signifier_scores, weights)
            g1, g4 = group_means["g1"], group_means["g4"]
            score = float((g1 + g4) / 2.0)
            metrics = self._metrics_from_signifiers(signifier_scores, landmarks=landmarks, frame_shape=frame.shape)
            for k in ("g1", "g2", "g3", "g4"):
                self._group_history[k].append(group_means[k])
            now = time.time()
            speech_tags = get_recent_speech_tags(12)
            acoustic_tags, acoustic_neg = get_recent_acoustic_tags_and_negative_strength()
            self._check_composite_at_100(
                group_means, composite_score, now,
                signifier_scores=signifier_scores,
                has_recent_speech=len(speech_tags) > 0,
            )
            self._check_spike_alerts(group_means, now)
            acoustic_tags_filtered = [t for t in acoustic_tags if not self._metric_config or t in self._metric_config.acoustic_tags]
            composite_metrics_raw = compute_composite_metrics(
                group_means, signifier_scores, speech_tags,
                acoustic_tags=acoustic_tags_filtered,
                acoustic_negative_strength=acoustic_neg,
            )
            composite_keys = self._metric_config.composite_keys if self._metric_config else None
            composite_metrics = {k: v for k, v in composite_metrics_raw.items() if not composite_keys or k in composite_keys}
            self._check_opportunity_alerts(
                group_means, signifier_scores, now,
                composite_score=composite_score,
                composite_metrics=composite_metrics,
                acoustic_tags=acoustic_tags,
            )

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
        
        # Determine engagement level from raw score (with hysteresis to reduce flicker)
        prev_level = self.current_state.level if self.current_state else None
        level = EngagementLevel.from_score(raw_score, prev_level)
        
        # Generate context for AI coaching from current-frame metrics (no smoothing)
        context = self.context_generator.generate_context(
            raw_score,
            metrics,
            level,
            composite_metrics=composite_metrics,
            acoustic_tags=acoustic_tags,
        )
        
        # Confidence from recent score variance (no smoothing of score itself)
        confidence = self._calculate_confidence()
        
        # Filter signifier_scores to active metric config (dynamic selection)
        signifier_scores_filtered = None
        if signifier_scores and self._metric_config:
            active_keys = set(self._metric_config.signifier_keys)
            signifier_scores_filtered = {k: v for k, v in signifier_scores.items() if k in active_keys}

        return EngagementState(
            score=float(raw_score),
            level=level,
            metrics=metrics,
            context=context,
            timestamp=time.time(),
            face_detected=True,
            confidence=confidence,
            signifier_scores=signifier_scores_filtered or signifier_scores,
            detection_method=self.detection_method,
            azure_metrics=azure_result if (is_azure or is_unified) else None,
            composite_metrics=composite_metrics,
        )
    
    def _metrics_from_signifiers(
        self,
        s: dict,
        landmarks: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int, int]] = None,
    ) -> EngagementMetrics:
        """Build EngagementMetrics from signifier scores (lightweight path). See README.md §16.
        When landmarks and frame_shape are provided, blend attention and eye_contact with scorer (EAR-based).
        """
        def m(*keys: str) -> float:
            vals = [s.get(k, 0.0) for k in keys if k in s]
            return float(np.mean(vals)) if vals else 0.0
        att_signer = m("g1_duchenne", "g1_eye_contact", "g1_eyebrow_flash", "g1_pupil_dilation")
        eye_signer = float(s.get("g1_eye_contact", 0.0))
        # Hybrid: blend with scorer EAR-based attention and eye_contact when landmarks available
        if landmarks is not None and frame_shape is not None and landmarks.size > 0:
            try:
                att_scorer, eye_scorer = self.scorer.compute_attention_eye_contact_from_landmarks(
                    landmarks, frame_shape
                )
                att_signer = 0.5 * att_signer + 0.5 * att_scorer
                eye_signer = 0.5 * eye_signer + 0.5 * eye_scorer
            except Exception:
                pass  # Keep signifier-only on error
        # Facial expressiveness: include varied expression signals so different expressions (smile, brow, mouth, etc.)
        # can push the score high. Use mean + max blend so one strong expression raises the metric above medium.
        expr_keys = (
            "g1_duchenne", "g1_parted_lips", "g4_smile_transition", "g1_softened_forehead",
            "g1_eyebrow_flash", "g1_micro_smile", "g1_eye_widening", "g1_brow_raise_sustained",
        )
        expr_vals = [float(s.get(k, 0.0)) for k in expr_keys if k in s]
        if expr_vals:
            expr_mean = float(np.mean(expr_vals))
            expr_max = float(np.max(expr_vals))
            facial_expressiveness = 0.5 * expr_mean + 0.5 * expr_max  # strong on any dimension → high score
        else:
            facial_expressiveness = 0.0
        # Head movement: tilt, nod, lean + general motion (inverse of stillness) so rapid/noticeable
        # movement can reach high levels. Use mean+max so any strong component raises the score.
        head_keys = ("g1_head_tilt", "g1_rhythmic_nodding", "g1_forward_lean")
        head_vals = [float(s.get(k, 0.0)) for k in head_keys if k in s]
        stillness = float(s.get("g2_stillness", 50.0))
        movement_amount = 100.0 - stillness  # high when face is moving (low stillness)
        head_vals.append(movement_amount)
        if head_vals:
            head_mean = float(np.mean(head_vals))
            head_max = float(np.max(head_vals))
            head_movement = 0.5 * head_mean + 0.5 * head_max
        else:
            head_movement = 0.0
        return EngagementMetrics(
            attention=att_signer,
            eye_contact=eye_signer,
            facial_expressiveness=min(100.0, facial_expressiveness),
            head_movement=min(100.0, head_movement),
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

    def _count_concurrent_features(
        self,
        group_means: dict,
        composite_score: float,
        signifier_scores: Optional[dict],
        has_recent_speech: bool,
        include_speech: bool = True,
        composite_metrics: Optional[dict] = None,
    ) -> int:
        """
        Count how many "concurrent features" are active. Used to require multiple
        features before showing a positive insight popup.
        """
        count = 0
        for gkey in ("g1", "g2", "g3", "g4"):
            if group_means.get(gkey, 0.0) >= 55.0:
                count += 1
        if composite_score >= 60.0:
            count += 1
        if signifier_scores:
            detected = sum(1 for v in signifier_scores.values() if float(v) >= 65.0)
            if detected >= 2:
                count += 1
        if include_speech and has_recent_speech:
            count += 1
        if composite_metrics:
            if composite_metrics.get("decision_readiness_multimodal", 0) >= 60:
                count += 1
            elif composite_metrics.get("verbal_nonverbal_alignment", 0) >= 60:
                count += 1
        return count

    def _check_composite_at_100(
        self,
        group_means: dict,
        composite_score: float,
        now: float,
        signifier_scores: Optional[dict] = None,
        has_recent_speech: Optional[bool] = None,
    ) -> None:
        """
        When composite metrics reach very high engagement (g1/g4/composite >= 92), trigger
        popup only if at least _min_concurrent_features_positive are active (reduce frequency).
        """
        if has_recent_speech is None:
            has_recent_speech = len(get_recent_speech_tags(12)) > 0
        with self.lock:
            if self._pending_alert is not None:
                return
            if (now - self._last_composite_100_time) < self._composite_100_cooldown_sec:
                return
            g1 = group_means.get("g1", 0.0)
            g4 = group_means.get("g4", 0.0)
            triggered = False
            group_hit = ""
            if g1 >= 92.0:
                triggered = True
                group_hit = "g1"
            elif g4 >= 92.0:
                triggered = True
                group_hit = "g4"
            elif composite_score >= 92.0:
                triggered = True
                group_hit = "composite"
            if triggered:
                n = self._count_concurrent_features(
                    group_means, composite_score, signifier_scores, has_recent_speech
                )
                if n < self._min_concurrent_features_positive:
                    return
                self._last_composite_100_time = now
                self._pending_alert = {
                    "type": "composite_100",
                    "group": group_hit,
                    "message": self._SPIKE_MESSAGES.get(
                        group_hit if group_hit in ("g1", "g4") else "g1",
                        "Strong engagement signal—consider acting on it.",
                    ),
                }

    def _check_spike_alerts(self, group_means: dict, now: float) -> None:
        """
        Detect sudden spike in any metric group (G1..G4) and set _pending_alert.
        DISABLED by default (_enable_single_spike_alerts=False): use combination-based
        detection instead for more psychologically meaningful insights.
        """
        if not self._enable_single_spike_alerts:
            return  # Skip single-metric spikes; use combinations
        
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
        self,
        group_means: dict,
        signifier_scores: Optional[dict],
        now: float,
        composite_score: float = 0.0,
        composite_metrics: Optional[dict] = None,
        acoustic_tags: Optional[List[str]] = None,
    ) -> None:
        """
        Detect B2B opportunity features. Negative require 1 concurrent feature;
        positive require 4 (multiple positive signals only).
        """
        speech_tags = get_recent_speech_tags(12)
        has_recent_speech = len(speech_tags) > 0
        with self.lock:
            if self._pending_alert is not None:
                return
            result = detect_opportunity(
                group_means=group_means,
                group_history=self._group_history,
                signifier_scores=signifier_scores,
                now=now,
                recent_speech_tags=speech_tags,
                composite_metrics=composite_metrics,
                acoustic_tags=acoustic_tags,
            )
            if result is not None:
                oid, context = result
                min_required = (
                    self._min_concurrent_features_negative
                    if oid in NEGATIVE_OPPORTUNITY_IDS
                    else self._min_concurrent_features_positive
                )
                n = self._count_concurrent_features(
                    group_means, composite_score, signifier_scores, has_recent_speech,
                    composite_metrics=composite_metrics,
                )
                if n < min_required:
                    return
                self._pending_alert = {
                    "type": "opportunity",
                    "opportunity_id": oid,
                    "context": context,
                    "message": "Opportunity detected—consider acting on it.",
                }
    
    def get_pending_alert(self) -> Optional[dict]:
        """Peek at pending alert without clearing. Returns None if none."""
        with self.lock:
            return self._pending_alert
    
    def clear_pending_alert(self) -> None:
        """Clear pending alert."""
        with self.lock:
            self._pending_alert = None
    
    def can_show_insight(self, alert: Optional[dict] = None) -> bool:
        """
        True if enough time has passed since last insight popup.
        Uses shorter buffer for negative alerts (retention), longer for positive.
        If alert is None, uses the more permissive (negative) buffer.
        """
        elapsed = time.time() - self._last_insight_popup_time
        is_positive = self._is_positive_alert(alert)
        buffer = self._insight_buffer_sec_positive if is_positive else self._insight_buffer_sec_negative
        return elapsed >= buffer

    def _is_positive_alert(self, alert: Optional[dict]) -> bool:
        """True if alert is positive (spike/composite_100 or positive opportunity)."""
        if alert is None:
            return False
        if alert.get("type") in ("spike", "composite_100"):
            return True
        if alert.get("type") == "opportunity":
            oid = alert.get("opportunity_id") or ""
            return oid in POSITIVE_OPPORTUNITY_IDS
        # Aural (phrase-triggered) = treat as negative for buffer (shorter cooldown)
        return False

    def record_insight_shown(self, alert: Optional[dict] = None) -> None:
        """Call when an insight popup was shown (any type)."""
        self._last_insight_popup_time = time.time()

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
            w_azure, w_mp = get_fusion_weights()
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
