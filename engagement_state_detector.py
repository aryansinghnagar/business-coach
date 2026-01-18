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
from utils.business_meeting_feature_extractor import BusinessMeetingFeatureExtractor


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
        detection_method: Optional[str] = None
    ):
        """
        Initialize the engagement state detector.
        
        Args:
            smoothing_window: Number of frames to average for smoothing (default: 10)
            min_face_confidence: Minimum confidence for face detection (default: 0.5)
            update_callback: Optional callback function called when state updates
            detection_method: Face detection method to use ("mediapipe" or "azure_face_api").
                             If None, uses config.FACE_DETECTION_METHOD
        """
        # Determine detection method
        if detection_method is None:
            detection_method = config.FACE_DETECTION_METHOD.lower()
        
        # Initialize face detector based on method
        self.face_detector: Optional[FaceDetectorInterface] = None
        
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
        self.score_history.clear()
        self.metrics_history.clear()
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
        
        # Detect faces using the configured detector
        face_results = self.face_detector.detect_faces(frame)
        
        if not face_results:
            # No face detected
            self.consecutive_no_face_frames += 1
            
            with self.lock:
                if self.current_state and self.current_state.face_detected:
                    # If we had a face before but lost it, maintain state briefly
                    # but reset after too many consecutive frames without face
                    if self.consecutive_no_face_frames < self.max_no_face_frames:
                        return EngagementState(
                            score=self.current_state.score,
                            level=self.current_state.level,
                            metrics=self.current_state.metrics,
                            context=self.current_state.context,
                            timestamp=time.time(),
                            face_detected=False,
                            confidence=max(0.0, self.current_state.confidence - 0.1)
                        )
                    else:
                        # Too many frames without face - reset to prevent stuck state
                        print("Warning: Too many consecutive frames without face, resetting state")
                        self.consecutive_no_face_frames = 0
                        return None
                else:
                    # Never had a face or already marked as not detected
                    return None
        
        # Use the first detected face
        face_result = face_results[0]
        landmarks = face_result.landmarks
        
        # Extract business-meeting focused blendshape features (100 most important)
        blendshape_features = self.feature_extractor.extract_features(
            landmarks, frame.shape, face_result
        )
        
        # Debug: Check if features are all zeros or very small
        if np.allclose(blendshape_features, 0.0, atol=1e-6):
            print("Warning: All features are zero or near-zero, check feature extraction")
        elif np.max(np.abs(blendshape_features)) < 0.01:
            print(f"Warning: Features are very small (max={np.max(np.abs(blendshape_features)):.6f})")
        
        # Compute engagement metrics
        metrics = self.scorer.compute_metrics(blendshape_features, landmarks, frame.shape)
        
        # Debug: Log metrics for troubleshooting
        if len(self.score_history) % 30 == 0:  # Log every ~1 second at 30 FPS
            print(f"Debug - Metrics: attention={metrics.attention:.1f}, eye_contact={metrics.eye_contact:.1f}, "
                  f"expressiveness={metrics.facial_expressiveness:.1f}, head_movement={metrics.head_movement:.1f}, "
                  f"symmetry={metrics.symmetry:.1f}, mouth_activity={metrics.mouth_activity:.1f}")
            print(f"Debug - Features[0-2] (EAR): {blendshape_features[0]:.4f}, {blendshape_features[1]:.4f}, {blendshape_features[2]:.4f}")
        
        # Calculate overall engagement score
        score = self.scorer.calculate_score(metrics)
        
        # Validate score is reasonable (not NaN or invalid)
        if not np.isfinite(score) or score < 0 or score > 100:
            print(f"Warning: Invalid score computed: {score}, using fallback")
            score = 50.0  # Default to medium engagement
        
        # Apply smoothing using simple moving average over last 10 frames
        self.score_history.append(score)
        self.metrics_history.append(metrics)
        
        # Calculate simple average over the last 10 frames (or all available if less than 10)
        # This provides smooth updates while still being responsive to changes
        if len(self.score_history) > 0:
            # Use the last 10 frames (or all available if less than 10)
            recent_scores = list(self.score_history)[-10:] if len(self.score_history) >= 10 else list(self.score_history)
            smoothed_score = np.mean(recent_scores)
        else:
            smoothed_score = score
        
        # Ensure smoothed score is valid
        if not np.isfinite(smoothed_score):
            smoothed_score = score if np.isfinite(score) else 50.0
        
        # Clamp to valid range
        smoothed_score = max(0.0, min(100.0, smoothed_score))
        
        # Debug: Log score progression (less frequently to avoid spam)
        if len(self.score_history) % 30 == 0:  # Every ~1 second at 30 FPS
            print(f"Debug - Raw score: {score:.1f}, Smoothed (10-frame avg): {smoothed_score:.1f}, History size: {len(self.score_history)}")
        
        smoothed_metrics = self._average_metrics() if self.metrics_history else metrics
        
        # Reset no-face counter since we detected a face
        self.consecutive_no_face_frames = 0
        
        # Determine engagement level
        level = EngagementLevel.from_score(smoothed_score)
        
        # Generate context for AI coaching
        context = self.context_generator.generate_context(
            smoothed_score,
            smoothed_metrics,
            level
        )
        
        # Calculate confidence based on detection stability
        confidence = self._calculate_confidence()
        
        return EngagementState(
            score=float(smoothed_score),
            level=level,
            metrics=smoothed_metrics,
            context=context,
            timestamp=time.time(),
            face_detected=True,
            confidence=confidence
        )
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """
        Calculate the area of a polygon using the shoelace formula.
        
        Args:
            vertices: numpy array of vertex coordinates (N, 2)
        
        Returns:
            float: Polygon area
        """
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
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
