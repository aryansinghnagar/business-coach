"""
Utilities package for Business Meeting Copilot.

This package contains utility functions and helpers for engagement detection,
video source handling, context generation, and business-meeting feature extraction.
"""

from .video_source_handler import VideoSourceHandler, VideoSourceType
from .engagement_scorer import EngagementScorer, EngagementMetrics
from .context_generator import ContextGenerator, EngagementContext
from .face_detection_interface import FaceDetectorInterface, FaceDetectionResult
from .mediapipe_detector import MediaPipeFaceDetector
from .azure_face_detector import AzureFaceAPIDetector
from .business_meeting_feature_extractor import BusinessMeetingFeatureExtractor
from .expression_signifiers import ExpressionSignifierEngine

__all__ = [
    'VideoSourceHandler',
    'VideoSourceType',
    'EngagementScorer',
    'EngagementMetrics',
    'ContextGenerator',
    'EngagementContext',
    'FaceDetectorInterface',
    'FaceDetectionResult',
    'MediaPipeFaceDetector',
    'AzureFaceAPIDetector',
    'BusinessMeetingFeatureExtractor',
    'ExpressionSignifierEngine',
]
