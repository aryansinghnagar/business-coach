"""
Azure Engagement Metrics Module

Computes engagement metrics and B2B-relevant composite features from Azure Face API
output: 27 landmarks + emotion detection (anger, contempt, disgust, fear, happiness,
neutral, sadness, surprise). Used only when detection method is Azure Face API.
"""

from typing import Dict, Optional, Any
import numpy as np
from utils.face_detection_interface import FaceDetectionResult


# Azure Face API emotion keys (0-1 each)
EMOTION_KEYS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]

# B2B composite feature definitions: (name, formula weights or description)
# Each composite is 0-100. Formula: weighted sum of emotion scores (0-1) then * 100, clamped.
COMPOSITE_DEFINITIONS = {
    # Positive for engagement
    "receptive": {"happiness": 0.45, "contempt": -0.25, "anger": -0.30},  # high happy, low contempt/anger
    "focused": {"neutral": 0.55, "fear": -0.25, "disgust": -0.20},       # attentive, calm
    "interested": {"surprise": 0.50, "happiness": 0.50},                  # curious, positive
    "agreeable": {"happiness": 0.50, "contempt": -0.25, "disgust": -0.25},
    "open": {"happiness": 0.40, "surprise": 0.40, "contempt": -0.20},
    # Negative for engagement (resistance / discomfort)
    "skeptical": {"contempt": 0.55, "happiness": -0.45},
    "concerned": {"fear": 0.50, "sadness": 0.50},
    "disagreeing": {"contempt": 0.50, "anger": 0.50},
    "stressed": {"fear": 0.50, "anger": 0.50},
    "disengaged": {"neutral": 0.50, "happiness": -0.25, "surprise": -0.25},  # flat, not reacting
}


def _get_emotions(face_result: Optional[FaceDetectionResult]) -> Dict[str, float]:
    """Return emotion dict with all keys in [0, 1]; missing keys = 0."""
    out = {k: 0.0 for k in EMOTION_KEYS}
    if not face_result or not getattr(face_result, "emotions", None) or not isinstance(face_result.emotions, dict):
        return out
    for k in EMOTION_KEYS:
        v = face_result.emotions.get(k)
        if v is not None and isinstance(v, (int, float)):
            out[k] = float(np.clip(v, 0.0, 1.0))
    return out


def _get_head_pose(face_result: Optional[FaceDetectionResult]) -> Optional[Dict[str, float]]:
    """Return head pose (pitch, yaw, roll) in degrees if available."""
    if not face_result or not getattr(face_result, "head_pose", None):
        return None
    return face_result.head_pose


def compute_base_metrics(face_result: Optional[FaceDetectionResult]) -> Dict[str, float]:
    """
    Compute base emotion metrics (0-100) from Azure Face API emotions.
    Each emotion score (0-1) is scaled to 0-100.
    """
    emotions = _get_emotions(face_result)
    return {k: round(float(emotions[k]) * 100.0, 1) for k in EMOTION_KEYS}


def _score_composite(emotions: Dict[str, float], weights: Dict[str, float]) -> float:
    """One composite score 0-100: 0 = not detected, 100 = strongly detected."""
    raw = 0.0
    for emotion_key, w in weights.items():
        raw += emotions.get(emotion_key, 0.0) * w
    # Map weighted sum to 0-100; raw <= 0 -> 0, raw > 0 -> scale to 100
    return float(np.clip(raw * 200.0, 0.0, 100.0))


def compute_composite_metrics(face_result: Optional[FaceDetectionResult]) -> Dict[str, float]:
    """
    Compute B2B-relevant composite features from emotion combinations.
    Returns dict of composite name -> 0-100 score.
    """
    emotions = _get_emotions(face_result)
    out = {}
    for name, weights in COMPOSITE_DEFINITIONS.items():
        out[name] = round(_score_composite(emotions, weights), 1)
    return out


def compute_azure_engagement_score(
    face_result: Optional[FaceDetectionResult],
    base_metrics: Optional[Dict[str, float]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute overall engagement score (0-100) for Azure path.
    Favors receptive, focused, interested, agreeable, open; penalizes skeptical,
    concerned, disagreeing, stressed, disengaged.
    """
    if base_metrics is None:
        base_metrics = compute_base_metrics(face_result)
    if composite_metrics is None:
        composite_metrics = compute_composite_metrics(face_result)

    # Weights for overall score: positive composites add, negative subtract
    positive_composites = ["receptive", "focused", "interested", "agreeable", "open"]
    negative_composites = ["skeptical", "concerned", "disagreeing", "stressed", "disengaged"]

    pos_sum = sum(composite_metrics.get(k, 0.0) for k in positive_composites) / len(positive_composites)
    neg_sum = sum(composite_metrics.get(k, 0.0) for k in negative_composites) / len(negative_composites)
    # 0 = no engagement, 100 = high; raw = pos - neg so all zeros -> 0
    raw = pos_sum - neg_sum
    return float(np.clip(raw, 0.0, 100.0))


def get_all_azure_metrics(face_result: Optional[FaceDetectionResult]) -> Dict[str, Any]:
    """
    Return base metrics (emotions 0-100), composite metrics (B2B features 0-100),
    and overall engagement score (0-100). Used when detection method is Azure Face API.
    """
    base = compute_base_metrics(face_result)
    composite = compute_composite_metrics(face_result)
    score = compute_azure_engagement_score(face_result, base_metrics=base, composite_metrics=composite)
    return {
        "base": base,
        "composite": composite,
        "score": round(score, 1),
    }
