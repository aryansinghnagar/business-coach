"""
Azure Engagement Metrics Module

Computes engagement metrics and B2B-relevant composite features from Azure Face API
emotions. Psychology-informed mappings (Ekman-based, with nuanced combinations):
- Contempt: one-sided lip raise; skepticism/resistance (validated cross-culturally).
- Happiness + low contempt: receptive; contempt + low happiness: skeptical.
- Neutral + low reactivity: disengaged (flat affect).
- Surprise + happiness: interested/curious; fear + sadness: concerned.
"""

from typing import Dict, Optional, Any
import numpy as np
from utils.face_detection_interface import FaceDetectionResult


EMOTION_KEYS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]

# B2B composites: weighted sums of emotion scores (0-1) -> 0-100
# Positive: receptive, focused, interested, agreeable, open
# Negative: skeptical (contempt), concerned (fear+sadness), disagreeing, stressed, disengaged
COMPOSITE_DEFINITIONS = {
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


def binarize_metrics(metrics_dict: Dict[str, float], up: float = 55.0) -> Dict[str, float]:
    """
    Convert continuous 0-100 metrics to strict 0 or 100 for display/API.
    v >= up -> 100, else 0. Default 0 when missing/invalid.
    """
    if not metrics_dict or not isinstance(metrics_dict, dict):
        return {}
    out = {}
    for k, v in metrics_dict.items():
        try:
            x = float(v)
            out[k] = 100.0 if (np.isfinite(x) and x >= up) else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


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

    pos_sum = sum(composite_metrics.get(k, 0.0) for k in POSITIVE_COMPOSITES) / len(POSITIVE_COMPOSITES)
    neg_sum = sum(composite_metrics.get(k, 0.0) for k in NEGATIVE_COMPOSITES) / len(NEGATIVE_COMPOSITES)
    raw = pos_sum - neg_sum
    # Spike when positive engagement is high; drop when negative is high
    if pos_sum > 70.0:
        raw = min(100.0, raw + 8.0)
    if neg_sum > 70.0:
        raw = max(0.0, raw - 12.0)
    return float(np.clip(raw, 0.0, 100.0))


# Positive/negative composite names for overall score (must match compute_azure_engagement_score)
POSITIVE_COMPOSITES = ["receptive", "focused", "interested", "agreeable", "open"]
NEGATIVE_COMPOSITES = ["skeptical", "concerned", "disagreeing", "stressed", "disengaged"]


def get_azure_score_breakdown(
    face_result: Optional[FaceDetectionResult],
    base_metrics: Optional[Dict[str, float]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Return a step-by-step breakdown of how the Azure engagement score is calculated.
    Used for frontend "How is the score calculated?" and transparency.
    """
    if base_metrics is None:
        base_metrics = compute_base_metrics(face_result)
    if composite_metrics is None:
        composite_metrics = compute_composite_metrics(face_result)

    pos_avg = sum(composite_metrics.get(k, 0.0) for k in POSITIVE_COMPOSITES) / len(POSITIVE_COMPOSITES)
    neg_avg = sum(composite_metrics.get(k, 0.0) for k in NEGATIVE_COMPOSITES) / len(NEGATIVE_COMPOSITES)
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
        "positiveComposites": POSITIVE_COMPOSITES,
        "negativeComposites": NEGATIVE_COMPOSITES,
        "posAvg": round(pos_avg, 2),
        "negAvg": round(neg_avg, 2),
        "rawBeforeAdjustments": round(pos_avg - neg_avg, 2),
        "adjustments": adjustments if adjustments else ["none"],
        "score": round(score, 1),
    }


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
