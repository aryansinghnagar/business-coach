"""
Signifier Weights Loader

Loads pre-trained weights for the 30 expression signifiers and 4 group weights
from an ML backend (URL), local file, or uses built-in defaults. Used by
ExpressionSignifierEngine for weighted individual and overall score calculation.

Psychology-informed defaults; research refinement (FACS, Mehrabian, Cialdini, Kahneman):
- Eye contact: strong correlate of trust and engagement (Mehrabian; 70-80% ideal in 1:1)—up-weighted.
- Duchenne (AU6+AU12): genuine positive affect (Ekman & Friesen)—up-weighted.
- Gaze aversion (sustained), lip compression, contempt: robust resistance/disengagement cues—up-weighted.
- G4 (fixed_gaze, relaxed_exhale, smile_transition): decision/commitment signals (Cialdini)—up-weighted.
- Pupil dilation: noisy proxy—de-weighted. Chin stroke, mouth cover: placeholders—de-weighted.

JSON format:
  {"signifier": [w1..w30], "group": [W1, W2, W3, W4], optional "fusion": {"azure": float, "mediapipe": float}}
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

import config

# Must match ExpressionSignifierEngine._all_keys() / expression_signifiers.SIGNIFIER_KEYS order
SIGNIFIER_KEYS_ORDER: List[str] = [
    "g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
    "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
    "g1_micro_smile", "g1_brow_raise_sustained", "g1_mouth_open_receptive", "g1_eye_widening", "g1_nod_intensity",
    "g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
    "g2_stillness", "g2_lowered_brow", "g2_brow_furrow_deep", "g2_gaze_shift_frequency", "g2_mouth_tight_eval",
    "g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
    "g3_lip_corner_dip", "g3_brow_lower_sustained", "g3_eye_squeeze", "g3_head_shake",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition", "g4_mouth_relax", "g4_smile_sustain",
]

# Research-refined signifier weights (FACS, Mehrabian, Cialdini, Kahneman; Waller 2024, Edmondson).
# 44 elements; new signifiers default 1.0.
DEFAULT_SIGNIFIER_WEIGHTS: List[float] = [
    1.05, 0.55, 1.00, 1.55, 1.20,  # g1: duchenne+, pupil-, eyebrow_flash, eye_contact+, head_tilt
    1.25, 1.00, 1.35, 1.10, 1.00,  # g1: forward_lean, symmetry, nodding, parted_lips, softened_forehead
    1.05, 1.00, 1.00, 1.00, 1.10,  # g1: micro_smile, brow_raise_sustained, mouth_open_receptive, eye_widening, nod_intensity
    1.00, 1.00, 1.15, 1.30, 0.50,  # g2: look_up_lr, lip_pucker, eye_squint, thinking_brow, chin_stroke-
    1.10, 1.00, 1.15, 1.00, 1.05,  # g2: stillness, lowered_brow, brow_furrow_deep, gaze_shift_frequency, mouth_tight_eval
    1.40, 1.10, 1.50, 1.00, 1.25,  # g3: contempt+, nose_crinkle, lip_compression+, eye_block, jaw_clench
    1.00, 1.55, 1.05, 0.85, 0.50,  # g3: rapid_blink, gaze_aversion+, no_nod, narrowed_pupils, mouth_cover-
    1.10, 1.10, 1.00, 1.00,        # g3: lip_corner_dip, brow_lower_sustained, eye_squeeze, head_shake
    1.40, 1.50, 1.45, 1.15, 1.20,  # g4: relaxed_exhale+, fixed_gaze+, smile_transition, mouth_relax, smile_sustain
]

# Group weights: G1 (interest), G2 (cognitive load), G3 (resistance), G4 (decision-ready)
# Balanced; G2 (cognitive load) elevated—thinking/evaluating is meaningful in B2B
DEFAULT_GROUP_WEIGHTS: List[float] = [0.30, 0.20, 0.30, 0.20]

# Fusion weights for unified mode: score = azure_w * azure_score + mediapipe_w * mediapipe_score
DEFAULT_FUSION_AZURE: float = 0.5
DEFAULT_FUSION_MEDIAPIPE: float = 0.5

# In-memory weights (updated by load_weights, set_weights)
_current: Dict[str, List[float]] = {
    "signifier": list(DEFAULT_SIGNIFIER_WEIGHTS),
    "group": list(DEFAULT_GROUP_WEIGHTS),
}
_fusion: Dict[str, float] = {"azure": DEFAULT_FUSION_AZURE, "mediapipe": DEFAULT_FUSION_MEDIAPIPE}


def get_weights() -> Dict[str, List[float]]:
    """Return current signifier and group weights. Safe to modify the returned dict."""
    return {"signifier": list(_current["signifier"]), "group": list(_current["group"])}


def set_weights(
    signifier: Optional[List[float]] = None,
    group: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """
    Update in-memory weights. Partial update: only provided keys are changed.
    signifier: 30 floats; group: 4 floats.
    """
    if signifier is not None:
        n = len(SIGNIFIER_KEYS_ORDER)
        if len(signifier) != n:
            raise ValueError(f"signifier must have {n} elements")
        _current["signifier"] = [float(x) for x in signifier]
    if group is not None:
        if len(group) != 4:
            raise ValueError("group must have 4 elements")
        _current["group"] = [float(x) for x in group]
    return get_weights()


def _apply(data: dict) -> None:
    sig = data.get("signifier")
    grp = data.get("group")
    n = len(SIGNIFIER_KEYS_ORDER)
    if isinstance(sig, list) and len(sig) == n:
        _current["signifier"] = [float(x) for x in sig]
    if isinstance(grp, list) and len(grp) == 4:
        _current["group"] = [float(x) for x in grp]
    fusion = data.get("fusion")
    if isinstance(fusion, dict):
        a = fusion.get("azure")
        m = fusion.get("mediapipe")
        if isinstance(a, (int, float)) and isinstance(m, (int, float)):
            total = float(a) + float(m)
            if total > 0:
                _fusion["azure"] = float(a) / total
                _fusion["mediapipe"] = float(m) / total


def get_fusion_weights() -> Tuple[float, float]:
    """Return (azure_weight, mediapipe_weight) for unified score fusion. Sum to 1.0."""
    return (_fusion["azure"], _fusion["mediapipe"])


def set_fusion_weights(azure: float, mediapipe: float) -> Tuple[float, float]:
    """
    Set fusion weights for unified mode. Weights are normalized to sum to 1.0.
    Returns (azure_weight, mediapipe_weight) after normalization.
    """
    total = float(azure) + float(mediapipe)
    if total <= 0:
        return get_fusion_weights()
    _fusion["azure"] = float(azure) / total
    _fusion["mediapipe"] = float(mediapipe) / total
    return get_fusion_weights()


def load_weights() -> Dict[str, List[float]]:
    """
    Load from SIGNIFIER_WEIGHTS_URL, else SIGNIFIER_WEIGHTS_PATH, else defaults.
    Updates _current and returns get_weights().
    """
    # 1) URL
    url = getattr(config, "SIGNIFIER_WEIGHTS_URL", None)
    if url and requests is not None:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                _apply(r.json())
                return get_weights()
        except Exception:
            pass

    # 2) File
    path = getattr(config, "SIGNIFIER_WEIGHTS_PATH", "weights/signifier_weights.json")
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _apply(json.load(f))
            return get_weights()
        except Exception:
            pass

    # 3) Defaults (already in _current from module load)
    _current["signifier"] = list(DEFAULT_SIGNIFIER_WEIGHTS)
    _current["group"] = list(DEFAULT_GROUP_WEIGHTS)
    _fusion["azure"] = getattr(config, "FUSION_AZURE_WEIGHT", DEFAULT_FUSION_AZURE)
    _fusion["mediapipe"] = getattr(config, "FUSION_MEDIAPIPE_WEIGHT", DEFAULT_FUSION_MEDIAPIPE)
    total = _fusion["azure"] + _fusion["mediapipe"]
    if total > 0:
        _fusion["azure"] /= total
        _fusion["mediapipe"] /= total
    return get_weights()


def build_weights_provider() -> Callable[[], Dict[str, List[float]]]:
    """Return a callable that returns the current weights (for ExpressionSignifierEngine)."""
    return get_weights
