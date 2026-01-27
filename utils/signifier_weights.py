"""
Signifier Weights Loader

Loads pre-trained weights for the 30 expression signifiers and 4 group weights
from an ML backend (URL), local file, or uses built-in defaults. Used by
ExpressionSignifierEngine for weighted individual and overall score calculation.

JSON format:
  {"signifier": [w1..w30], "group": [W1, W2, W3, W4]}
- signifier: 30 floats, one per signifier in SIGNIFIER_KEYS_ORDER. Weight for
  weighted-average within each group. Use 1.0 for equal.
- group: 4 floats for G1..G4 in composite. Default [0.35, 0.15, 0.35, 0.15].
"""

import json
import os
from typing import Callable, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

import config

# Must match ExpressionSignifierEngine._all_keys() order
SIGNIFIER_KEYS_ORDER: List[str] = [
    "g1_duchenne", "g1_pupil_dilation", "g1_eyebrow_flash", "g1_eye_contact", "g1_head_tilt",
    "g1_forward_lean", "g1_facial_symmetry", "g1_rhythmic_nodding", "g1_parted_lips", "g1_softened_forehead",
    "g2_look_up_lr", "g2_lip_pucker", "g2_eye_squint", "g2_thinking_brow", "g2_chin_stroke",
    "g2_stillness", "g2_lowered_brow",
    "g3_contempt", "g3_nose_crinkle", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition",
]

DEFAULT_SIGNIFIER_WEIGHTS: List[float] = [1.0] * 30
DEFAULT_GROUP_WEIGHTS: List[float] = [0.35, 0.15, 0.35, 0.15]

# In-memory weights (updated by load_weights, set_weights)
_current: Dict[str, List[float]] = {
    "signifier": list(DEFAULT_SIGNIFIER_WEIGHTS),
    "group": list(DEFAULT_GROUP_WEIGHTS),
}


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
        if len(signifier) != 30:
            raise ValueError("signifier must have 30 elements")
        _current["signifier"] = [float(x) for x in signifier]
    if group is not None:
        if len(group) != 4:
            raise ValueError("group must have 4 elements")
        _current["group"] = [float(x) for x in group]
    return get_weights()


def _apply(data: dict) -> None:
    sig = data.get("signifier")
    grp = data.get("group")
    if isinstance(sig, list) and len(sig) == 30:
        _current["signifier"] = [float(x) for x in sig]
    if isinstance(grp, list) and len(grp) == 4:
        _current["group"] = [float(x) for x in grp]


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
    return get_weights()


def build_weights_provider() -> Callable[[], Dict[str, List[float]]]:
    """Return a callable that returns the current weights (for ExpressionSignifierEngine)."""
    return get_weights
