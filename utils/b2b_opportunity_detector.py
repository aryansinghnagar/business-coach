"""
B2B Opportunity Detector — Complex features from engagement metrics

Derives 20+ psychology-based opportunity features from the 30 signifiers and 4 group means
to predict engagement state and detect B2B-relevant moments in real time. Used to trigger
custom popup + TTS via Azure OpenAI (e.g. alleviate cognitive load, address skepticism,
finalize a deal).

Opportunities are evaluated in priority order; the first that fires (and passes cooldown)
is returned. All thresholds use the 0–100 scale (G1, G2, G3, G4; G3 = 100 - resistance_raw).
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Cooldown per opportunity type (seconds)
OPPORTUNITY_COOLDOWN_SEC = 32
_HISTORY_LEN = 12

# Priority order: closing/decision first, then resistance/confusion, then interest/rapport
OPPORTUNITY_PRIORITY: List[str] = [
    "closing_window",
    "decision_ready",
    "ready_to_sign",
    "buying_signal",
    "commitment_cue",
    "cognitive_overload_risk",
    "confusion_moment",
    "need_clarity",
    "skepticism_surface",
    "objection_moment",
    "resistance_peak",
    "hesitation_moment",
    "disengagement_risk",
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
    t = _last_fire_time.get(opportunity_id, 0.0)
    return (now - t) >= OPPORTUNITY_COOLDOWN_SEC


def _fire(opportunity_id: str, now: float) -> None:
    _last_fire_time[opportunity_id] = now


def _eval_closing_window(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g4"]) < 6:
        return False
    h4 = hist["g4"]
    if g4 >= 60 and (g4 - min(h4)) >= 16 and g3 >= 55:
        return True
    return False


def _eval_decision_ready(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g4 >= 62 and g1 >= 56 and g3 >= 58


def _eval_ready_to_sign(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g4 >= 65 and g2 < 50 and g3 >= 56


def _eval_buying_signal(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g1 >= 60 and g4 >= 52 and g3 >= 58


def _eval_commitment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g4"]) < 4:
        return False
    return g4 >= 58 and np.mean(hist["g4"][-4:]) >= 56 and g3 >= 56


def _eval_cognitive_overload_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 58 and g1 < 52


def _eval_confusion_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    r = _g3_raw(g3)
    return g2 >= 58 and (r >= 48 or (len(hist["g2"]) >= 4 and g2 > np.mean(hist["g2"][:-2])))


def _eval_need_clarity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 55 and g1 < 56


def _eval_skepticism_surface(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    r = _g3_raw(g3)
    if len(hist["g3"]) < 4:
        return r >= 50
    mean_g3 = np.mean(hist["g3"][:-1])
    r_prev = 100.0 - mean_g3
    return r >= 50 and r > r_prev + 4


def _eval_objection_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return _g3_raw(g3) >= 52


def _eval_resistance_peak(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return _g3_raw(g3) >= 56


def _eval_hesitation_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 54 and _g3_raw(g3) >= 46


def _eval_disengagement_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g1 < 46 and _g3_raw(g3) >= 48


def _eval_objection_fading(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g3"]) < 5:
        return False
    mean_prev = np.mean(hist["g3"][:-2])
    return mean_prev < 52 and g3 >= mean_prev + 10


def _eval_aha_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g1"]) < 3 or len(hist["g2"]) < 3:
        return False
    g2_ago = hist["g2"][-2]
    return g2_ago >= 55 and g1 >= 58 and g3 >= 54


def _eval_re_engagement_opportunity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g1"]) < 6:
        return False
    return min(hist["g1"]) < 45 and g1 >= 50


def _eval_alignment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g1"]) < 6 or len(hist["g4"]) < 6:
        return False
    return (g1 - hist["g1"][0]) >= 10 and (g4 - hist["g4"][0]) >= 10


def _eval_genuine_interest(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g1 >= 58 and g3 >= 56


def _eval_listening_active(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g1 >= 55 and g3 >= 54


def _eval_trust_building_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 55:
        return g1 >= 54 and g3 >= 58
    return g1 >= 54 and g3 >= 58


def _eval_urgency_sensitive(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g4 >= 56 and g2 >= 50 and g3 >= 54


def _eval_processing_deep(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 56 and g3 >= 58


def _eval_attention_peak(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    if sig is not None and sig.get("g1_eye_contact", 0) >= 58:
        return g1 >= 60
    return g1 >= 62


def _eval_rapport_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 52:
        return g1 >= 54
    return g1 >= 56


_EVALUATORS: Dict[str, Any] = {
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
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Evaluate B2B opportunity features in priority order. Returns (opportunity_id, context) for the
    first opportunity that fires and has passed cooldown, else None.

    Args:
        group_means: Current G1, G2, G3, G4 (0–100).
        group_history: Optional dict of deques (g1, g2, g3, g4) for temporal logic. If None, internal history is used (must be updated via update_history).
        signifier_scores: Optional 30 signifier scores for features that use eye_contact, symmetry, etc.
        now: Timestamp for cooldown; defaults to time.time().

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

    for oid in OPPORTUNITY_PRIORITY:
        if not _check_cooldown(oid, now):
            continue
        fn = _EVALUATORS.get(oid)
        if fn is None:
            continue
        try:
            if oid in ("trust_building_moment", "attention_peak", "rapport_moment"):
                fired = fn(g1, g2, g3, g4, hist, sig)
            else:
                fired = fn(g1, g2, g3, g4, hist)
            if fired:
                _fire(oid, now)
                context = {
                    "g1": g1, "g2": g2, "g3": g3, "g4": g4,
                    "signifier_scores": sig,
                }
                return (oid, context)
        except Exception:
            continue
    return None


def update_history_from_detector(group_means: Dict[str, float]) -> None:
    """Update internal history when caller uses external group_history (e.g. engagement_state_detector)."""
    _update_history(group_means)


def clear_opportunity_state() -> None:
    """Clear cooldowns and history (e.g. when engagement stops)."""
    global _last_fire_time, _history_means
    _last_fire_time.clear()
    for k in _history_means:
        _history_means[k].clear()
