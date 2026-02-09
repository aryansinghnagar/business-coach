"""
B2B Opportunity Detector — Psychology-based metric combinations

Detects psychologically meaningful COMBINATIONS of engagement metrics (not single spikes)
to predict engagement state and trigger insights. Based on cutting-edge research:
- Duchenne smile + forward lean + eye contact = genuine interest (approach motivation)
- High G2 + low G1 + gaze aversion = cognitive overload (Kahneman System 2 maxed)
- G3 spike + G1 drop = resistance surfacing (approach-avoidance conflict)
- G1 + G4 high, G2/G3 low = decision-ready (commitment signals, low cognitive load)

Opportunities are evaluated in priority order; the first that fires (and passes cooldown)
is returned. All thresholds use the 0–100 scale (G1, G2, G3, G4; G3 = 100 - resistance_raw).

Client retention strategy:
- Negative opportunities (confusion, skepticism, objection, resistance, disengagement, cognitive
  overload) are prioritized: lower detection thresholds and shorter cooldowns so the user can
  address concerns before the client voices them.
- Positive opportunities (closing, decision-ready, buying signal, rapport) use higher thresholds
  and longer cooldowns to reduce popup frequency while still surfacing strong moments to close
  or improve perception.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import config as _config
except Exception:
    _config = None

# Type for recent speech tags (from insight_generator.get_recent_speech_tags)
SpeechTag = Dict[str, Any]  # {"category": str, "phrase": str, "time": float}

# Cooldowns by polarity (config overrides if present)
NEGATIVE_OPPORTUNITY_COOLDOWN_SEC = float(getattr(_config, "NEGATIVE_OPPORTUNITY_COOLDOWN_SEC", 14))
POSITIVE_OPPORTUNITY_COOLDOWN_SEC = float(getattr(_config, "POSITIVE_OPPORTUNITY_COOLDOWN_SEC", 50))
_HISTORY_LEN = 12

# Negative = risk to relationship/deal; surface more readily (shorter cooldown, lower thresholds)
NEGATIVE_OPPORTUNITY_IDS: set = {
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
    "cognitive_overload_risk",
    "confusion_moment",
    "need_clarity",
    "skepticism_surface",
    "objection_moment",
    "resistance_peak",
    "hesitation_moment",
    "disengagement_risk",
}
# Positive = moments to capitalize; require stronger evidence, longer cooldown
POSITIVE_OPPORTUNITY_IDS: set = {
    "decision_readiness_multimodal",
    "aha_insight_multimodal",
    "closing_window",
    "decision_ready",
    "ready_to_sign",
    "buying_signal",
    "commitment_cue",
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
}

# Priority order: all NEGATIVE first so we don't miss concerns, then positive
OPPORTUNITY_PRIORITY: List[str] = [
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
    "cognitive_overload_risk",
    "confusion_moment",
    "need_clarity",
    "skepticism_surface",
    "objection_moment",
    "resistance_peak",
    "hesitation_moment",
    "disengagement_risk",
    "decision_readiness_multimodal",
    "aha_insight_multimodal",
    "closing_window",
    "decision_ready",
    "ready_to_sign",
    "buying_signal",
    "commitment_cue",
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

# Opportunity IDs that require recent_speech_tags (multimodal)
MULTIMODAL_OPPORTUNITY_IDS: set = {
    "decision_readiness_multimodal",
    "cognitive_overload_multimodal",
    "skepticism_objection_multimodal",
    "aha_insight_multimodal",
    "disengagement_multimodal",
    "confusion_multimodal",
    "tension_objection_multimodal",
}

# Opportunity IDs that need composite_metrics and/or acoustic_tags (passed to evaluator)
COMPOSITE_ACOUSTIC_OPPORTUNITY_IDS: set = {
    "confusion_multimodal",
    "tension_objection_multimodal",
    "loss_of_interest",
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
}

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
    """Use shorter cooldown for negative (retention), longer for positive (reduce frequency)."""
    t = _last_fire_time.get(opportunity_id, 0.0)
    sec = NEGATIVE_OPPORTUNITY_COOLDOWN_SEC if opportunity_id in NEGATIVE_OPPORTUNITY_IDS else POSITIVE_OPPORTUNITY_COOLDOWN_SEC
    return (now - t) >= sec


def _fire(opportunity_id: str, now: float) -> None:
    _last_fire_time[opportunity_id] = now


def _has_recent_category(speech_tags: List[SpeechTag], categories: List[str]) -> bool:
    """True if any recent speech tag has category in categories."""
    if not speech_tags or not categories:
        return False
    cats = set(categories)
    return any((t.get("category") or "") in cats for t in speech_tags)


def _eval_closing_window(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: higher threshold to reduce frequency; only clear closing signals."""
    if len(hist["g4"]) < 6:
        return False
    h4 = hist["g4"]
    if g4 >= 65 and (g4 - min(h4)) >= 18 and g3 >= 55:
        return True
    return False


def _eval_decision_ready(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Decision-ready = high commitment (G4) + positive engagement (G1) + low resistance.
    Positive: higher threshold so positive insights only when multiple strong signals.
    """
    return g4 >= 70 and g1 >= 62 and g3 >= 62


def _eval_ready_to_sign(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Ready to sign = high commitment (G4) + LOW cognitive load (G2 < 50) +
    low resistance. Positive: higher threshold to reduce frequency.
    """
    return g4 >= 70 and g2 < 50 and g3 >= 58


def _eval_buying_signal(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Buying signal = high interest (G1) + decision cues (G4) + low resistance.
    Positive: higher threshold so we only surface on clear multiple positive signals.
    """
    return g1 >= 66 and g4 >= 60 and g3 >= 62


def _eval_commitment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if len(hist["g4"]) < 4:
        return False
    return g4 >= 62 and np.mean(hist["g4"][-4:]) >= 58 and g3 >= 58


def _eval_cognitive_overload_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Cognitive overload = high G2 + LOW G1. Negative: lower threshold for early catch.
    """
    return g2 >= 54 and g1 < 54


def _eval_confusion_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Confusion = high G2 + resistance or G2 spike. Negative: lower threshold so we don't miss.
    """
    r = _g3_raw(g3)
    return g2 >= 53 and (r >= 46 or (len(hist["g2"]) >= 4 and g2 > np.mean(hist["g2"][:-2])))


def _eval_need_clarity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Need clarity = moderate G2 + low G1. Negative: lower threshold so we don't miss subtle need.
    """
    return g2 >= 50 and g1 < 60


def _eval_skepticism_surface(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Skepticism = resistance rising. Negative: lower threshold for early catch.
    """
    r = _g3_raw(g3)
    if len(hist["g3"]) < 4:
        return r >= 46
    mean_g3 = np.mean(hist["g3"][:-1])
    r_prev = 100.0 - mean_g3
    return r >= 46 and r > r_prev + 3


def _eval_objection_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Objection = moderate-high resistance. Negative: lower threshold for early catch.
    Temporal consistency: require G3_raw >= 48 in at least 2 of last 3 frames (group_history)
    so one-frame spikes do not fire.
    """
    r = _g3_raw(g3)
    if r < 48:
        return False
    if len(hist["g3"]) < 3:
        return True
    # 2 of last 3 frames with G3_raw >= 48 (i.e. G3 <= 52)
    recent = hist["g3"][-3:]
    count_high_r = sum(1 for g in recent if _g3_raw(g) >= 48)
    return count_high_r >= 2


def _eval_resistance_peak(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Resistance peak = high G3 raw. Negative: lower threshold for early catch.
    Temporal consistency: require G3_raw >= 52 in at least 2 of last 3 frames so one-frame
    spikes do not fire.
    """
    r = _g3_raw(g3)
    if r < 52:
        return False
    if len(hist["g3"]) < 3:
        return True
    recent = hist["g3"][-3:]
    count_high_r = sum(1 for g in recent if _g3_raw(g) >= 52)
    return count_high_r >= 2


def _eval_hesitation_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Hesitation = moderate G2 + moderate resistance. Negative: lower threshold.
    """
    return g2 >= 52 and _g3_raw(g3) >= 44


def _eval_disengagement_risk(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Disengagement = LOW G1 + moderate resistance. Negative: lower threshold to catch slippage.
    """
    return g1 < 52 and _g3_raw(g3) >= 44


def _eval_objection_fading(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    if len(hist["g3"]) < 5:
        return False
    mean_prev = np.mean(hist["g3"][:-2])
    return mean_prev < 52 and g3 >= mean_prev + 10


def _eval_aha_moment(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Aha = was processing -> now engaged. Positive: modestly higher bar.
    """
    if len(hist["g1"]) < 3 or len(hist["g2"]) < 3:
        return False
    g2_ago = hist["g2"][-2]
    return g2_ago >= 56 and g1 >= 60 and g3 >= 56


def _eval_re_engagement_opportunity(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Re-engagement = was disengaged -> attention returning. Positive: slightly higher g1.
    """
    if len(hist["g1"]) < 6:
        return False
    return min(hist["g1"]) < 44 and g1 >= 52


def _eval_alignment_cue(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """
    Psychology: Alignment = G1 and G4 rising together. Positive: require slightly larger rise.
    """
    if len(hist["g1"]) < 6 or len(hist["g4"]) < 6:
        return False
    return (g1 - hist["g1"][0]) >= 12 and (g4 - hist["g4"][0]) >= 12


def _eval_genuine_interest(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Psychology: Genuine interest = high G1 + low resistance. Positive: higher threshold for multi-signal."""
    return g1 >= 64 and g3 >= 60


def _eval_listening_active(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    """Positive: slightly higher bar to reduce frequency."""
    return g1 >= 57 and g3 >= 56


def _eval_trust_building_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 55:
        return g1 >= 58 and g3 >= 58
    return g1 >= 58 and g3 >= 58


def _eval_urgency_sensitive(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g4 >= 56 and g2 >= 50 and g3 >= 54


def _eval_processing_deep(g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]]) -> bool:
    return g2 >= 56 and g3 >= 58


def _eval_attention_peak(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_eye_contact", 0) >= 58:
        return g1 >= 64
    return g1 >= 66


def _eval_rapport_moment(
    g1: float, g2: float, g3: float, g4: float, hist: Dict[str, List[float]], sig: Optional[Dict[str, float]] = None
) -> bool:
    """Positive: higher threshold to reduce frequency."""
    if sig is not None and sig.get("g1_facial_symmetry", 0) >= 52:
        return g1 >= 58
    return g1 >= 60


# ----- Multimodal composites (facial + speech; require recent_speech_tags) -----


def _eval_decision_readiness_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Decision-readiness with speech. Positive: higher threshold."""
    if not _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    return g4 >= 64 and g1 >= 58 and g3 >= 58


def _eval_cognitive_overload_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Cognitive overload with speech. Negative: lower threshold for early catch."""
    if not _has_recent_category(speech_tags, ["confusion", "concern"]):
        return False
    return g2 >= 53 and g1 < 56


def _eval_skepticism_objection_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Skepticism/objection with speech. Negative: lower threshold for early catch."""
    if not _has_recent_category(speech_tags, ["objection", "concern"]):
        return False
    return _g3_raw(g3) >= 44


def _eval_aha_insight_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Aha/insight with speech. Positive: slightly higher bar."""
    if not _has_recent_category(speech_tags, ["interest", "realization"]):
        return False
    if len(hist["g1"]) < 3 or len(hist["g2"]) < 3:
        return False
    g2_ago = hist["g2"][-2]
    return g2_ago >= 53 and g1 >= 60 and g3 >= 55


def _eval_disengagement_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
) -> bool:
    """Disengagement with lack of positive speech. Negative: lower threshold for early catch."""
    if _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    return g1 < 48 and g4 < 52 and _g3_raw(g3) >= 42


# ----- Composite + acoustic opportunity evaluators (need composite_metrics, acoustic_tags) -----


def _eval_confusion_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Confusion: speech confusion/concern AND (G2 high or composite confusion_multimodal high). Negative."""
    if not _has_recent_category(speech_tags, ["confusion", "concern"]):
        return False
    comp = composite_metrics or {}
    if comp.get("confusion_multimodal", 0) >= 55:
        return True
    return g2 >= 50


def _eval_tension_objection_multimodal(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Tension/objection: G3 raw elevated AND (recent objection/concern or composite high). Negative."""
    if _g3_raw(g3) < 42:
        return False
    if _has_recent_category(speech_tags, ["objection", "concern"]):
        return True
    comp = composite_metrics or {}
    return comp.get("tension_objection_multimodal", 0) >= 55


def _eval_loss_of_interest(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Loss of interest: composite loss_of_interest_multimodal high OR (G1 low + no commit/interest + acoustic withdrawal). Negative."""
    comp = composite_metrics or {}
    if comp.get("loss_of_interest_multimodal", 0) >= 58:
        return True
    if g1 >= 50:
        return False
    if _has_recent_category(speech_tags, ["commitment", "interest"]):
        return False
    ac = set(acoustic_tags or [])
    if "acoustic_disengagement_risk" in ac:
        return True
    return g1 < 45 and not _has_recent_category(speech_tags, ["commitment", "interest"])


def _eval_acoustic_disengagement_risk(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice suggests disengagement and face not highly engaged. Negative: don't miss withdrawal."""
    ac = set(acoustic_tags or [])
    if "acoustic_disengagement_risk" not in ac:
        return False
    return g1 < 60


def _eval_acoustic_uncertainty(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice uncertainty + (elevated G2 or recent confusion/concern). Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_uncertainty" not in ac:
        return False
    if g2 >= 50:
        return True
    return _has_recent_category(speech_tags, ["confusion", "concern"])


def _eval_acoustic_tension(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice tension + resistance. Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_tension" not in ac:
        return False
    return _g3_raw(g3) >= 40


def _eval_acoustic_roughness_proxy(
    g1: float, g2: float, g3: float, g4: float,
    hist: Dict[str, List[float]],
    speech_tags: List[SpeechTag],
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> bool:
    """Voice roughness proxy (strain/tension). Negative."""
    ac = set(acoustic_tags or [])
    if "acoustic_roughness_proxy" not in ac:
        return False
    return _g3_raw(g3) >= 35


_EVALUATORS: Dict[str, Any] = {
    "decision_readiness_multimodal": _eval_decision_readiness_multimodal,
    "cognitive_overload_multimodal": _eval_cognitive_overload_multimodal,
    "skepticism_objection_multimodal": _eval_skepticism_objection_multimodal,
    "aha_insight_multimodal": _eval_aha_insight_multimodal,
    "disengagement_multimodal": _eval_disengagement_multimodal,
    "confusion_multimodal": _eval_confusion_multimodal,
    "tension_objection_multimodal": _eval_tension_objection_multimodal,
    "loss_of_interest": _eval_loss_of_interest,
    "acoustic_disengagement_risk": _eval_acoustic_disengagement_risk,
    "acoustic_uncertainty": _eval_acoustic_uncertainty,
    "acoustic_tension": _eval_acoustic_tension,
    "acoustic_roughness_proxy": _eval_acoustic_roughness_proxy,
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
    recent_speech_tags: Optional[List[SpeechTag]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
    acoustic_tags: Optional[List[str]] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Evaluate B2B opportunity features in priority order. Returns (opportunity_id, context) for the
    first opportunity that fires and has passed cooldown, else None.

    Args:
        group_means: Current G1, G2, G3, G4 (0–100).
        group_history: Optional dict of deques (g1, g2, g3, g4) for temporal logic.
        signifier_scores: Optional 30 signifier scores.
        now: Timestamp for cooldown; defaults to time.time().
        recent_speech_tags: Optional list of recent speech tags for multimodal composites.
        composite_metrics: Optional composite metrics (confusion_multimodal, tension_objection_multimodal, etc.).
        acoustic_tags: Optional list of acoustic tags (e.g. from get_recent_acoustic_tags()).

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
    speech_tags = recent_speech_tags if recent_speech_tags is not None else []
    comp = composite_metrics
    ac_tags = acoustic_tags if acoustic_tags is not None else []

    for oid in OPPORTUNITY_PRIORITY:
        if not _check_cooldown(oid, now):
            continue
        fn = _EVALUATORS.get(oid)
        if fn is None:
            continue
        try:
            if oid in COMPOSITE_ACOUSTIC_OPPORTUNITY_IDS:
                fired = fn(g1, g2, g3, g4, hist, speech_tags, comp, ac_tags)
            elif oid in MULTIMODAL_OPPORTUNITY_IDS:
                fired = fn(g1, g2, g3, g4, hist, speech_tags)
            elif oid in ("trust_building_moment", "attention_peak", "rapport_moment"):
                fired = fn(g1, g2, g3, g4, hist, sig)
            else:
                fired = fn(g1, g2, g3, g4, hist)
            if fired:
                _fire(oid, now)
                context = {
                    "g1": g1, "g2": g2, "g3": g3, "g4": g4,
                    "signifier_scores": sig,
                    "recent_speech_tags": speech_tags,
                    "composite_metrics": comp,
                    "acoustic_tags": ac_tags,
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
