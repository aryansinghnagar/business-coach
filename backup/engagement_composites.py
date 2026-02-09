"""
Engagement Composite Metrics (Facial + Speech)

Psychology-based composite scores (0-100) combining facial signifiers/group means
with recent speech tags. Research: Cialdini (consistency), Kahneman (cognitive load),
rapport and decision readiness in B2B contexts.

Used by EngagementStateDetector; results attached to EngagementState.composite_metrics.

Mathematical logic
------------------
All outputs are clamped to [0, 100]. G1, G2, G3, G4 are group means (0-100); G3 is
"low resistance" (high G3 = good). Speech strengths are 0-1 from _category_strength():
  strength = min(1, 0.5 * recency + 0.5 * count_norm), where recency is averaged
  (1 - (now - t)/window_sec) over matching tags in the last window_sec (12s), and
  count_norm = min(1, len(matches)/3).

  verbal_nonverbal_alignment = (commit_interest * 0.6 + face_positive * 0.4) * 100
    with face_positive = (G4*0.5 + G1*0.5) / 100

  cognitive_load_multimodal = (g2_norm * 0.55 + confusion_concern * 0.45) * 100
    with g2_norm = G2/100

  rapport_engagement = (G1/100*0.4 + interest_realization*0.35 + G3/100*0.25) * 100

  skepticism_objection_strength = (objection_concern*0.6 + (100-G3)/100*0.4) * 100

  decision_readiness_multimodal = (G4/100*0.55 + commit_interest*0.45) * 100
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

# Speech tag: {"category": str, "phrase": str, "time": float}
SpeechTag = Dict[str, Any]


def _has_category(tags: List[SpeechTag], categories: List[str]) -> bool:
    if not tags or not categories:
        return False
    cats = set(categories)
    return any((t.get("category") or "") in cats for t in tags)


def _category_strength(tags: List[SpeechTag], categories: List[str], window_sec: float = 12.0) -> float:
    """Return 0-1 strength from recency and count of matching tags in window."""
    if not tags or not categories:
        return 0.0
    now = time.time()
    cutoff = now - window_sec
    cats = set(categories)
    matches = [t for t in tags if t.get("time", 0) >= cutoff and (t.get("category") or "") in cats]
    if not matches:
        return 0.0
    # More recent + more matches = higher strength
    recency = sum(1.0 - (now - t.get("time", 0)) / window_sec for t in matches) / max(len(matches), 1)
    count_norm = min(1.0, len(matches) / 3.0)
    return min(1.0, 0.5 * recency + 0.5 * count_norm)


def compute_composite_metrics(
    group_means: Dict[str, float],
    signifier_scores: Optional[Dict[str, float]] = None,
    speech_tags: Optional[List[SpeechTag]] = None,
    composite_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Compute 0-100 composite metrics from facial (G1-G4, signifiers) and speech tags.

    composite_weights: optional dict of composite_key -> { "speech": w, "face": w } or similar
    to override default formula weights (e.g. from config or PUT /api/weights/insight).
    If not provided, default formula weights are used.

    Returns dict with keys:
      - verbal_nonverbal_alignment: commitment/interest speech + high G4/G1 (Cialdini)
      - cognitive_load_multimodal: high G2 + confusion/concern speech (Kahneman)
      - rapport_engagement: G1 + interest/realization speech + low resistance G3
      - skepticism_objection_strength: objection/concern speech + high G3 resistance
      - decision_readiness_multimodal: high G4 + commitment/interest speech
      - opportunity_strength: decision_readiness_multimodal + verbal_nonverbal_alignment (closing moments)
    """
    out: Dict[str, float] = {}
    g1 = group_means.get("g1", 50.0)
    g2 = group_means.get("g2", 50.0)
    g3 = group_means.get("g3", 50.0)  # high = low resistance
    g4 = group_means.get("g4", 50.0)
    tags = speech_tags or []
    cw = composite_weights or {}

    # Speech strengths (0-1)
    commit_interest = _category_strength(tags, ["commitment", "interest"])
    confusion_concern = _category_strength(tags, ["confusion", "concern"])
    objection_concern = _category_strength(tags, ["objection", "concern"])
    interest_realization = _category_strength(tags, ["interest", "realization"])

    # Verbal-nonverbal alignment: words + face agree (commitment/interest + G4/G1)
    wa = cw.get("verbal_nonverbal_alignment", {})
    w_align_speech = wa.get("speech", 0.6)
    w_align_face = wa.get("face", 0.4)
    face_positive = (g4 * 0.5 + g1 * 0.5) / 100.0
    align_raw = (commit_interest * w_align_speech + face_positive * w_align_face) * 100.0
    out["verbal_nonverbal_alignment"] = float(np.clip(align_raw, 0.0, 100.0))

    # Cognitive load (multimodal): G2 high + confusion/concern speech
    w_load = cw.get("cognitive_load_multimodal", {})
    w_g2 = w_load.get("g2", 0.55)
    w_conf = w_load.get("speech", 0.45)
    g2_norm = g2 / 100.0
    load_raw = (g2_norm * w_g2 + confusion_concern * w_conf) * 100.0
    out["cognitive_load_multimodal"] = float(np.clip(load_raw, 0.0, 100.0))

    # Rapport: G1 + interest/realization + low resistance (G3 high = good)
    w_rapport = cw.get("rapport_engagement", {})
    w_g1_r = w_rapport.get("g1", 0.4)
    w_ir = w_rapport.get("speech", 0.35)
    w_g3_r = w_rapport.get("g3", 0.25)
    g3_norm = g3 / 100.0
    rapport_raw = (g1 / 100.0 * w_g1_r + interest_realization * w_ir + g3_norm * w_g3_r) * 100.0
    out["rapport_engagement"] = float(np.clip(rapport_raw, 0.0, 100.0))

    # Skepticism/objection: objection/concern speech + resistance (G3 low = high resistance)
    w_sk = cw.get("skepticism_objection_strength", {})
    w_obj = w_sk.get("speech", 0.6)
    w_res = w_sk.get("resistance", 0.4)
    g3_resistance = 100.0 - g3  # high = more resistance
    skept_raw = (objection_concern * w_obj + (g3_resistance / 100.0) * w_res) * 100.0
    out["skepticism_objection_strength"] = float(np.clip(skept_raw, 0.0, 100.0))

    # Decision readiness (multimodal): G4 + commitment/interest speech
    w_ready = cw.get("decision_readiness_multimodal", {})
    w_g4 = w_ready.get("g4", 0.55)
    w_ci = w_ready.get("speech", 0.45)
    ready_raw = (g4 / 100.0 * w_g4 + commit_interest * w_ci) * 100.0
    out["decision_readiness_multimodal"] = float(np.clip(ready_raw, 0.0, 100.0))

    # Opportunity strength: for closing moments; combines decision readiness + verbal-nonverbal alignment
    dr = out["decision_readiness_multimodal"] / 100.0
    vn = out["verbal_nonverbal_alignment"] / 100.0
    w_opp = cw.get("opportunity_strength", {})
    w_dr = w_opp.get("decision_readiness", 0.55)
    w_vn = w_opp.get("verbal_nonverbal", 0.45)
    opp_raw = (dr * w_dr + vn * w_vn) * 100.0
    out["opportunity_strength"] = float(np.clip(opp_raw, 0.0, 100.0))

    return out
