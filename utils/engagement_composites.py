"""
Engagement Composite Metrics (Facial + Speech + Acoustic)

Psychology-based composite scores (0-100) combining facial signifiers/group means,
recent speech tags, and optional acoustic tags. Research: Cialdini (consistency),
Kahneman (cognitive load), Scherer/Bachorowski (vocal affect), Edmondson (psychological
safety), rapport and decision readiness in B2B contexts.

Used by EngagementStateDetector; results attached to EngagementState.composite_metrics.

Mathematical logic
------------------
All outputs are clamped to [0, 100]. G1, G2, G3, G4 are group means (0-100); G3 is
"low resistance" (high G3 = good). Speech strengths are 0-1 from _category_strength().
Acoustic: acoustic_tags (e.g. acoustic_uncertainty, acoustic_tension, acoustic_disengagement_risk)
and optional acoustic_negative_strength (0-1) boost relevant composites.

  cognitive_load_multimodal: G2 + confusion/concern speech + acoustic_uncertainty (Kahneman System 2)
  skepticism_objection_strength: objection/concern + resistance + acoustic_tension (Ekman, Cialdini)
  disengagement_risk_multimodal: low G1 + no commit + resistance + acoustic_disengagement_risk
  confusion_multimodal: G2 + confusion/concern speech + acoustic_uncertainty
  tension_objection_multimodal: resistance + objection/concern + acoustic_tension (Gobl)
  loss_of_interest_multimodal: (100-G1) + (1-commit_interest) + acoustic_disengagement_risk (Mehrabian)
  decision_plus_voice: decision_readiness + acoustic_falling_contour/arousal_high (Cialdini)
  psychological_safety_proxy: Low G3 + interest/confirmation + low tension (Edmondson)
  urgency_sensitivity: timeline/budget speech + arousal + G4
  skepticism_strength: skepticism speech + G3 resistance + tension/roughness
  enthusiasm_multimodal: enthusiasm speech + G1 + acoustic_arousal_high
  hesitation_multimodal: hesitation speech + G2 + acoustic_uncertainty (Kahneman)
  authority_deferral: authority speech + G2 + acoustic_monotone
  rapport_depth: G1 + interest/confirmation + acoustic_falling_contour
  cognitive_overload_proxy: G2 + confusion + acoustic_uncertainty + stillness
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

# Speech tag: {"category": str, "phrase": str, "time": float, "discourse_boost": bool?}
SpeechTag = Dict[str, Any]


def _has_category(tags: List[SpeechTag], categories: List[str]) -> bool:
    if not tags or not categories:
        return False
    cats = set(categories)
    return any((t.get("category") or "") in cats for t in tags)


def _category_strength(tags: List[SpeechTag], categories: List[str], window_sec: float = 12.0) -> float:
    """Return 0-1 strength from recency and count of matching tags in window. discourse_boost adds +0.1."""
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
    base = min(1.0, 0.5 * recency + 0.5 * count_norm)
    # discourse_boost: align discourse marker with category -> +0.1 strength
    boost = 0.1 if any(t.get("discourse_boost") for t in matches) else 0.0
    return min(1.0, base + boost)


def compute_composite_metrics(
    group_means: Dict[str, float],
    signifier_scores: Optional[Dict[str, float]] = None,
    speech_tags: Optional[List[SpeechTag]] = None,
    composite_weights: Optional[Dict[str, Dict[str, float]]] = None,
    acoustic_tags: Optional[List[str]] = None,
    acoustic_negative_strength: float = 0.0,
) -> Dict[str, float]:
    """
    Compute 0-100 composite metrics from facial (G1-G4), speech tags, and optional acoustic.

    acoustic_tags: e.g. from get_recent_acoustic_tags(); used to boost confusion, tension, disengagement.
    acoustic_negative_strength: 0-1 from get_acoustic_negative_strength() (optional).

    Returns dict with keys including verbal_nonverbal_alignment, cognitive_load_multimodal,
    rapport_engagement, skepticism_objection_strength, decision_readiness_multimodal,
    opportunity_strength, trust_rapport, disengagement_risk_multimodal,
    confusion_multimodal, tension_objection_multimodal, loss_of_interest_multimodal,
    decision_plus_voice.
    """
    out: Dict[str, float] = {}
    g1 = group_means.get("g1", 50.0)
    g2 = group_means.get("g2", 50.0)
    g3 = group_means.get("g3", 50.0)  # high = low resistance
    g4 = group_means.get("g4", 50.0)
    tags = speech_tags or []
    cw = composite_weights or {}
    ac_tags = set(acoustic_tags or [])

    # Speech strengths (0-1)
    commit_interest = _category_strength(tags, ["commitment", "interest"])
    confusion_concern = _category_strength(tags, ["confusion", "concern"])
    objection_concern = _category_strength(tags, ["objection", "concern"])
    interest_realization = _category_strength(tags, ["interest", "realization"])
    interest_confirmation = _category_strength(tags, ["interest", "confirmation"])
    timeline_budget = _category_strength(tags, ["timeline", "budget"])
    skepticism_speech = _category_strength(tags, ["skepticism"])
    enthusiasm_speech = _category_strength(tags, ["enthusiasm"])
    hesitation_speech = _category_strength(tags, ["hesitation"])
    authority_speech = _category_strength(tags, ["authority"])

    # Acoustic boosts (0-1) for blending into composites
    has_uncertainty = 0.25 if "acoustic_uncertainty" in ac_tags else 0.0
    has_tension = 0.25 if "acoustic_tension" in ac_tags else 0.0
    has_disengagement = 0.25 if "acoustic_disengagement_risk" in ac_tags else 0.0
    has_roughness = 0.2 if "acoustic_roughness_proxy" in ac_tags else 0.0
    has_falling = 0.15 if "acoustic_falling_contour" in ac_tags else 0.0
    has_arousal_high = 0.15 if "acoustic_arousal_high" in ac_tags else 0.0
    has_monotone = 0.2 if "acoustic_monotone" in ac_tags else 0.0

    # Verbal-nonverbal alignment: words + face agree (commitment/interest + G4/G1)
    wa = cw.get("verbal_nonverbal_alignment", {})
    w_align_speech = wa.get("speech", 0.6)
    w_align_face = wa.get("face", 0.4)
    face_positive = (g4 * 0.5 + g1 * 0.5) / 100.0
    align_raw = (commit_interest * w_align_speech + face_positive * w_align_face) * 100.0
    out["verbal_nonverbal_alignment"] = float(np.clip(align_raw, 0.0, 100.0))

    # Cognitive load (multimodal): G2 + confusion/concern speech + acoustic_uncertainty
    w_load = cw.get("cognitive_load_multimodal", {})
    w_g2 = w_load.get("g2", 0.55)
    w_conf = w_load.get("speech", 0.45)
    g2_norm = g2 / 100.0
    load_raw = (g2_norm * w_g2 + confusion_concern * w_conf) * 100.0
    load_raw = min(100.0, load_raw + has_uncertainty * 100.0)
    out["cognitive_load_multimodal"] = float(np.clip(load_raw, 0.0, 100.0))

    # Rapport: G1 + interest/realization + low resistance (G3 high = good)
    w_rapport = cw.get("rapport_engagement", {})
    w_g1_r = w_rapport.get("g1", 0.4)
    w_ir = w_rapport.get("speech", 0.35)
    w_g3_r = w_rapport.get("g3", 0.25)
    g3_norm = g3 / 100.0
    rapport_raw = (g1 / 100.0 * w_g1_r + interest_realization * w_ir + g3_norm * w_g3_r) * 100.0
    out["rapport_engagement"] = float(np.clip(rapport_raw, 0.0, 100.0))

    # Skepticism/objection: objection/concern speech + resistance + acoustic_tension + acoustic_roughness
    w_sk = cw.get("skepticism_objection_strength", {})
    w_obj = w_sk.get("speech", 0.6)
    w_res = w_sk.get("resistance", 0.4)
    g3_resistance = 100.0 - g3  # high = more resistance
    skept_raw = (objection_concern * w_obj + (g3_resistance / 100.0) * w_res) * 100.0
    skept_raw = min(100.0, skept_raw + (has_tension + has_roughness) * 100.0)
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

    # Trust/rapport: G1 (interest face) + interest/realization speech + low resistance (G3 high)
    w_tr = cw.get("trust_rapport", {})
    w_g1_tr = w_tr.get("g1", 0.5)
    w_ir_tr = w_tr.get("speech", 0.3)
    w_g3_tr = w_tr.get("g3", 0.2)
    trust_raw = (g1 / 100.0 * w_g1_tr + interest_realization * w_ir_tr + g3_norm * w_g3_tr) * 100.0
    out["trust_rapport"] = float(np.clip(trust_raw, 0.0, 100.0))

    # Disengagement risk (multimodal): low G1 + no positive speech + resistance + acoustic_disengagement_risk
    g3_res = (100.0 - g3) / 100.0
    no_commit = 1.0 - commit_interest
    w_dis = cw.get("disengagement_risk_multimodal", {})
    w_g1_dis = w_dis.get("g1_low", 0.35)
    w_nocommit = w_dis.get("no_commit", 0.35)
    w_res_dis = w_dis.get("resistance", 0.30)
    dis_raw = ((100.0 - g1) / 100.0 * w_g1_dis + no_commit * w_nocommit + g3_res * w_res_dis) * 100.0
    dis_raw = min(100.0, dis_raw + has_disengagement * 100.0)
    out["disengagement_risk_multimodal"] = float(np.clip(dis_raw, 0.0, 100.0))

    # Confusion (multimodal): G2 + confusion/concern speech + acoustic_uncertainty (Kahneman + vocal affect)
    conf_raw = (g2 / 100.0 * 0.45 + confusion_concern * 0.40) * 100.0 + has_uncertainty * 25.0
    out["confusion_multimodal"] = float(np.clip(conf_raw, 0.0, 100.0))

    # Tension/objection (multimodal): resistance + objection/concern + acoustic_tension + roughness (Ekman, Cialdini, Gobl)
    g3_res_norm = (100.0 - g3) / 100.0
    tension_raw = (g3_res_norm * 0.45 + objection_concern * 0.40) * 100.0 + (has_tension + has_roughness) * 25.0
    out["tension_objection_multimodal"] = float(np.clip(tension_raw, 0.0, 100.0))

    # Loss of interest (multimodal): low G1 + no commitment language + acoustic withdrawal (Mehrabian, Driskell)
    loss_g1 = (100.0 - g1) / 100.0
    loss_raw = (loss_g1 * 0.45 + no_commit * 0.40) * 100.0 + has_disengagement * 25.0 + acoustic_negative_strength * 15.0
    out["loss_of_interest_multimodal"] = float(np.clip(loss_raw, 0.0, 100.0))

    # Decision plus voice: decision_readiness + acoustic closure/arousal (Cialdini + vocal readiness)
    dr_val = out["decision_readiness_multimodal"] / 100.0
    voice_boost = min(0.2, has_falling + has_arousal_high)
    out["decision_plus_voice"] = float(np.clip((dr_val + voice_boost) * 100.0, 0.0, 100.0))

    # --- New composites (Part 5 of plan) ---

    # Psychological safety proxy (Edmondson): Low G3 + interest/confirmation + low tension
    low_tension = 1.0 - (has_tension + has_roughness)
    safety_raw = (g3 / 100.0 * 0.45 + interest_confirmation * 0.35 + low_tension * 0.2) * 100.0
    out["psychological_safety_proxy"] = float(np.clip(safety_raw, 0.0, 100.0))

    # Urgency sensitivity: timeline/budget speech + arousal + G4
    urgency_raw = (timeline_budget * 0.4 + (has_arousal_high * 4.0) * 0.3 + g4 / 100.0 * 0.3) * 100.0
    out["urgency_sensitivity"] = float(np.clip(urgency_raw, 0.0, 100.0))

    # Skepticism strength: skepticism speech + resistance + tension/roughness
    skept2_raw = (skepticism_speech * 0.5 + g3_resistance / 100.0 * 0.3) * 100.0
    skept2_raw = min(100.0, skept2_raw + (has_tension + has_roughness) * 50.0)
    out["skepticism_strength"] = float(np.clip(skept2_raw, 0.0, 100.0))

    # Enthusiasm multimodal: enthusiasm speech + G1 + acoustic_arousal_high
    enth_raw = (enthusiasm_speech * 0.4 + g1 / 100.0 * 0.4 + has_arousal_high * 2.0 * 0.2) * 100.0
    out["enthusiasm_multimodal"] = float(np.clip(enth_raw, 0.0, 100.0))

    # Hesitation multimodal: hesitation speech + G2 + acoustic_uncertainty
    hes_raw = (hesitation_speech * 0.4 + g2 / 100.0 * 0.4 + has_uncertainty * 2.0 * 0.2) * 100.0
    out["hesitation_multimodal"] = float(np.clip(hes_raw, 0.0, 100.0))

    # Authority deferral: authority speech + G2 + acoustic_monotone
    auth_raw = (authority_speech * 0.5 + g2 / 100.0 * 0.3 + has_monotone * 2.0 * 0.2) * 100.0
    out["authority_deferral"] = float(np.clip(auth_raw, 0.0, 100.0))

    # Rapport depth: G1 + interest/confirmation + acoustic_falling_contour
    rap_raw = (g1 / 100.0 * 0.45 + interest_confirmation * 0.35 + has_falling * 2.0 * 0.2) * 100.0
    out["rapport_depth"] = float(np.clip(rap_raw, 0.0, 100.0))

    # Cognitive overload proxy: G2 + confusion speech + acoustic_uncertainty + stillness
    still_norm = (signifier_scores or {}).get("g2_stillness", 50.0) / 100.0
    overload_raw = (g2 / 100.0 * 0.35 + confusion_concern * 0.35 + has_uncertainty * 2.0 * 0.15 + still_norm * 0.15) * 100.0
    out["cognitive_overload_proxy"] = float(np.clip(overload_raw, 0.0, 100.0))

    return out
