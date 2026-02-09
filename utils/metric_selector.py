"""
Dynamic Metric Selection Subroutine.

Returns the active metric set (signifier_keys, speech_categories, acoustic_tags,
composite_keys) based on system resources: CPU cores, RAM, and optional network
latency. Enables the application to scale metric computation to available
compute power and memory.

Tiers:
  - High (4+ cores, 8+ GB RAM): All metrics enabled (up to 100)
  - Medium (2-4 cores, 4-8 GB RAM): ~75 metrics
  - Low (1-2 cores, <4 GB RAM): ~50 metrics

Uses psutil for CPU/RAM when available; falls back to os.cpu_count() and Medium tier.
"""

from dataclasses import dataclass
from typing import List, Optional

# -----------------------------------------------------------------------------
# Full metric lists (all available metrics; tiers select subsets)
# -----------------------------------------------------------------------------
def _get_full_signifier_keys() -> List[str]:
    from utils.expression_signifiers import SIGNIFIER_KEYS
    return list(SIGNIFIER_KEYS)

# Must stay in sync with services.insight_generator.PHRASE_CATEGORIES keys
FULL_SPEECH_CATEGORIES: List[str] = [
    "objection", "interest", "confusion", "commitment", "concern",
    "timeline", "budget", "realization",
    "urgency", "skepticism", "enthusiasm", "authority", "hesitation", "confirmation",
]

# All acoustic tags from utils.acoustic_interpreter
FULL_ACOUSTIC_TAGS: List[str] = [
    "acoustic_disengagement_risk", "acoustic_arousal_high", "acoustic_uncertainty",
    "acoustic_tension", "acoustic_roughness_proxy", "acoustic_falling_contour",
    "acoustic_monotone", "acoustic_emphasis_proxy", "acoustic_creakiness_proxy",
    "acoustic_breathiness_proxy", "acoustic_speech_rate_high", "acoustic_speech_rate_low",
]

# All composite metrics from utils.engagement_composites
FULL_COMPOSITE_KEYS: List[str] = [
    "verbal_nonverbal_alignment", "cognitive_load_multimodal", "rapport_engagement",
    "skepticism_objection_strength", "decision_readiness_multimodal", "opportunity_strength",
    "trust_rapport", "disengagement_risk_multimodal", "confusion_multimodal",
    "tension_objection_multimodal", "loss_of_interest_multimodal", "decision_plus_voice",
    "psychological_safety_proxy", "urgency_sensitivity", "skepticism_strength",
    "enthusiasm_multimodal", "hesitation_multimodal", "authority_deferral",
    "rapport_depth", "cognitive_overload_proxy",
]

# Signifiers to drop in Medium tier (heavier / less critical)
MEDIUM_DROP_SIGNIFIERS: List[str] = [
    "g1_pupil_dilation", "g1_eyebrow_flash", "g1_facial_symmetry", "g1_softened_forehead",
    "g1_micro_smile", "g1_brow_raise_sustained", "g1_eye_widening",
    "g2_chin_stroke", "g2_stillness", "g2_brow_furrow_deep", "g2_gaze_shift_frequency",
    "g3_no_nod", "g3_narrowed_pupils", "g3_mouth_cover", "g3_lip_corner_dip", "g3_brow_lower_sustained",
]

# Signifiers to keep in Low tier (core G1/G3/G4 only)
LOW_KEEP_SIGNIFIERS: List[str] = [
    "g1_duchenne", "g1_eye_contact", "g1_head_tilt", "g1_forward_lean",
    "g1_rhythmic_nodding", "g1_parted_lips", "g1_mouth_open_receptive", "g1_nod_intensity",
    "g2_look_up_lr", "g2_eye_squint", "g2_thinking_brow", "g2_lowered_brow", "g2_mouth_tight_eval",
    "g3_contempt", "g3_lip_compression", "g3_eye_block", "g3_jaw_clench",
    "g3_rapid_blink", "g3_gaze_aversion", "g3_eye_squeeze", "g3_head_shake",
    "g4_relaxed_exhale", "g4_fixed_gaze", "g4_smile_transition", "g4_mouth_relax", "g4_smile_sustain",
]

# Composites to drop in Medium tier
MEDIUM_DROP_COMPOSITES: List[str] = [
    "opportunity_strength", "trust_rapport", "urgency_sensitivity", "enthusiasm_multimodal",
    "rapport_depth", "psychological_safety_proxy",
]

# Composites to keep in Low tier
LOW_KEEP_COMPOSITES: List[str] = [
    "verbal_nonverbal_alignment", "cognitive_load_multimodal", "rapport_engagement",
    "skepticism_objection_strength", "decision_readiness_multimodal",
    "disengagement_risk_multimodal", "confusion_multimodal", "tension_objection_multimodal",
    "loss_of_interest_multimodal", "decision_plus_voice",
    "skepticism_strength", "hesitation_multimodal", "authority_deferral", "cognitive_overload_proxy",
]


@dataclass
class MetricConfig:
    """Active metric configuration based on system tier."""
    tier: str  # "high" | "medium" | "low"
    signifier_keys: List[str]
    speech_categories: List[str]
    acoustic_tags: List[str]
    composite_keys: List[str]


def _get_system_resources() -> tuple:
    """Return (cpu_cores, ram_gb). Uses psutil if available; else os.cpu_count(), None."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or 1
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)
        return (cores, ram_gb)
    except Exception:
        import os
        cores = os.cpu_count() or 2
        return (cores, None)


def _determine_tier(
    cpu_cores: int,
    ram_gb: Optional[float],
    override: Optional[str],
) -> str:
    """Return 'high', 'medium', or 'low' based on resources and override."""
    if override and override.lower() in ("high", "medium", "low"):
        return override.lower()
    # High: 4+ cores, 8+ GB
    if cpu_cores >= 4 and (ram_gb is None or ram_gb >= 8.0):
        return "high"
    # Low: 1-2 cores, or <4 GB
    if cpu_cores <= 2 or (ram_gb is not None and ram_gb < 4.0):
        return "low"
    # Medium: 2-4 cores, 4-8 GB
    return "medium"


def _build_config(tier: str) -> MetricConfig:
    """Build MetricConfig for the given tier."""
    full_signifiers = _get_full_signifier_keys()
    if tier == "high":
        return MetricConfig(
            tier="high",
            signifier_keys=list(full_signifiers),
            speech_categories=list(FULL_SPEECH_CATEGORIES),
            acoustic_tags=list(FULL_ACOUSTIC_TAGS),
            composite_keys=list(FULL_COMPOSITE_KEYS),
        )
    if tier == "medium":
        signifiers = [k for k in full_signifiers if k not in MEDIUM_DROP_SIGNIFIERS]
        composites = [k for k in FULL_COMPOSITE_KEYS if k not in MEDIUM_DROP_COMPOSITES]
        return MetricConfig(
            tier="medium",
            signifier_keys=signifiers,
            speech_categories=list(FULL_SPEECH_CATEGORIES),
            acoustic_tags=list(FULL_ACOUSTIC_TAGS),
            composite_keys=composites,
        )
    # low
    signifiers = [k for k in full_signifiers if k in LOW_KEEP_SIGNIFIERS]
    composites = [k for k in FULL_COMPOSITE_KEYS if k in LOW_KEEP_COMPOSITES]
    speech = list(FULL_SPEECH_CATEGORIES)[:8]  # all 8
    acoustic = list(FULL_ACOUSTIC_TAGS)[:6]    # all 6
    return MetricConfig(
        tier="low",
        signifier_keys=signifiers,
        speech_categories=speech,
        acoustic_tags=acoustic,
        composite_keys=composites,
    )


def get_active_metrics(
    cpu_cores: Optional[int] = None,
    ram_gb: Optional[float] = None,
    network_latency_ms: Optional[float] = None,
    override: Optional[str] = None,
) -> MetricConfig:
    """
    Return the active metric configuration based on system resources.

    Args:
        cpu_cores: Number of CPU cores (default: from psutil or os.cpu_count).
        ram_gb: RAM in GB (default: from psutil if available).
        network_latency_ms: Optional network latency; currently unused, reserved for future use.
        override: Force tier "high", "medium", or "low" (overrides auto-detection).

    Returns:
        MetricConfig with signifier_keys, speech_categories, acoustic_tags, composite_keys.
    """
    if cpu_cores is None or ram_gb is None:
        detected_cores, detected_ram = _get_system_resources()
        cpu_cores = cpu_cores if cpu_cores is not None else detected_cores
        ram_gb = ram_gb if ram_gb is not None else detected_ram

    tier = _determine_tier(cpu_cores, ram_gb, override)
    return _build_config(tier)


def get_active_metrics_with_config() -> MetricConfig:
    """
    Get active metrics using application config (METRIC_SELECTOR_ENABLED,
    METRIC_SELECTOR_OVERRIDE). Call from engagement_state_detector and other consumers.
    """
    try:
        import config
        if not getattr(config, "METRIC_SELECTOR_ENABLED", True):
            return _build_config("high")  # Full metrics when disabled
        override = getattr(config, "METRIC_SELECTOR_OVERRIDE", None)
        return get_active_metrics(override=override)
    except Exception:
        return _build_config("high")  # Fallback to full on error
