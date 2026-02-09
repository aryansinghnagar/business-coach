"""
Acoustic interpretation layer for engagement context.

Maps acoustic feature windows (pitch, loudness, tone) to natural-language
engagement context using research-backed heuristics from vocal affect literature.

Research citations:
- Scherer, Ladd: pitch variability in semitones; intonation contours (question vs statement)
- Bachorowski: loudness, arousal, disengagement/withdrawal
- Gobl, Segalowitz: vocal tension, roughness
- Shennong (2023): creaky voice, breathiness proxies
- CREPE (Kim et al.): robust pitch + voicing confidence (down-weight when low)
- MultiMediate (2023): multimodal engagement from voice

Voicing confidence: CREPE confidence < 0.5 down-weights pitch-based heuristics.
"""

from typing import Any, Dict, List, Optional, Tuple


# Default thresholds (tunable); values are normalized or in expected ranges
# from client: loudness_norm 0-1, pitch_hz 80-400 typical speech
LOUDNESS_LOW = 0.15
LOUDNESS_HIGH = 0.6
PITCH_VARIABILITY_HIGH_ST = 2.0  # semitones std as "high" variability (Scherer, Ladd)
PITCH_VARIABILITY_LOW_ST = 0.5   # very low variability -> monotone (Scherer/Ladd)
VOICING_CONFIDENCE_THRESHOLD = 0.5  # below this, pitch-based tags are down-weighted
TONE_ROUGHNESS_THRESHOLD = 0.65  # high tone proxy + variability -> roughness proxy
PITCH_CREAKY_PROXY_HZ = 120      # low pitch + low variability -> creakiness proxy (Shennong)
PITCH_RISING = "rising"
PITCH_FALLING = "falling"
PITCH_FLAT = "flat"


def interpret_acoustic_windows(
    windows: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Turn a list of acoustic feature windows into a short NL summary and optional tags.

    Research grounding:
    - High pitch variability + rising contour -> uncertainty/questioning (Scherer, Ladd)
    - High loudness + high pitch -> elevated arousal, emphasis (Bachorowski)
    - Low energy + flat pitch -> disengagement/withdrawal (Bachorowski)
    - Tense tone proxy + high pitch -> objection/stress (Gobl, Ladd)
    - Voicing confidence (CREPE): down-weight pitch heuristics when confidence is low (Kim et al.)
    - Roughness proxy: high tone + high variability approximates jitter/shimmer strain (Gobl, Shennong)
    - Monotone: very low pitch variability + flat contour (Scherer, Ladd semitone bands)
    - Creakiness proxy: low pitch + low variability (Shennong 2023; coarse spectrum ideal)
    - Emphasis: high loudness + rising contour (Bachorowski)

    Args:
        windows: List of dicts with keys e.g. loudness_norm, pitch_hz, pitch_contour,
                 pitch_variability, tone_proxy, voicing_confidence, speech_active.
                 Missing keys are ignored.

    Returns:
        (summary_string, list of tags e.g. acoustic_uncertainty, acoustic_arousal_high)
    """
    if not windows:
        return "", []

    # Use only windows where speech was active
    active = [w for w in windows if w.get("speech_active", True)]
    if not active:
        return "", []

    tags: List[str] = []
    parts: List[str] = []

    # Averages over active windows
    loudness_vals = [float(w["loudness_norm"]) for w in active if "loudness_norm" in w and w["loudness_norm"] is not None]
    pitch_vals = [float(w["pitch_hz"]) for w in active if "pitch_hz" in w and w["pitch_hz"] and w["pitch_hz"] > 0]
    contours = [w.get("pitch_contour") for w in active if w.get("pitch_contour")]
    variability_vals = [float(w["pitch_variability"]) for w in active if "pitch_variability" in w and w["pitch_variability"] is not None]
    tone_vals = [float(w["tone_proxy"]) for w in active if "tone_proxy" in w and w["tone_proxy"] is not None]
    voicing_vals = [float(w["voicing_confidence"]) for w in active if "voicing_confidence" in w and w["voicing_confidence"] is not None]

    avg_loudness = sum(loudness_vals) / len(loudness_vals) if loudness_vals else 0.0
    avg_pitch = sum(pitch_vals) / len(pitch_vals) if pitch_vals else None
    avg_variability = sum(variability_vals) / len(variability_vals) if variability_vals else 0.0
    avg_tone = sum(tone_vals) / len(tone_vals) if tone_vals else None  # higher can mean tenser/brighter
    avg_voicing = sum(voicing_vals) / len(voicing_vals) if voicing_vals else None

    # When voicing confidence is low, pitch-based heuristics are less reliable
    pitch_trusted = avg_voicing is None or avg_voicing >= VOICING_CONFIDENCE_THRESHOLD

    rising_count = sum(1 for c in contours if c == PITCH_RISING)
    falling_count = sum(1 for c in contours if c == PITCH_FALLING)
    n_contours = len(contours)

    # --- Disengagement / withdrawal: low energy + flat pitch
    if avg_loudness < LOUDNESS_LOW and (not contours or (falling_count + rising_count) < n_contours / 2):
        parts.append("Low vocal energy and relatively flat pitch suggest possible disengagement or withdrawal; consider re-engaging.")
        tags.append("acoustic_disengagement_risk")

    # --- Elevated arousal / emphasis: high loudness and pitch (pitch trusted)
    if pitch_trusted and avg_loudness >= LOUDNESS_HIGH and avg_pitch and avg_pitch > 180:
        parts.append("Elevated loudness and higher pitch indicate heightened arousal; good moment for emphasis or closing.")
        tags.append("acoustic_arousal_high")

    # --- Uncertainty / questioning: high pitch variability + rising contour (pitch trusted)
    if pitch_trusted and (avg_variability >= PITCH_VARIABILITY_HIGH_ST or (n_contours and rising_count >= max(1, n_contours // 2))):
        parts.append("Rising pitch contour and/or high pitch variability suggest possible uncertainty or questioning; consider clarifying.")
        tags.append("acoustic_uncertainty")

    # --- Vocal tension / objection: tense tone proxy + elevated pitch (pitch trusted)
    if pitch_trusted and avg_tone is not None and avg_tone > 0.6 and avg_pitch and avg_pitch > 160:
        parts.append("Vocal tension and elevated pitch may signal objection or stress; acknowledge and address.")
        tags.append("acoustic_tension")

    # --- Roughness proxy: high tone + high variability (jitter/shimmer-inspired)
    if avg_tone is not None and avg_tone > TONE_ROUGHNESS_THRESHOLD and avg_variability >= PITCH_VARIABILITY_HIGH_ST:
        parts.append("Vocal roughness proxy suggests strain; may indicate tension or discomfort.")
        tags.append("acoustic_roughness_proxy")

    # --- Finality / closure: falling contour (pitch trusted)
    if pitch_trusted and n_contours and falling_count >= max(1, n_contours // 2):
        if "acoustic_uncertainty" not in tags:
            parts.append("Falling pitch contour can signal finality or resolution; watch for closing cues.")
        tags.append("acoustic_falling_contour")

    # --- Monotone: very low pitch variability + flat contour (Scherer, Ladd)
    if pitch_trusted and avg_variability < PITCH_VARIABILITY_LOW_ST and n_contours:
        flat_count = n_contours - rising_count - falling_count
        if flat_count >= max(1, n_contours // 2):
            parts.append("Very flat pitch contour suggests monotone delivery; may indicate disengagement or cognitive load.")
            tags.append("acoustic_monotone")

    # --- Emphasis proxy: high loudness + rising contour (Bachorowski)
    if pitch_trusted and avg_loudness >= LOUDNESS_HIGH and n_contours and rising_count >= max(1, n_contours // 2):
        if "acoustic_arousal_high" not in tags:
            tags.append("acoustic_emphasis_proxy")

    # --- Creakiness proxy: low pitch + low variability (Shennong; coarse spectrum would be ideal)
    if pitch_trusted and avg_pitch and avg_pitch < PITCH_CREAKY_PROXY_HZ and avg_variability < PITCH_VARIABILITY_LOW_ST:
        parts.append("Low pitch with minimal variation may suggest creaky voice; can signal hesitation or low energy.")
        tags.append("acoustic_creakiness_proxy")

    # --- Breathiness proxy: high tone + low pitch stability (spectral centroid ideal; proxy with tone+variability)
    if avg_tone is not None and avg_tone > 0.7 and avg_variability < 1.0 and pitch_trusted:
        tags.append("acoustic_breathiness_proxy")

    # --- Speech rate: if client sends words_per_sec or pause_density (optional)
    words_per_sec_vals = [float(w["words_per_sec"]) for w in active if "words_per_sec" in w and w["words_per_sec"] is not None]
    if words_per_sec_vals:
        avg_wps = sum(words_per_sec_vals) / len(words_per_sec_vals)
        if avg_wps > 3.0:
            tags.append("acoustic_speech_rate_high")
        elif avg_wps < 1.5:
            tags.append("acoustic_speech_rate_low")

    summary = " ".join(parts).strip() if parts else ""
    return summary, tags
