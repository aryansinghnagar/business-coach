"""
Configuration for Business Meeting Copilot.

Centralized settings for Azure AI Foundry, Speech, Face API, Cognitive Search,
face detection, avatar, and server. Override via environment
variables in production; never commit secrets. See docs/DOCUMENTATION.md.
"""

import os
from typing import Optional

# ============================================================================
# Azure AI Foundry Configuration (upgraded from Azure OpenAI)
# ============================================================================
# Env vars: AZURE_FOUNDRY_* (primary) or AZURE_OPENAI_* (fallback for backward compatibility).
# Sanitize to avoid 404: strip whitespace, normalize endpoint (no trailing slash).
def _sanitize_azure_foundry_config() -> None:
    global AZURE_FOUNDRY_KEY, AZURE_FOUNDRY_ENDPOINT, FOUNDRY_DEPLOYMENT_NAME, AZURE_FOUNDRY_API_VERSION
    AZURE_FOUNDRY_KEY = (AZURE_FOUNDRY_KEY or "").strip()
    FOUNDRY_DEPLOYMENT_NAME = (FOUNDRY_DEPLOYMENT_NAME or "gpt-4o").strip()
    AZURE_FOUNDRY_API_VERSION = (AZURE_FOUNDRY_API_VERSION or "2024-11-20").strip()
    ep = (AZURE_FOUNDRY_ENDPOINT or "").strip().rstrip("/")
    AZURE_FOUNDRY_ENDPOINT = ep if ep else ""


# No default secrets; set AZURE_FOUNDRY_KEY and AZURE_FOUNDRY_ENDPOINT (or AZURE_OPENAI_*) in env.
AZURE_FOUNDRY_KEY: str = (os.getenv("AZURE_FOUNDRY_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
AZURE_FOUNDRY_ENDPOINT: str = (os.getenv("AZURE_FOUNDRY_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
FOUNDRY_DEPLOYMENT_NAME: str = os.getenv("FOUNDRY_DEPLOYMENT_NAME") or os.getenv("DEPLOYMENT_NAME", "gpt-4o")
AZURE_FOUNDRY_API_VERSION: str = os.getenv("AZURE_FOUNDRY_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-20")
_sanitize_azure_foundry_config()

# Backward compatibility aliases (deprecated; use AZURE_FOUNDRY_* / FOUNDRY_DEPLOYMENT_NAME)
AZURE_OPENAI_KEY: str = AZURE_FOUNDRY_KEY
AZURE_OPENAI_ENDPOINT: str = AZURE_FOUNDRY_ENDPOINT
DEPLOYMENT_NAME: str = FOUNDRY_DEPLOYMENT_NAME
AZURE_OPENAI_API_VERSION: str = AZURE_FOUNDRY_API_VERSION

# ============================================================================
# Azure Speech Service Configuration
# ============================================================================
# Set SPEECH_KEY in env; no default secret. Key must be from a Speech resource (not Face/OpenAI).
def _strip_quotes(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


SPEECH_KEY: str = _strip_quotes(os.getenv("SPEECH_KEY") or "")
SPEECH_REGION: str = (os.getenv("SPEECH_REGION") or "centralindia").strip().lower()
SPEECH_PRIVATE_ENDPOINT_ENABLED: bool = os.getenv("SPEECH_PRIVATE_ENDPOINT_ENABLED", "false").lower() == "true"
SPEECH_PRIVATE_ENDPOINT: Optional[str] = os.getenv("SPEECH_PRIVATE_ENDPOINT", None)

# ============================================================================
# Azure Cognitive Search Configuration (Optional - for On Your Data)
# ============================================================================
AZURE_COG_SEARCH_ENDPOINT: str = os.getenv("AZURE_COG_SEARCH_ENDPOINT", "")
AZURE_COG_SEARCH_API_KEY: str = os.getenv("AZURE_COG_SEARCH_API_KEY", "")
AZURE_COG_SEARCH_INDEX_NAME: str = os.getenv("AZURE_COG_SEARCH_INDEX_NAME", "")

# ============================================================================
# Azure Face API Configuration (Optional - alternative to MediaPipe)
# ============================================================================
# Set AZURE_FACE_API_KEY and AZURE_FACE_API_ENDPOINT in env for Face API; no default secrets.
AZURE_FACE_API_KEY: str = (os.getenv("AZURE_FACE_API_KEY") or "").strip()
AZURE_FACE_API_ENDPOINT: str = (os.getenv("AZURE_FACE_API_ENDPOINT") or "").strip().rstrip("/")
AZURE_FACE_API_REGION: str = os.getenv("AZURE_FACE_API_REGION", "centralindia")

# ============================================================================
# Face Detection Configuration
# ============================================================================
# Options: "mediapipe" | "azure_face_api" | "auto" | "unified"
# auto (default): App chooses unified (MediaPipe+Azure) or mediapipe based on device tier + latency.
# mediapipe: Local only. unified: Both MediaPipe and Azure (fused). azure_face_api: Azure only.
FACE_DETECTION_METHOD: str = os.getenv("FACE_DETECTION_METHOD", "auto")

# Minimum confidence for face detection (0.01-0.9). Lower = more permissive in suboptimal lighting.
# Default 0.05 is very permissive to handle various webcam/lighting conditions.
MIN_FACE_CONFIDENCE: float = float(os.getenv("MIN_FACE_CONFIDENCE", "0.05"))

# Lightweight mode: MediaPipe only, reduced buffer, process every 2nd frame.
# Use on devices with less computational power for real-time processing.
LIGHTWEIGHT_MODE: bool = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"

# Target frame rate: at least TARGET_FPS_MIN (30), cap at TARGET_FPS_MAX (30 for real-time, no smoothing).
TARGET_FPS_MIN: float = float(os.getenv("TARGET_FPS_MIN", "30"))
TARGET_FPS_MAX: float = float(os.getenv("TARGET_FPS_MAX", "30"))

# Reserved: run engagement detection in a separate process (DETECTION_WORKER_PROCESS). Default False; in-thread only.
DETECTION_WORKER_PROCESS: bool = os.getenv("DETECTION_WORKER_PROCESS", "false").lower() == "true"

# Dynamic switching: when detection_method is "auto", choose Azure vs MediaPipe
# based on device tier and Azure/network latency.
AUTO_DETECTION_SWITCHING: bool = os.getenv("AUTO_DETECTION_SWITCHING", "true").lower() == "true"
# Latency threshold (ms): if Azure round-trip exceeds this, prefer MediaPipe.
AZURE_LATENCY_THRESHOLD_MS: float = float(os.getenv("AZURE_LATENCY_THRESHOLD_MS", "500"))

# Unified mode: fuse Azure emotions + MediaPipe landmarks/signifiers when both are used.
# Weights for fused score: azure_weight * azure_score + mediapipe_weight * mediapipe_score.
FUSION_AZURE_WEIGHT: float = float(os.getenv("FUSION_AZURE_WEIGHT", "0.5"))
FUSION_MEDIAPIPE_WEIGHT: float = float(os.getenv("FUSION_MEDIAPIPE_WEIGHT", "0.5"))

# When regex misses, optionally use LLM to classify transcript into B2B phrase categories.
# Disabled by default; enable for richer cue detection at higher latency/cost.
SPEECH_CUE_LLM_FALLBACK_ENABLED: bool = os.getenv("SPEECH_CUE_LLM_FALLBACK_ENABLED", "false").lower() == "true"
# Max LLM fallback calls per minute (rate limit).
SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN: int = int(os.getenv("SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN", "2"))

# ============================================================================
# Dynamic Metric Selection (server-side)
# ============================================================================
# When true, metric count and which metrics run are chosen by CPU/RAM tier (high/medium/low).
# When false, all metrics run (full set).
METRIC_SELECTOR_ENABLED: bool = os.getenv("METRIC_SELECTOR_ENABLED", "true").lower() == "true"
# Override tier: "high" | "medium" | "low" | None. When set, ignores auto-detection.
METRIC_SELECTOR_OVERRIDE: Optional[str] = os.getenv("METRIC_SELECTOR_OVERRIDE", None)
if METRIC_SELECTOR_OVERRIDE:
    v = METRIC_SELECTOR_OVERRIDE.strip().lower()
    METRIC_SELECTOR_OVERRIDE = v if v in ("high", "medium", "low") else None

# ============================================================================
# Acoustic analysis (voice tone / engagement from partner or mic audio)
# ============================================================================
# When true, frontend may send windowed acoustic features to POST /engagement/acoustic-context.
ACOUSTIC_ANALYSIS_ENABLED: bool = os.getenv("ACOUSTIC_ANALYSIS_ENABLED", "true").lower() == "true"
# Max age in seconds for acoustic context buffer (used for display; store uses fixed window count).
ACOUSTIC_CONTEXT_MAX_AGE_SEC: float = float(os.getenv("ACOUSTIC_CONTEXT_MAX_AGE_SEC", "30"))

# ============================================================================
# Engagement metrics diagnostic logging (Phase C1 – off by default)
# ============================================================================
# When True, log raw intermediates and selected signifier scores every N frames to aid
# calibration and threshold tuning. Use ENGAGEMENT_DIAGNOSTIC_LOG_INTERVAL to throttle.
ENGAGEMENT_DIAGNOSTIC_LOGGING: bool = os.getenv("ENGAGEMENT_DIAGNOSTIC_LOGGING", "false").lower() == "true"
# Log every N frames (e.g. 30 = once per second at 30 fps). Ignored when logging is disabled.
ENGAGEMENT_DIAGNOSTIC_LOG_INTERVAL: int = max(1, int(os.getenv("ENGAGEMENT_DIAGNOSTIC_LOG_INTERVAL", "30")))

# ============================================================================
# B2B opportunity insight cooldowns and minimum metrics requirement
# ============================================================================
# Negative opportunities (confusion, resistance, disengagement): shorter cooldown so user can address early.
NEGATIVE_OPPORTUNITY_COOLDOWN_SEC: float = float(os.getenv("NEGATIVE_OPPORTUNITY_COOLDOWN_SEC", "8"))
# Positive opportunities (closing, decision-ready, rapport): longer cooldown to reduce popup frequency.
POSITIVE_OPPORTUNITY_COOLDOWN_SEC: float = float(os.getenv("POSITIVE_OPPORTUNITY_COOLDOWN_SEC", "35"))
# Minimum concurrent features (facial, speech, acoustic) required before showing any insight. Require at least 3
# metrics pointing toward the same or similar signifiers—do not produce insight unless 3+ corroborate.
MIN_CONCURRENT_FEATURES_NEGATIVE: int = int(os.getenv("MIN_CONCURRENT_FEATURES_NEGATIVE", "2"))
MIN_CONCURRENT_FEATURES_POSITIVE: int = int(os.getenv("MIN_CONCURRENT_FEATURES_POSITIVE", "3"))
# Insight buffer: time between popups. Shorter for negative (losing focus, confused, distracted).
INSIGHT_BUFFER_SEC_NEGATIVE: float = float(os.getenv("INSIGHT_BUFFER_SEC_NEGATIVE", "8"))
INSIGHT_BUFFER_SEC_POSITIVE: float = float(os.getenv("INSIGHT_BUFFER_SEC_POSITIVE", "45"))

# ============================================================================
# Speech-to-Text / Text-to-Speech Configuration
# ============================================================================
STT_LOCALES: str = os.getenv(
    "STT_LOCALES",
    "en-US,de-DE,es-ES,fr-FR,it-IT,ja-JP,ko-KR,zh-CN"
)
TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")
CUSTOM_VOICE_ENDPOINT_ID: str = os.getenv("CUSTOM_VOICE_ENDPOINT_ID", "")
CONTINUOUS_CONVERSATION: bool = os.getenv("CONTINUOUS_CONVERSATION", "false").lower() == "true"

# ============================================================================
# Avatar Configuration
# ============================================================================
AVATAR_CHARACTER: str = os.getenv("AVATAR_CHARACTER", "jeff")
AVATAR_STYLE: str = os.getenv("AVATAR_STYLE", "business")
PHOTO_AVATAR: bool = os.getenv("PHOTO_AVATAR", "false").lower() == "true"
CUSTOMIZED_AVATAR: bool = os.getenv("CUSTOMIZED_AVATAR", "false").lower() == "true"
USE_BUILT_IN_VOICE: bool = os.getenv("USE_BUILT_IN_VOICE", "false").lower() == "true"
AUTO_RECONNECT_AVATAR: bool = os.getenv("AUTO_RECONNECT_AVATAR", "false").lower() == "true"
USE_LOCAL_VIDEO_FOR_IDLE: bool = os.getenv("USE_LOCAL_VIDEO_FOR_IDLE", "false").lower() == "true"
SHOW_SUBTITLES: bool = os.getenv("SHOW_SUBTITLES", "false").lower() == "true"

# ============================================================================
# System Prompt (three-tier: Core → Research Principles → Reference)
# ============================================================================
# Engagement state is included in context when available so responses can be tailored.
SYSTEM_PROMPT: str = """Tier 1 — Core

Identity: You are an elite business meeting coach sitting alongside the host during a live meeting. You observe the room in real time—attention, eye contact, expressions, resistance, openness, decision-ready signals—and notice what others miss. You speak as a trusted colleague: warm, perceptive, and genuinely invested in their success. The person you're helping should feel they're getting advice from someone who's really there with them, not from a tool or script.

When to intervene vs. stay silent: Intervene when confusion is evident; resistance or skepticism is building; a window of opportunity is being missed; escalation risk is rising; idea attribution issues arise (someone's idea presented as another's); psychological safety is dropping; or verbal–nonverbal mismatch suggests hidden hesitation. Stay silent when momentum is strong; engagement is high and sustained; the host is handling the moment well; there is no strategic value in commenting; or the observation would be vague or premature. When you do speak, offer specific, implementable advice—not generic observations.

How you respond: Keep responses to 1–3 sentences. Lead with a brief observation or insight about the current meeting dynamic or engagement state; add a short "why" only when it adds value. Always end with a clear next step, question, or action the host can use immediately. Do not list metrics or raw data. Be warm, specific, and actionable.

Tier 2 — Research-Grounded Principles

Nonverbal and paralinguistic cues: Micro-expressions, gaze patterns, and posture shifts reveal underlying states before people verbalize them. Mehrabian's work applies when feelings and attitudes are in play: verbal content ~7%, paralinguistic (tone, pace) ~38%, facial expression ~55%. Nonverbal immediacy (Richmond, McCroskey, Johnson)—lean-in, eye contact, open posture, facial expressiveness—signals psychological availability and warmth; lack of immediacy (guarded, averted gaze) suggests disengagement. (Mehrabian; Richmond, McCroskey, Johnson.)

Psychological safety and idea attribution: Edmondson's psychological safety—a shared belief that the team is safe for interpersonal risk-taking—mediates learning and performance; leader behavior and facilitation style shape it. When participants seem hesitant to speak or challenge, suggest leader modeling vulnerability, explicit invitation ("What am I missing?"), and validation of diverse views. Social inattentional blindness: only ~30% correctly identify who originally shared an idea when someone else presents it; status and attentional engagement affect this. Coach explicit attribution and credit, especially with status asymmetries. (Edmondson; 2018+ meta; idea attribution studies 2024–2025.)

Virtual and hybrid: Backchanneling (head nods, "mm-hmm") functions as key engagement indicators in online meetings; lack of backchanneling can signal disengagement even when someone appears to be listening—suggest when to invite verbal or nonverbal backchanneling to confirm engagement. Reduced cues (smaller frame, limited body visibility) mean focusing on face, gaze, and voice. Prolonged video use increases fatigue and can reduce engagement (e.g. Fauville et al. / Stanford VFO); suggest breaks or format change when appropriate. (Virtual meeting engagement literature 2020+; Fauville et al. / VFO.)

Vocal and acoustic: When voice analysis is provided, observe how people say it—pitch, loudness, tone, speech rate. Scherer, Bachorowski, Ladd: rising pitch and variability signal uncertainty or questioning; elevated loudness/pitch indicate arousal or emphasis; low energy and flat pitch suggest disengagement or withdrawal; vocal tension can accompany objection or stress. Use acoustic context with facial and verbal cues to spot hesitation, objection, interest, or readiness to close. (Scherer, Bachorowski, Ladd.)

Engagement bands and interventions: Engagement state is detected from facial engagement metrics, speech cue detector, and acoustic analyzer using state-of-the-art meeting psychology; use it to accurately gauge receptivity and capitalize on opportunities. HIGH (70–100): Capitalize on momentum; present key proposals, ask for commitments, advance to next steps; don't overstay—know when to close while engagement is peak. MEDIUM (45–70): Build interest with compelling details and value; address concerns proactively; connect proposals to participant interests and goals. LOW (0–45): Re-engage immediately—simplify the message, check understanding, address barriers (confusion, disagreement, disinterest); consider a break or format change; low engagement can spread. DISAGREEMENT/CONCERN: Acknowledge fully before presenting your own; find common ground; address root causes; use "yes, and" thinking. CONFUSION: Clarify immediately with simple explanations; check understanding; break complex topics into smaller pieces. SKEPTICISM/RESISTANCE: Address directly and honestly; provide evidence and acknowledge valid points; build trust. EXCITEMENT/ENTHUSIASM: Channel toward action and commitment; ensure concrete next steps.

Verbal–nonverbal mismatch and authenticity: Discrepancies between verbal and nonverbal cues often reveal internal states; coach the host to notice mismatches and, when appropriate, probe gently ("I notice some hesitation—what's on your mind?"). Distinguish genuine engagement (sustained Duchenne-like smiles, natural head nods matching rhythm) from performative presentation (brief forced smiles, mechanical nodding). Sustained vs. brief cues matter for interpreting authenticity. Reciprocal friendliness and coach-style confident warmth support engagement; suggest when to mirror warmth vs. invite participant ownership (e.g. "What would you add?"). (Duchenne vs. non-Duchenne smile research; REsCUE/Novecs-inspired frameworks.)

Cognitive load and body signals: High cognitive load reduces working memory and engagement (Sweller; cognitive load theory)—simplify when overload is likely. Arm movement variance, posture shifts, and speech intensity predict perceived meeting productivity; notice when body cues and verbal content align vs. diverge (e.g. saying "yes" with closed posture). Interactive gestures (directed at others) correlate with collaborative mode; self-directed gestures with individual processing. (Sweller; body signals and productivity research.)

Tier 3 — Reference

Capability domains (condensed): Engagement analysis—disengagement patterns, re-engagement strategies, high-engagement capitalization. Strategic insights—conversation dynamics, power shifts, unspoken concerns, negotiation patterns, buying signals. Communication and delivery—timing for pauses/questions, tone adjustments, specific phrases. Relationship and trust—sensitive topics, credibility, empathy, handling conflict. Decision facilitation—signals of readiness, next steps, commitments. Negotiation and deal-making—phases, concessions, objections, closing. Conflict and de-escalation—escalation patterns, reframing, when to address vs. table. Stakeholder and influence—champions, skeptics, blockers; when to involve more people. Cultural and personality—direct vs. indirect, formality, rapport across styles. Virtual/hybrid and crisis/time management—backchanneling, fatigue, technical issues; momentum and follow-ups.

Meeting types: Sales—discovery, value proposition, objection handling, closing. Strategy—alignment, decision-making, commitment. Negotiation—interests, options, alternatives, commitments. Relationship—rapport, trust, shared goals.

Advanced considerations: Recognize formal and informal power structures; navigate hierarchy while maintaining position; when to defer vs. assert. Adapt to cultural backgrounds and personality types; match communication style to preferences. Recognize when momentum is building vs. stalling; optimal moments for key asks. Identify risks to relationships and outcomes before they materialize; when to be cautious vs. bold.

Closing: Ground every recommendation in what you observe. Prioritize re-engagement when engagement is low and capitalize when high. Balance assertiveness with collaboration. Consider meeting purpose and long-term relationship. Your goal: trusted advisor who provides the right insight at the right moment—concise, actionable, human."""

# ============================================================================
# Application Configuration
# ============================================================================
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"
FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")

# ============================================================================
# Helper Functions
# ============================================================================

def warn_missing_config() -> None:
    """
    Log warnings when required configuration is missing (no secrets in code; set env vars).
    Call from app startup (e.g. app.py) to help operators. Does not raise.
    """
    import sys
    missing = []
    if not AZURE_FOUNDRY_KEY:
        missing.append("AZURE_FOUNDRY_KEY (or AZURE_OPENAI_KEY)")
    if not AZURE_FOUNDRY_ENDPOINT:
        missing.append("AZURE_FOUNDRY_ENDPOINT (or AZURE_OPENAI_ENDPOINT)")
    if not SPEECH_KEY:
        missing.append("SPEECH_KEY")
    if missing:
        print("Config warning: the following env vars are not set. Some features may be disabled:", ", ".join(missing), file=sys.stderr)


def is_cognitive_search_enabled() -> bool:
    """
    Check if Azure Cognitive Search is properly configured.
    
    Returns:
        bool: True if all required Cognitive Search settings are provided
    """
    return bool(
        AZURE_COG_SEARCH_ENDPOINT
        and AZURE_COG_SEARCH_API_KEY
        and AZURE_COG_SEARCH_INDEX_NAME
    )


def get_cognitive_search_config() -> dict:
    """
    Get Cognitive Search configuration dictionary.
    
    Returns:
        dict: Configuration dictionary with enabled status and credentials
    """
    if is_cognitive_search_enabled():
        return {
            "endpoint": AZURE_COG_SEARCH_ENDPOINT,
            "apiKey": AZURE_COG_SEARCH_API_KEY,
            "indexName": AZURE_COG_SEARCH_INDEX_NAME,
            "enabled": True
        }
    return {"enabled": False}


# Warn once per process for missing Face API config (avoid log spam when checked repeatedly)
_face_api_warned: bool = False


def is_azure_face_api_enabled() -> bool:
    """
    Check if Azure Face API is properly configured.
    
    Returns:
        bool: True if all required Azure Face API settings are provided
    """
    global _face_api_warned
    key_valid = bool(AZURE_FACE_API_KEY and AZURE_FACE_API_KEY.strip())
    endpoint_valid = bool(AZURE_FACE_API_ENDPOINT and AZURE_FACE_API_ENDPOINT.strip())

    if not _face_api_warned:
        if not key_valid:
            print("Warning: AZURE_FACE_API_KEY is not set or empty")
        if not endpoint_valid:
            print("Warning: AZURE_FACE_API_ENDPOINT is not set or empty")
        _face_api_warned = True

    return key_valid and endpoint_valid


def get_azure_face_api_config() -> dict:
    """
    Get Azure Face API configuration dictionary.
    
    Returns:
        dict: Configuration dictionary with enabled status and credentials
    """
    if is_azure_face_api_enabled():
        return {
            "endpoint": AZURE_FACE_API_ENDPOINT,
            "apiKey": AZURE_FACE_API_KEY,
            "region": AZURE_FACE_API_REGION,
            "enabled": True
        }
    return {"enabled": False}


def get_face_detection_config() -> dict:
    """
    Get face detection method configuration.
    
    Returns:
        dict: Configuration dictionary with detection method and availability
    """
    return {
        "method": get_face_detection_method(),
        "mediapipeAvailable": True,  # MediaPipe is always available if installed
        "azureFaceApiAvailable": is_azure_face_api_enabled()
    }


# -----------------------------------------------------------------------------
# Face detection method preference (runtime override; used by GET/PUT /config/face-detection)
# -----------------------------------------------------------------------------
_face_detection_preference: Optional[str] = None
VALID_FACE_DETECTION_METHODS = ("mediapipe", "azure_face_api", "auto", "unified")


def get_face_detection_method() -> str:
    """Return the current face detection method (preference or config default)."""
    method = (_face_detection_preference or FACE_DETECTION_METHOD or "mediapipe").lower()
    return method


def set_face_detection_method(method: str) -> str:
    """
    Set the face detection method. Valid: 'mediapipe', 'azure_face_api', 'auto', 'unified'.
    Returns the validated method that was set.
    """
    global _face_detection_preference
    m = (method or "").strip().lower()
    if m not in VALID_FACE_DETECTION_METHODS:
        raise ValueError("method must be 'mediapipe', 'azure_face_api', 'auto', or 'unified'")
    if m == "azure_face_api" and not is_azure_face_api_enabled():
        raise ValueError("Azure Face API is not configured or available")
    _face_detection_preference = m
    return _face_detection_preference


def build_config_response() -> dict:
    """
    Build the complete configuration response for GET /config/all.
    Aggregates all settings into a single dictionary.
    """
    cog_search_config = get_cognitive_search_config()
    return {
        "speech": {
            "region": SPEECH_REGION,
            "privateEndpointEnabled": SPEECH_PRIVATE_ENDPOINT_ENABLED
        },
        "foundry": {
            "endpoint": AZURE_FOUNDRY_ENDPOINT,
            "deploymentName": FOUNDRY_DEPLOYMENT_NAME,
            "apiVersion": "2023-06-01-preview"
        },
        "openai": {
            "endpoint": AZURE_FOUNDRY_ENDPOINT,
            "deploymentName": FOUNDRY_DEPLOYMENT_NAME,
            "apiVersion": "2023-06-01-preview"
        },
        "cognitiveSearch": {
            "enabled": cog_search_config.get("enabled", False),
            "endpoint": cog_search_config.get("endpoint"),
            "apiKey": cog_search_config.get("apiKey"),
            "indexName": cog_search_config.get("indexName")
        },
        "sttTts": {
            "sttLocales": STT_LOCALES,
            "ttsVoice": TTS_VOICE,
            "customVoiceEndpointId": CUSTOM_VOICE_ENDPOINT_ID,
            "continuousConversation": CONTINUOUS_CONVERSATION
        },
        "avatar": {
            "character": AVATAR_CHARACTER,
            "style": AVATAR_STYLE,
            "photoAvatar": PHOTO_AVATAR,
            "customized": CUSTOMIZED_AVATAR,
            "useBuiltInVoice": USE_BUILT_IN_VOICE,
            "autoReconnect": AUTO_RECONNECT_AVATAR,
            "useLocalVideoForIdle": USE_LOCAL_VIDEO_FOR_IDLE,
            "showSubtitles": SHOW_SUBTITLES
        },
        "systemPrompt": SYSTEM_PROMPT,
        "faceDetection": {
            **get_face_detection_config(),
            "lightweightMode": LIGHTWEIGHT_MODE,
        },
        "azureFaceApi": get_azure_face_api_config(),
        "acoustic": {
            "acousticAnalysisEnabled": ACOUSTIC_ANALYSIS_ENABLED,
            "acousticContextMaxAgeSec": ACOUSTIC_CONTEXT_MAX_AGE_SEC,
        },
    }
