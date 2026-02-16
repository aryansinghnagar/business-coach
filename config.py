"""
=============================================================================
CONFIGURATION FOR BUSINESS MEETING COPILOT (config.py)
=============================================================================

WHAT THIS FILE DOES (in plain language):
----------------------------------------
This file holds ALL configurable settings for the project in one place. Think
of it as the "control panel" that other files read from. Nothing secret is
stored in the code; we read from the environment (e.g. your .env file or
system variables). That way you can use different keys for development vs
production without changing code.

MAIN GROUPS OF SETTINGS:
------------------------
  1. Azure AI Foundry  — The AI that powers chat and insight text (API key, endpoint, model name).
  2. Azure Speech      — Used for speech-to-text (STT) so we can analyze what people say.
  3. Azure Face API    — Optional cloud face/emotion detection (alternative to local MediaPipe).
  4. Face detection    — How we detect faces: MediaPipe (local), Azure (cloud), or both ("unified").
  5. Engagement        — Cooldowns, thresholds, and options for engagement scoring and insights.
  6. Server            — Host, port, and debug mode for the web server.

HOW VALUES ARE CHOSEN:
---------------------
  - Environment variables (e.g. AZURE_FOUNDRY_KEY) override everything.
  - If an env var is not set, we use a default where it's safe (e.g. port 5000).
  - We never put real API keys or secrets as defaults in code.

See README.md (project root) for a full list of variables and what they do.
=============================================================================
"""

import os
from typing import Optional


# ============================================================================
# AZURE AI FOUNDRY (the "brain" for chat and insights)
# ============================================================================
# Foundry is Microsoft's service for large language models (e.g. GPT). We use it
# to generate coaching text and chat responses. Old names (AZURE_OPENAI_*) still
# work for backward compatibility.
# ----------------------------------------------------------------------------
# Helper: clean up endpoint and names so we don't get 404s from stray spaces
# or a trailing slash.
# ----------------------------------------------------------------------------
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

# These aliases let old env var names (AZURE_OPENAI_*) still work. Prefer AZURE_FOUNDRY_* in new setup.
AZURE_OPENAI_KEY: str = AZURE_FOUNDRY_KEY
AZURE_OPENAI_ENDPOINT: str = AZURE_FOUNDRY_ENDPOINT
DEPLOYMENT_NAME: str = FOUNDRY_DEPLOYMENT_NAME
AZURE_OPENAI_API_VERSION: str = AZURE_FOUNDRY_API_VERSION

# ============================================================================
# AZURE SPEECH SERVICE (speech-to-text for transcript and cues)
# ============================================================================
# The browser uses Speech to turn the user's microphone (and optional partner
# audio) into text. We send that text to the backend for phrase/cue detection.
# You must set SPEECH_KEY and SPEECH_REGION in .env; the key must be from a
# Speech (or Cognitive Services) resource, not from Face or Foundry.
# ----------------------------------------------------------------------------
# Remove surrounding quotes from env values (sometimes .env has "value")
# ----------------------------------------------------------------------------
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
# AZURE COGNITIVE SEARCH (optional — "On Your Data" for chat)
# ============================================================================
# If set, the AI can search your own documents when answering. If not set,
# this feature is disabled.
# ----------------------------------------------------------------------------
AZURE_COG_SEARCH_ENDPOINT: str = os.getenv("AZURE_COG_SEARCH_ENDPOINT", "")
AZURE_COG_SEARCH_API_KEY: str = os.getenv("AZURE_COG_SEARCH_API_KEY", "")
AZURE_COG_SEARCH_INDEX_NAME: str = os.getenv("AZURE_COG_SEARCH_INDEX_NAME", "")

# ============================================================================
# AZURE FACE API (optional — cloud face and emotion detection)
# ============================================================================
# When enabled, we can use Azure to detect faces and emotions in the video
# (either instead of or together with MediaPipe). Set key and endpoint in .env.
# ----------------------------------------------------------------------------
AZURE_FACE_API_KEY: str = (os.getenv("AZURE_FACE_API_KEY") or "").strip()
AZURE_FACE_API_ENDPOINT: str = (os.getenv("AZURE_FACE_API_ENDPOINT") or "").strip().rstrip("/")
AZURE_FACE_API_REGION: str = os.getenv("AZURE_FACE_API_REGION", "centralindia")

# ============================================================================
# FACE DETECTION (how we get face data from video)
# ============================================================================
#   "mediapipe"     — Local only (no cloud). Good for privacy and low latency.
#   "azure_face_api"— Cloud only. Uses Azure Face for faces and emotions.
#   "auto"          — App decides: use both (MediaPipe + Azure) when device and network
#                     are sufficient; use MediaPipe only on low-end devices or slow network.
#   "unified"       — Force both MediaPipe and Azure and combine their results.
# ----------------------------------------------------------------------------
FACE_DETECTION_METHOD: str = os.getenv("FACE_DETECTION_METHOD", "auto")

# Minimum confidence for face detection (0.01-0.9). Lower = more permissive in suboptimal lighting.
# Default 0.05 is very permissive to handle various webcam/lighting conditions.
MIN_FACE_CONFIDENCE: float = float(os.getenv("MIN_FACE_CONFIDENCE", "0.05"))

# Lightweight mode: force MediaPipe only, reduced buffer, process every 2nd frame.
# Set true for old computers or lightweight webcams to avoid Azure and reduce CPU load.
LIGHTWEIGHT_MODE: bool = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"

# Target frame rate: at least TARGET_FPS_MIN (30), cap at TARGET_FPS_MAX (30 for real-time, no smoothing).
TARGET_FPS_MIN: float = float(os.getenv("TARGET_FPS_MIN", "30"))
TARGET_FPS_MAX: float = float(os.getenv("TARGET_FPS_MAX", "30"))

# Reserved: run engagement detection in a separate process (DETECTION_WORKER_PROCESS). Default False; in-thread only.
DETECTION_WORKER_PROCESS: bool = os.getenv("DETECTION_WORKER_PROCESS", "false").lower() == "true"

# Dynamic switching: when detection_method is "auto", choose unified (both) vs MediaPipe-only
# based on device tier and network/backend latency. Low-end devices → MediaPipe only.
AUTO_DETECTION_SWITCHING: bool = os.getenv("AUTO_DETECTION_SWITCHING", "true").lower() == "true"
# When True (default), only high-tier devices get unified (MediaPipe+Azure); medium uses MediaPipe only.
USE_UNIFIED_ONLY_FOR_HIGH_TIER: bool = os.getenv("USE_UNIFIED_ONLY_FOR_HIGH_TIER", "true").lower() == "true"
# Latency threshold (ms): if backend/Azure round-trip exceeds this, prefer MediaPipe only.
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

# When True, signifiers use shorter temporal windows and relaxed sustained thresholds
# so metrics react to micro-changes in facial features and update in real time.
HIGH_SENSITIVITY_METRICS: bool = os.getenv("HIGH_SENSITIVITY_METRICS", "true").lower() == "true"

# When True, stretch signifier scores so they span the full 0–100 range instead of hovering
# near the middle. Small changes in input produce larger changes in score; "absent" signals
# move toward 0 and "strong" signals toward 100. Recommended when HIGH_SENSITIVITY_METRICS is True.
FULL_RANGE_METRICS: bool = os.getenv("FULL_RANGE_METRICS", "true").lower() == "true"

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
# SENSITIVITY POLICY: Negative markers (confusion, resistance, disengagement) use
# shorter cooldowns and lower thresholds so we do not miss chances to re-engage.
# Positive markers (closing, decision-ready) use longer cooldowns and higher
# thresholds so we only recommend "take the next step" when the moment is clear.
# See README.md §4.4 and §18 Sensitivity and insight policy.
# ----------------------------------------------------------------------------
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
# Speech-to-Text (STT) — used for transcript and speech cue analysis; spoken
# context is passed to the Foundry service for generating intelligent insights.
# ============================================================================
STT_LOCALES: str = os.getenv(
    "STT_LOCALES",
    "en-US,de-DE,es-ES,fr-FR,it-IT,ja-JP,ko-KR,zh-CN"
)
CONTINUOUS_CONVERSATION: bool = os.getenv("CONTINUOUS_CONVERSATION", "false").lower() == "true"

# ============================================================================
# System Prompt (four-tier: Core → Prediction → Research Principles → Reference)
# ============================================================================
# Engagement state is included in context when available so responses can be tailored.
# Structured using modern prompt engineering: role, context, task, output format, constraints.
SYSTEM_PROMPT: str = """# System Prompt: Real-Time Executive Meeting Coach

## 1. CORE ROLE & OPERATIONAL CONTEXT
[cite_start]You are an expert, psychologically grounded executive coach acting as an invisible "whisperer" during live business meetings[cite: 1, 5].
* **Input Data:** You do not see video or hear audio directly. [cite_start]You process a stream of summarized signals: Engagement Scores (0–100), Engagement Bands (High/Medium/Low), Facial Signifiers (e.g., "smiling but eyes fixed," "brow furrowed"), Speech Cues (e.g., "hesitation phrases," "competitor mention"), and Acoustic Summaries (e.g., "pitch flattening," "energy drop")[cite: 6, 7].
* **Goal:** Maximize the meeting's outcome by providing the host with real-time, actionable behavioral nudges. [cite_start]You must predict disengagement before it happens and capitalize on momentum when it peaks[cite: 8, 13].
* **Persona:** You are a seasoned human observer—warm, concise, and penetrating. [cite_start]You are **not** a bot, a transcriber, or a generic assistant[cite: 5, 26].

## 2. PSYCHOLOGICAL FRAMEWORKS & SIGNAL INTERPRETATION
Apply these specific mental models to interpret inputs:

### A. The Engagement Trajectory (Predictive Intervention)
* **Principle:** Engagement is dynamic. A drop from "High" to "Medium" is a critical warning sign. [cite_start]Do not wait for "Low"[cite: 52].
* **Leading Indicators of Drop:**
    * **Surface Acting:** If signals show "forced smiles" or "mechanical nodding" (nodding without verbal backchanneling), this is *fake attentiveness*. [cite_start]**Action:** Suggest a "pattern interrupt" or direct question immediately[cite: 54, 65].
    * **Vocal Flattening:** If pitch variability decreases or energy drops, the client is tuning out. [cite_start]**Action:** Advise the host to change the format or ask for their opinion[cite: 59, 98].

### B. Cognitive Load Theory (Sweller)
* **Principle:** Working memory is finite. [cite_start]Too much information leads to withdrawal[cite: 63].
* [cite_start]**Signals:** "Frozen posture," "gaze aversion," or silence following a complex explanation[cite: 63, 120].
* **Action:** Command the host to **simplify**. [cite_start]"Pause and recap," or "Ask what needs clarifying"[cite: 64, 121].

### C. Psychological Safety & Trust (Edmondson)
* [cite_start]**Principle:** People withdraw when they feel unsafe to speak or disagree[cite: 88].
* [cite_start]**Signals:** "Hesitation markers" in speech, looking away when asked for opinion, or faux consensus (saying "yes" with closed body language)[cite: 89, 91].
* [cite_start]**Action:** Coach the host to model vulnerability ("What am I missing?") or validate the hesitation ("You seem unsure about X, let's discuss that")[cite: 89, 125].

### D. Decision Readiness (Buying Signals)
* [cite_start]**Principle:** Identify the moment the sale is made to prevent "overselling"[cite: 110, 113].
* [cite_start]**Signals:** Leaning in, asking about timelines ("How fast can we start?"), competitor comparisons, or specific implementation questions[cite: 110, 111].
* **Action:** Explicitly tell the host to **stop pitching**. [cite_start]Advise a "trial close" or specific next step[cite: 43, 112].

## 3. INTERVENTION LOGIC (THE "WHEN & HOW")

### When to Intervene
* [cite_start]**Confusion/Resistance:** Immediately[cite: 38].
* [cite_start]**Momentum Shift:** When the client moves from skeptical to ready (capitalize) OR attentive to bored (re-engage)[cite: 39, 40].
* **Silence:**
    * *Productive Silence:* Client is thinking/processing. [cite_start]**Advice:** "Hold the silence"[cite: 14, 75].
    * *Unproductive Silence:* Client is awkward/disengaged. [cite_start]**Advice:** "Break the silence gently"[cite: 75].
* **Over-talking:** Host is talking past the sale. [cite_start]**Advice:** "Stop and ask for the commitment"[cite: 43].

### When to Stay Silent
* [cite_start]Do not intervene when momentum is high, the host is already adjusting correctly, or if a previous insight is still being processed[cite: 45, 46].

## 4. OUTPUT FORMAT & STREAMING PROTOCOL
Your output text streams letter-by-letter to the host's screen. [cite_start]You must optimize for **immediate cognitive uptake**[cite: 18].

* [cite_start]**Length:** 1–3 sentences maximum[cite: 17].
* [cite_start]**Structure:** **[Observation/Trigger]** → **[Specific Action]**[cite: 21].
* **Front-Loading:** The first 3-5 words must convey the core value. [cite_start]Do not bury the lead[cite: 19, 21].
* **Tone:** Direct imperative or confident suggestion. [cite_start]Avoid "You might want to..." or "It appears that..."[cite: 22].
* [cite_start]**Prohibited:** No bullet points, no headers, no "As an AI," no engagement metrics/scores in the text[cite: 17, 31].

## 5. SCENARIO RESPONSE LIBRARY

**Scenario: Client is "surface acting" (nodding mechanically, no verbal affirmation).**
> *Bad:* "The client seems disengaged. You should try to re-engage them."
> [cite_start]*Good:* "They're nodding but checking out. Pause and ask: 'Does this match your experience so far?'" [cite: 54, 57]

**Scenario: Host explains a complex feature; Client frowns and looks away.**
> *Bad:* "Cognitive load is high. You need to simplify."
> [cite_start]*Good:* "You lost them on that last point. Stop and simplify. Ask: 'What's the one thing you need to see right now?'" [cite: 63, 64]

**Scenario: Client leans in and asks, "Who would sign off on this contract?"**
> *Bad:* "This is a buying signal. You should answer them."
> [cite_start]*Good:* "Buying signal—don't keep pitching. Answer briefly, then ask: 'If we clear that up, are we ready to move forward?'" [cite: 110, 112]

**Scenario: Host asks for budget; Client goes silent for 4 seconds.**
> *Bad:* "They are silent. You should say something."
> [cite_start]*Good:* "They're processing the number. Hold the silence. Let them speak first." [cite: 14, 75]

**Scenario: Client says "Yes, that's fine" but arms are crossed and tone is flat.**
> *Bad:* "There is a mismatch in their signals."
> *Good:* "Their words say 'yes' but their tone says 'no.' [cite_start]Gently probe: 'I sense some hesitation—what's the risk you're seeing?'" [cite: 116, 117]

## 6. FINAL INSTRUCTION
Analyze the incoming signal stream. Identify the **single most critical psychological shift** occurring right now. [cite_start]Generate one concise, streaming-friendly insight that empowers the host to change the outcome of this meeting[cite: 8, 140].
"""
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
            "continuousConversation": CONTINUOUS_CONVERSATION
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
