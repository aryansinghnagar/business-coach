"""
Insight Generator Service

Generates real-time, actionable B2B meeting insights by combining:
- Visual cues: engagement detector metric spikes (facial expression groups G1–G4)
- Speech cues: recent transcript from meeting partner audio (POST /engagement/transcript)
- Phrase-triggered: when partner says B2B-relevant phrases (objections, interest, etc.)

Flow: Spike / phrase / opportunity -> backend fetches context -> Azure AI Foundry -> insight.
Stock messages are RESERVE FALLBACKS only—used when Azure AI Foundry is unavailable, times out, or
returns empty. When Azure is available, we ALWAYS use it to generate custom, context-aware responses.
Do not rely on stock insights when the service works. Use Azure liberally for unique, human-sounding
advice grounded in facial signifiers, speech cues, acoustic features, and user-provided context.
See docs/DOCUMENTATION.md.
"""

import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

import config
from services.azure_foundry import get_foundry_service
from services.acoustic_context_store import get_recent_acoustic_context, get_recent_acoustic_tags
from utils.b2b_opportunity_detector import NEGATIVE_OPPORTUNITY_IDS

# -----------------------------------------------------------------------------
# Insight weights (external backend: prompt_suffix, max_length, opportunity_thresholds)
# -----------------------------------------------------------------------------
INSIGHT_WEIGHTS_TTL_SEC = 300
_insight_weights_memory: Dict[str, Any] = {}
_insight_weights_ts: float = 0.0
_insight_weights_lock = threading.Lock()

DEFAULT_INSIGHT_WEIGHTS: Dict[str, Any] = {
    "prompt_suffix": "",
    "max_length": 100,  # 2–3 sentences for popups; tokens ~= words * 1.3
    "opportunity_thresholds": {},
}


def get_insight_weights() -> Dict[str, Any]:
    """
    Return insight-generation parameters. Fetches from INSIGHT_WEIGHTS_URL with TTL cache,
    or uses in-memory values set via set_insight_weights (PUT /api/weights/insight), or defaults.
    """
    global _insight_weights_ts
    with _insight_weights_lock:
        now = time.time()
        url = getattr(config, "INSIGHT_WEIGHTS_URL", None)
        if url and requests is not None and (now - _insight_weights_ts >= INSIGHT_WEIGHTS_TTL_SEC or not _insight_weights_memory):
            try:
                r = requests.get(url, timeout=5)
                if r.ok:
                    data = r.json()
                    if isinstance(data, dict):
                        _insight_weights_memory.update({
                            k: v for k, v in data.items()
                            if k in ("prompt_suffix", "max_length", "opportunity_thresholds")
                        })
                        _insight_weights_ts = now
            except Exception:
                pass
        out = dict(DEFAULT_INSIGHT_WEIGHTS)
        out.update(_insight_weights_memory)
        return out


def set_insight_weights(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update in-memory insight weights (e.g. from PUT /api/weights/insight).
    Keys: prompt_suffix (str), max_length (int), opportunity_thresholds (dict). Partial update.
    """
    with _insight_weights_lock:
        if isinstance(data.get("prompt_suffix"), str):
            _insight_weights_memory["prompt_suffix"] = data["prompt_suffix"]
        if isinstance(data.get("max_length"), (int, float)):
            _insight_weights_memory["max_length"] = int(data["max_length"])
        if isinstance(data.get("opportunity_thresholds"), dict):
            _insight_weights_memory["opportunity_thresholds"] = dict(data["opportunity_thresholds"])
        return get_insight_weights()

# -----------------------------------------------------------------------------
# Phrase-triggered insights (aural cues) — B2B meeting relevance
# -----------------------------------------------------------------------------
# Tier 1 (primary for composites): commitment, interest, objection, confusion,
# concern, realization. Tier 2: timeline, budget (used in opportunity detection).
# Categories and phrase patterns; match is case-insensitive.
# Order matters: first match wins; put more specific phrases earlier if needed.
# Expanded with B2B meeting corpora, negation/hedging, meeting-specific phrases.
PHRASE_CATEGORIES = {
    "objection": [
        r"\bI'm not sure\b", r"\bI have concerns\b", r"\bthat doesn't work\b",
        r"\bwe can't\b", r"\bnot interested\b", r"\bpush back\b", r"\bI disagree\b",
        r"\bthat won't work\b", r"\bno way\b", r"\bnot convinced\b",
        r"\bI'm not convinced yet\b", r"\bwe might need to reconsider\b",
        r"\bthat's a non-starter\b", r"\bdoesn't fit\b", r"\bcan't agree\b",
        r"\bnot on board\b", r"\bthat won't fly\b", r"\bwe'll have to pass\b",
    ],
    "interest": [
        r"\bthat's interesting\b", r"\btell me more\b", r"\bI like that\b",
        r"\bI'd like to hear more\b", r"\bmakes sense\b", r"\bsounds good\b",
        r"\bI see\b", r"\bgot it\b", r"\bthat could work\b", r"\binteresting point\b",
        r"\bgood point\b", r"\bI like where this is going\b", r"\bcurious about\b",
        r"\bthat resonates\b", r"\bI can see the value\b", r"\bworth exploring\b",
    ],
    "confusion": [
        r"\bI don't understand\b", r"\bcan you clarify\b", r"\bwhat do you mean\b",
        r"\bI'm confused\b", r"\bcan you explain\b", r"\bnot following\b",
        r"\bunclear\b", r"\bcould you repeat\b", r"\blost me\b", r"\bnot sure I follow\b",
        r"\bcan you walk me through\b", r"\bwhat does that mean\b", r"\bhow so\b",
    ],
    "commitment": [
        r"\blet's do it\b", r"\bI'm in\b", r"\bI'm ready to move forward\b",
        r"\bwe'll move forward\b", r"\bnext steps\b", r"\bwhat's the next step\b",
        r"\bsign off\b", r"\bwe're ready\b", r"\bdeal\b", r"\bagreed\b",
        r"\blet's proceed\b", r"\bmove ahead\b", r"\bwe're good to go\b",
        r"\blet's lock this in\b", r"\bI'm ready to sign\b", r"\bcount me in\b",
    ],
    "concern": [
        r"\bworried about\b", r"\bconcerned\b", r"\bhesitant\b",
        r"\bneed to think\b", r"\bnot ready yet\b", r"\brisky\b",
        r"\bI'll need to run this by\b", r"\blet me circle back\b",
        r"\bwe're still evaluating\b", r"\bneed to discuss with\b",
        r"\bsome reservations\b", r"\bnot quite there\b", r"\bholding off\b",
    ],
    "timeline": [
        r"\bwhen can we\b", r"\bby when\b", r"\btimeline\b", r"\bdeadline\b",
        r"\bhow soon\b", r"\bdelivery date\b", r"\bwhen would we\b",
        r"\bstart date\b", r"\blaunch date\b", r"\bhow long will it take\b",
    ],
    "budget": [
        r"\bbudget\b", r"\bcost\b", r"\bpricing\b", r"\bexpensive\b",
        r"\bafford\b", r"\bprice point\b", r"\binvestment\b", r"\bROI\b",
        r"\btotal cost\b", r"\bwhat's the price\b", r"\bpaying for\b",
    ],
    "realization": [
        r"\baha\b", r"\boh I see\b", r"\bthat makes sense\b", r"\bgot it\b",
        r"\bI get it\b", r"\bnow I understand\b", r"\bthat clicks\b", r"\bclicked\b",
        r"\bthat clarifies it\b", r"\bmakes more sense now\b", r"\bI see what you mean\b",
    ],
    "urgency": [
        r"\basap\b", r"\bneed it soon\b", r"\btime[- ]?sensitive\b", r"\burgent\b",
        r"\bpriority\b", r"\brush\b", r"\bdeadline\b", r"\bsoon(?:er)?\s+than\b",
    ],
    "skepticism": [
        r"\bsounds too good\b", r"\bprove it\b", r"\bshow me\b", r"\bwe'll see\b",
        r"\bI'll believe it when\b",         r"\btoo good to be true\b", r"\bneed to see\b",
        r"\bconvince me\b", r"\bhas to be true\b", r"\breally\?\b",  # skepticism "really?"
    ],
    "enthusiasm": [
        r"\blove it\b", r"\bexcited about\b", r"\bcan't wait\b", r"\bgreat idea\b",
        r"\bthat's great\b", r"\bawesome\b", r"\bfantastic\b", r"\bperfect\b",
        r"\bexactly what we need\b", r"\bbrilliant\b",
    ],
    "authority": [
        r"\bI'll have to check with\b", r"\bmy team\b", r"\bthe board\b",
        r"\bdecision makers\b", r"\bneed to run this by\b", r"\bhave to get approval\b",
        r"\bmy manager\b", r"\bstakeholders\b", r"\bneed sign[- ]?off\b",
    ],
    "hesitation": [
        r"\bum\b", r"\buh\b", r"\bwell\b", r"\bI guess\b", r"\bsort of\b",
        r"\bmaybe\b", r"\bI'm not sure\b", r"\bkind of\b", r"\bperhaps\b",
        r"\bit depends\b", r"\bmight\b", r"\bcould be\b",
    ],
    "confirmation": [
        r"\bexactly\b", r"\bright\b", r"\bcorrect\b", r"\bthat's it\b",
        r"\bprecisely\b", r"\byes\b", r"\babsolutely\b", r"\bthat's right\b",
        r"\bspot on\b", r"\bexactly right\b",
    ],
}

# Discourse markers (IBM/AAAI research: sentiment-carrying markers enhance NLP).
# Used as secondary signal to boost confidence when combined with primary category match.
# Format: (pattern, polarity) where polarity is "positive", "negative", or "hedging".
# Does not drive alerts alone; augments primary matches.
DISCOURSE_MARKERS = {
    "positive": [
        r"\bfortunately\b", r"\bthankfully\b", r"\bactually\b(?!\s+(not|don't|can't|won't))",
        r"\bdefinitely\b", r"\babsolutely\b",
    ],
    "negative": [
        r"\bunfortunately\b", r"\bfrankly\b", r"\bhonestly\b", r"\bactually\s+(not|don't|can't|won't)",
        r"\bto be honest\b", r"\bthe problem is\b",
    ],
    "hedging": [
        r"\bbasically\b", r"\bI mean\b", r"\byou know\b", r"\banyway\b",
        r"\bkind of\b", r"\bsort of\b", r"\bperhaps\b", r"\bmaybe\b",
    ],
}

# Cooldown between phrase-triggered alerts (seconds)
AURAL_COOLDOWN_SEC = 15  # Shorter buffer for phrase-triggered insights (often confusion/objection)
_pending_aural_alert: Optional[dict] = None
_aural_alert_lock = threading.Lock()
_last_aural_trigger_time: float = 0.0
_last_aural_trigger_category: Optional[str] = None

# LLM fallback rate limit
_llm_fallback_calls: deque = deque(maxlen=10)
_llm_fallback_lock = threading.Lock()

# Recent speech tags for multimodal composite detection (facial + speech)
# Ring buffer: (category, phrase, timestamp); kept for SPEECH_TAGS_WINDOW_SEC
SPEECH_TAGS_WINDOW_SEC = 12.0
_recent_speech_tags: deque = deque(maxlen=50)
_speech_tags_lock = threading.Lock()


def _check_for_trigger_phrases(text: str) -> List[Tuple[str, str]]:
    """Return list of (category, matched_phrase). Case-insensitive; allows multiple categories per text."""
    out: List[Tuple[str, str]] = []
    if not text or not text.strip():
        return out
    lower = text.lower().strip()
    seen_cats: set = set()
    for category, patterns in PHRASE_CATEGORIES.items():
        if category in seen_cats:
            continue
        for pat in patterns:
            m = re.search(pat, lower, re.IGNORECASE)
            if m:
                out.append((category, m.group(0)))
                seen_cats.add(category)
                break
    return out


def _check_discourse_marker(text: str) -> Optional[Tuple[str, str]]:
    """
    Return (polarity, matched_marker) or None. Used as secondary signal to boost
    confidence when combined with primary category match. Does not drive alerts alone.
    """
    if not text or not text.strip():
        return None
    lower = text.lower().strip()
    for polarity, patterns in DISCOURSE_MARKERS.items():
        for pat in patterns:
            m = re.search(pat, lower, re.IGNORECASE)
            if m:
                return (polarity, m.group(0))
    return None


def _set_pending_aural_alert(category: str, phrase: str) -> bool:
    """Set pending aural alert if cooldown has passed. Avoid duplicate for same category."""
    global _pending_aural_alert, _last_aural_trigger_time, _last_aural_trigger_category
    with _aural_alert_lock:
        now = time.time()
        if now - _last_aural_trigger_time < AURAL_COOLDOWN_SEC and _last_aural_trigger_category == category:
            return False
        if now - _last_aural_trigger_time < AURAL_COOLDOWN_SEC:
            return False
        _pending_aural_alert = {"type": "aural", "category": category, "phrase": phrase}
        _last_aural_trigger_time = now
        _last_aural_trigger_category = category
        return True


def _should_try_llm_fallback(text: str) -> bool:
    """True when transcript is non-trivial (e.g. >15 words or contains hedging markers)."""
    if not text or not text.strip():
        return False
    words = len(text.strip().split())
    if words > 15:
        return True
    dm = _check_discourse_marker(text)
    return dm is not None and dm[0] == "hedging"


def _llm_classify_phrase(text: str) -> Optional[Tuple[str, str]]:
    """
    Use Azure Foundry to classify transcript into B2B phrase category.
    Rate-limited and feature-flagged. Returns (category, phrase_snippet) or None.
    """
    if not getattr(config, "SPEECH_CUE_LLM_FALLBACK_ENABLED", False):
        return None
    rate_per_min = getattr(config, "SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN", 2)
    with _llm_fallback_lock:
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in list(_llm_fallback_calls) if t >= cutoff]
        if len(recent) >= rate_per_min:
            return None
        _llm_fallback_calls.clear()
        _llm_fallback_calls.extend(recent)
        _llm_fallback_calls.append(now)
    try:
        svc = get_foundry_service()
        if not svc:
            return None
        prompt = (
            "Classify this B2B meeting transcript snippet into exactly one category, or none. "
            "Reply with ONLY the category name or 'none'. Categories: objection, interest, confusion, "
            "commitment, concern, timeline, budget, realization, urgency, skepticism, enthusiasm, "
            "authority, hesitation, confirmation.\n\n"
            f"Snippet: {text[:500].strip()}"
        )
        resp = svc.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        if not resp or not isinstance(resp, str):
            return None
        cat = resp.strip().lower().replace(".", "")
        if cat == "none" or cat not in PHRASE_CATEGORIES:
            return None
        snippet = text[:80].strip() + ("..." if len(text) > 80 else "")
        return (cat, snippet)
    except Exception:
        return None


def _discourse_aligns_with_category(discourse_polarity: str, category: str) -> bool:
    """True when discourse marker polarity aligns with category (boosts confidence)."""
    negative_cats = {"objection", "concern", "confusion"}
    positive_cats = {"interest", "commitment", "realization"}
    if discourse_polarity == "negative" and category in negative_cats:
        return True
    if discourse_polarity == "positive" and category in positive_cats:
        return True
    if discourse_polarity == "hedging" and category in {"concern", "confusion"}:
        return True
    return False


def get_discourse_boost(text: str, category: str) -> bool:
    """
    Return True when a discourse marker in text aligns with the given category.
    Used as secondary signal to boost confidence for composite detection.
    """
    dm = _check_discourse_marker(text)
    if not dm:
        return False
    polarity, _ = dm
    return _discourse_aligns_with_category(polarity, category)


def check_speech_cues(text: str) -> List[Tuple[str, str]]:
    """
    Return list of (category, phrase) from regex multi-match and optional LLM fallback.
    Used by POST /engagement/transcript route.
    """
    matches = _check_for_trigger_phrases(text)
    if matches:
        return matches
    if _should_try_llm_fallback(text):
        llm_result = _llm_classify_phrase(text)
        if llm_result:
            return [llm_result]
    return []


def append_speech_tag(
    category: str,
    phrase: str,
    discourse_boost: bool = False,
) -> None:
    """Append a speech tag (category, phrase, timestamp) for multimodal composite detection."""
    with _speech_tags_lock:
        _recent_speech_tags.append({
            "category": category,
            "phrase": phrase,
            "time": time.time(),
            "discourse_boost": discourse_boost,
        })


def get_recent_speech_tags(within_sec: float = SPEECH_TAGS_WINDOW_SEC) -> list:
    """
    Return list of recent speech tags within the last within_sec seconds.
    Each item: {"category": str, "phrase": str, "time": float}.
    Thread-safe; used by composite detector for multimodal triggers.
    """
    now = time.time()
    cutoff = now - within_sec
    with _speech_tags_lock:
        return [t for t in list(_recent_speech_tags) if t["time"] >= cutoff]


def clear_recent_speech_tags() -> None:
    """Clear recent speech tags (e.g. when engagement stops)."""
    with _speech_tags_lock:
        _recent_speech_tags.clear()


def get_pending_aural_alert() -> Optional[dict]:
    """Peek at pending aural alert without clearing."""
    with _aural_alert_lock:
        return _pending_aural_alert


def get_and_clear_pending_aural_alert() -> Optional[dict]:
    """Thread-safe get and clear of pending aural alert."""
    global _pending_aural_alert
    with _aural_alert_lock:
        a = _pending_aural_alert
        _pending_aural_alert = None
        return a


def clear_pending_aural_alert() -> None:
    """Clear pending aural alert (e.g. when engagement stops)."""
    global _pending_aural_alert
    with _aural_alert_lock:
        _pending_aural_alert = None


# -----------------------------------------------------------------------------
# Recent transcript store (speech cues from meeting partner audio)
# -----------------------------------------------------------------------------
# Ring buffer: last N characters of transcript for context when generating insight.
# Written by POST /engagement/transcript; read when generating spike insight.
_TRANSCRIPT_MAX_CHARS = 1200
_transcript_buffer: deque = deque(maxlen=1)  # single string, we'll trim by char count
_transcript_lock = threading.Lock()


def append_transcript(text: str) -> None:
    """Append meeting partner speech transcript. Keeps last TRANSCRIPT_MAX_CHARS."""
    if not text or not text.strip():
        return
    with _transcript_lock:
        current = (_transcript_buffer[0] if len(_transcript_buffer) > 0 else "") or ""
        current = (current + " " + text.strip()).strip()
        if len(current) > _TRANSCRIPT_MAX_CHARS:
            current = current[-_TRANSCRIPT_MAX_CHARS:]
        _transcript_buffer.clear()
        _transcript_buffer.append(current)


def get_recent_transcript() -> str:
    """Return recent transcript for insight generation (thread-safe)."""
    with _transcript_lock:
        return (_transcript_buffer[0] if _transcript_buffer else "") or ""


def clear_transcript() -> None:
    """Clear transcript (e.g. when engagement stops)."""
    with _transcript_lock:
        _transcript_buffer.clear()


# -----------------------------------------------------------------------------
# Group descriptions for prompt (visual cue context) — Psychology-grounded
# -----------------------------------------------------------------------------
# Each description ties the metric group to specific facial action units (FACS),
# psychology research, and observable cues for more grounded insights.
GROUP_DESCRIPTIONS = {
    "g1": (
        "Interest & Engagement — You're seeing signs like genuine (Duchenne) smiling (orbicularis oculi + "
        "zygomaticus major), head tilt toward the speaker (active listening posture), sustained eye contact, "
        "eyebrow raises (surprise/interest), and forward lean. These are classic buying signals in sales "
        "psychology—approach motivation, curiosity, and social bonding. Per Ekman's FACS, AU6 (cheek raise) + "
        "AU12 (lip corner pull) = authentic positive affect. The partner is emotionally available and receptive."
    ),
    "g2": (
        "Cognitive Load — You're seeing signs like furrowed brow (AU4, corrugator supercilii = effortful thinking), "
        "reduced blinking (deep focus), slight lip compression (processing), gaze aversion (internal processing), "
        "and stillness (cognitive bandwidth consumed). Per Kahneman's System 2, they're engaging in deliberate, "
        "effortful thought—evaluating your proposal, comparing options, or working through complexity. Too much "
        "load risks decision fatigue and disengagement."
    ),
    "g3": (
        "Resistance & Discomfort — You're seeing signs like asymmetric expressions (e.g. one-sided lip pull = "
        "contempt/skepticism), compressed lips (AU23/24 = suppressed disagreement), nose wrinkle (AU9 = disgust/ "
        "distaste), gaze aversion (avoidance), and reduced facial expressiveness (guarded). These are distancing "
        "signals—approach-avoidance conflict, skepticism, or unresolved objections. Per Mehrabian's research, "
        "nonverbal leakage reveals true sentiment even when words are polite."
    ),
    "g4": (
        "Decision-Ready — You're seeing signs like sustained eye contact (commitment, trust), relaxed facial "
        "muscles (tension release = decision made), genuine smile, forward lean, and nodding (agreement cues). "
        "These are closing signals—they've resolved internal conflict and are ready to act. Per Cialdini's "
        "commitment/consistency principle, once they signal yes nonverbally, they're primed to follow through. "
        "This is the window to ask for commitment or next step."
    ),
}

# Stock fallbacks if Azure AI Foundry fails or times out — Psychology-grounded
STOCK_MESSAGES = {
    "g1": (
        "Their face just lit up—genuine smile, eyes engaged. That's authentic interest, not politeness. "
        "Lean in: ask what's resonating or go deeper on that point."
    ),
    "g2": (
        "Furrowed brow, stillness, gaze turning inward—they're processing hard. Give them a beat. "
        "Too much now risks overload. Pause, then ask what's on their mind."
    ),
    "g3": (
        "Tension in the face—compressed lips, maybe a slight pull. Something's not sitting right. "
        "Acknowledge it directly: 'I'm sensing some hesitation—what's your concern?'"
    ),
    "g4": (
        "Relaxed face, steady eye contact, slight nod—they've made up their mind. "
        "This is your window. Offer a clear next step or ask for the commitment."
    ),
}

# Stock fallbacks for phrase-triggered (aural) insights — Psychology-grounded
AURAL_STOCK_MESSAGES = {
    "objection": (
        "They just pushed back. That's not a no—it's a request for more information. "
        "Validate first: 'That's a fair point.' Then address the concern directly. Resistance drops when they feel heard."
    ),
    "interest": (
        "They're leaning in verbally—'that's interesting' or 'tell me more' is an invitation. "
        "This is curiosity signaling. Go deeper, or ask what specifically caught their attention."
    ),
    "confusion": (
        "They're lost—their words just told you. Cognitive load is spiking. "
        "Stop adding information. Simplify, recap, or ask 'What would help clarify this?'"
    ),
    "commitment": (
        "They just signaled readiness—'let's do it,' 'next steps,' or similar. That's commitment language. "
        "Don't oversell. Lock it in: confirm the next step and timeline."
    ),
    "concern": (
        "They voiced a worry. Underneath it is a need—security, clarity, or control. "
        "Name it: 'It sounds like you want to make sure...' Then address it directly."
    ),
    "timeline": (
        "Timing matters to them right now. They're mentally mapping this to their calendar and priorities. "
        "Be specific: give dates, milestones, or ask what timeline works for them."
    ),
    "budget": (
        "Money's on their mind. That's not necessarily a blocker—it's a value question. "
        "Reframe cost as investment. Tie it to outcomes they care about."
    ),
    "urgency": (
        "They're signaling time sensitivity. Be concrete: specific dates, milestones, or ask what timeline works."
    ),
    "skepticism": (
        "They're skeptical—'prove it' or 'we'll see' means they need evidence. "
        "Address it head-on: offer proof points, demos, or references."
    ),
    "enthusiasm": (
        "They're showing real enthusiasm—'love it,' 'great idea.' "
        "This is momentum. Build on it: ask what's resonating or propose the next step."
    ),
    "authority": (
        "They're deferring to others—'I'll check with my team' or 'the board.' "
        "Help them sell internally: arm them with one-pagers, talking points, or case studies."
    ),
    "hesitation": (
        "Hesitation markers—'um,' 'well,' 'maybe.' Something's holding them back. "
        "Lower the stakes: offer a smaller step or ask what would help them feel comfortable."
    ),
    "confirmation": (
        "They're affirming—'exactly,' 'correct,' 'that's right.' "
        "They're with you. Reinforce the point and move toward commitment."
    ),
}

# Fallbacks for B2B opportunity-triggered insights (visual/temporal cues) — Psychology-grounded
# Multimodal composites (facial + speech) have dedicated stock messages.
OPPORTUNITY_STOCK_MESSAGES = {
    "decision_readiness_multimodal": (
        "They're showing commitment in both words and expression—relaxed gaze, positive face, and language that signals readiness. "
        "Ask for the next step or confirmation now."
    ),
    "cognitive_overload_multimodal": (
        "Their face and words both point to overload—furrowed brow, gaze shifting, and confusion or concern in what they said. "
        "Pause, simplify, or ask: 'What would help clarify this?'"
    ),
    "skepticism_objection_multimodal": (
        "They've voiced a concern or objection and their face backs it up—tension, compressed lips, or averted gaze. "
        "Address it directly: 'What's your main concern?' and listen before responding."
    ),
    "aha_insight_multimodal": (
        "Something just landed—their expression shifted and they used language like 'got it' or 'that makes sense.' "
        "Capitalize: 'How does this fit with what you need?' or reinforce the key point."
    ),
    "disengagement_multimodal": (
        "Their face and lack of positive language suggest attention is drifting—low engagement cues and no recent commitment or interest phrases. "
        "Re-engage with a direct question or shift to what matters to them."
    ),
    "loss_of_interest": (
        "Interest is dropping—no commitment language, flat or withdrawn voice. "
        "Re-engage: ask what matters to them or shift to a topic that sparks energy."
    ),
    "closing_window": (
        "Their face has settled—tension released, eyes steady. Decision's made internally. "
        "This window closes fast. Ask for the commitment or next step now."
    ),
    "decision_ready": (
        "Relaxed brow, sustained gaze, maybe a slight nod. They've resolved the internal debate. "
        "Don't add more; ask: 'Does this work for you?' or 'What do you need to move forward?'"
    ),
    "ready_to_sign": (
        "All the signals say yes—genuine smile, forward lean, open posture. They're waiting for you to close. "
        "Propose the next step clearly and confidently."
    ),
    "buying_signal": (
        "Duchenne smile, raised brows, engaged eyes—classic approach signals. They like what they're hearing. "
        "Go deeper: 'What's resonating most?' or reinforce the value that sparked this."
    ),
    "commitment_cue": (
        "Nodding, eye contact, relaxed face—these are agreement signals. "
        "Cement it: summarize what you've agreed and ask for confirmation."
    ),
    "cognitive_overload_risk": (
        "Furrowed brow, gaze aversion, stillness—System 2 is maxed out. "
        "Stop adding. Pause, simplify, or ask: 'What's the most important thing to clarify?'"
    ),
    "confusion_moment": (
        "Their face just flickered—brow furrow, slight squint. Processing hit a wall. "
        "Don't push forward. Ask: 'What would help clarify this?' or recap the key point."
    ),
    "need_clarity": (
        "Subtle signs of uncertainty—gaze shifting, slight tension. They need something to land. "
        "Pause and summarize, or ask directly: 'What's unclear?'"
    ),
    "skepticism_surface": (
        "Asymmetric lip pull, narrowed eyes—that's skepticism leaking through. "
        "Name it: 'You look unconvinced—what's your concern?' Address it head-on."
    ),
    "objection_moment": (
        "Tension in the face, maybe compressed lips. An objection is brewing. "
        "Preempt it: 'I'm sensing something—what's on your mind?' Let them voice it."
    ),
    "resistance_peak": (
        "Face is tight, gaze averted, posture pulling back. Resistance is high. "
        "Don't push. Acknowledge: 'This might not feel right yet. What would help?'"
    ),
    "hesitation_moment": (
        "Micro-tension, slight freeze—they're on the fence. "
        "Lower the stakes: offer a smaller step, or ask what's holding them back."
    ),
    "disengagement_risk": (
        "Gaze drifting, face going flat. Attention is slipping away. "
        "Re-engage now: ask a direct question or shift to something that matters to them."
    ),
    "objection_fading": (
        "Tension is easing, face softening. The objection is losing steam. "
        "Don't reargue it. Gently reinforce value and suggest the next step."
    ),
    "aha_moment": (
        "Eyes widened, brows lifted, genuine smile—that's insight landing. They just 'got it.' "
        "Capitalize: 'It sounds like this clicked—how does it fit with what you need?'"
    ),
    "re_engagement_opportunity": (
        "They were drifting, but something just caught their attention—slight head turn, eyes refocused. "
        "Strike now: ask for their input or deliver your key point."
    ),
    "alignment_cue": (
        "Nodding, sustained eye contact, relaxed face—they're with you. "
        "Reinforce shared goals: 'So we're on the same page that...' and propose next step."
    ),
    "genuine_interest": (
        "Authentic engagement—Duchenne markers, forward lean, active eye contact. "
        "This is real curiosity. Go deeper or ask: 'What's most interesting to you?'"
    ),
    "listening_active": (
        "Eyes locked, body still, face open—they're absorbing every word. "
        "This is your moment. Deliver your most important point and pause for reaction."
    ),
    "trust_building_moment": (
        "Face is open, gaze steady, slight nod. Trust is forming. "
        "Be transparent—acknowledge a limitation or invite their concerns. Authenticity deepens this."
    ),
    "urgency_sensitive": (
        "They're asking about timing, face is alert. Urgency is on their mind. "
        "Be specific: give clear dates or ask what timeline works for them."
    ),
    "processing_deep": (
        "Stillness, gaze inward, brow slightly furrowed—deep processing mode. "
        "Give them space. Then reinforce one key point that matters most."
    ),
    "attention_peak": (
        "Eyes wide, face engaged, body oriented toward you. Attention is maxed out. "
        "Deliver your most important message right now. Don't dilute it."
    ),
    "rapport_moment": (
        "Genuine smile, mirroring your posture, warm eye contact. Rapport is high. "
        "Leverage it—strengthen the connection, then steer toward the outcome you want."
    ),
}

# Shared tone instructions: human business coach helping with meetings.
# Insights feel coach-generated—brief justification, clear next step—not fact reports.
_TONE_INSTRUCTIONS = (
    "VOICE: You are a human business coach sitting alongside the host, helping them read the room. "
    "Sound like a trusted advisor—warm, specific, human. Do NOT present raw facts or data. "
    "Lead with insight: what this moment means for the meeting. Give a short justification only when it helps. "
    "Always end with a clear next step: what the host should say or do right now. "
    "The host should feel coached, not informed. No jargon or corporate-speak. "
    "REAL-TIME: Ground responses in the cues below, but translate them into coach-style guidance—never list metrics or facts to the user."
)
# Brevity for insight popups: short, actionable, clear path forward—no text dumps.
_POPUP_BREVITY = (
    "LENGTH: Keep insight popups short—no more than 2–3 sentences. "
    "Structure: (1) Brief insight with short justification when helpful. (2) Clear next step. "
    "Prioritize what the host should DO over what you detected. Conversational yet professional."
)

# Max chars of last-context bundle to include so insights are tailored to the most recent context sent to OpenAI.
_LAST_CONTEXT_CAP = 2800

# Human-readable labels for signifiers (subset used in prompts)
_SIGNIFIER_LABELS: Dict[str, str] = {
    "g1_duchenne": "Duchenne smile", "g1_pupil_dilation": "Pupil dilation", "g1_eyebrow_flash": "Eyebrow flash",
    "g1_eye_contact": "Eye contact", "g1_head_tilt": "Head tilt", "g1_forward_lean": "Forward lean",
    "g2_look_up_lr": "Look up", "g2_lip_pucker": "Lip pucker", "g2_eye_squint": "Eye squint",
    "g2_thinking_brow": "Thinking brow", "g2_stillness": "Stillness", "g2_lowered_brow": "Lowered brow",
    "g3_contempt": "Contempt", "g3_lip_compression": "Lip compression", "g3_gaze_aversion": "Gaze aversion",
    "g3_jaw_clench": "Jaw clench", "g3_rapid_blink": "Rapid blinking", "g4_relaxed_exhale": "Relaxed exhale",
    "g4_fixed_gaze": "Fixed gaze", "g4_smile_transition": "Smile transition",
}
_COMPOSITE_LABELS: Dict[str, str] = {
    "cognitive_load_multimodal": "Cognitive load", "rapport_engagement": "Rapport",
    "skepticism_objection_strength": "Skepticism", "disengagement_risk_multimodal": "Disengagement risk",
    "decision_readiness_multimodal": "Decision readiness",
}
_ACOUSTIC_TAG_LABELS: Dict[str, str] = {
    "acoustic_uncertainty": "Voice uncertainty", "acoustic_tension": "Vocal tension",
    "acoustic_disengagement_risk": "Vocal disengagement",
}


def _build_rich_context_parts(
    signifier_scores: Optional[Dict[str, float]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
    recent_speech_tags: Optional[list] = None,
    acoustic_tags: Optional[list] = None,
) -> list:
    """Build prompt lines from facial signifiers, composite metrics, speech tags, and acoustic tags."""
    parts: list = []
    if signifier_scores:
        elevated = [(k, v) for k, v in signifier_scores.items() if isinstance(v, (int, float)) and float(v) >= 50]
        elevated.sort(key=lambda x: -float(x[1]))
        if elevated:
            items = [f"{_SIGNIFIER_LABELS.get(k, k)}: {int(v)}" for k, v in elevated[:10]]
            parts.append("ELEVATED FACIAL SIGNIFIERS: " + "; ".join(items) + ".")
    if composite_metrics:
        notable = [(k, v) for k, v in composite_metrics.items() if isinstance(v, (int, float)) and 40 <= float(v) <= 100]
        notable.sort(key=lambda x: -abs(float(x[1]) - 50))
        if notable:
            items = [f"{_COMPOSITE_LABELS.get(k, k)}: {int(v)}" for k, v in notable[:6]]
            parts.append("COMPOSITE SIGNALS: " + "; ".join(items) + ".")
    if recent_speech_tags:
        recent = [f"{t.get('category', '')}: \"{t.get('phrase', '')}\"" for t in (recent_speech_tags[-5:]) if t.get("category")]
        if recent:
            parts.append("RECENT SPEECH CUES: " + "; ".join(recent) + ".")
    if acoustic_tags:
        labels = [_ACOUSTIC_TAG_LABELS.get(t, t) for t in acoustic_tags[:5]]
        if labels:
            parts.append("VOICE DETECTED: " + "; ".join(labels) + ".")
    return parts


def generate_insight_for_spike(
    group: str,
    metrics_summary: Optional[dict] = None,
    recent_transcript: Optional[str] = None,
    recent_context_bundle: Optional[str] = None,
    signifier_scores: Optional[dict] = None,
    composite_metrics: Optional[dict] = None,
    recent_speech_tags: Optional[list] = None,
    acoustic_tags: Optional[list] = None,
    timeout_sec: float = 4.0,
) -> str:
    """
    Generate a one-sentence, actionable meeting insight using Azure AI Foundry.

    Combines the visual cue (which metric group spiked) with facial signifiers, composite metrics,
    speech tags, acoustic tags, and the most recent context sent to Azure AI Foundry.

    Returns:
        One-sentence insight for the meeting host.
    """
    import time
    weights = get_insight_weights()
    stock = STOCK_MESSAGES.get(
        group,
        "Something just shifted in their expression—it's worth noticing. "
        "Check in with a short question or pause to see what they need before moving on.",
    )
    desc = GROUP_DESCRIPTIONS.get(group, "engagement")
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-800:] if len(transcript) > 800 else transcript  # last 800 chars

    system = (
        "You are a human business coach watching the meeting in real time. You noticed something worth addressing "
        "and need to whisper a short note to the host.\n\n"
        "Do NOT list facts or metrics. Lead with insight: what this moment means and why it matters (brief justification). "
        "Always end with a clear next step—a specific phrase or action the host can use in the next 10 seconds. "
        "Sound like a coach helping them, not a system reporting data. 2–3 sentences max.\n\n"
        f"{_POPUP_BREVITY}\n\n"
        f"{_TONE_INSTRUCTIONS}"
    )
    prompt_suffix = (weights.get("prompt_suffix") or "").strip()
    if prompt_suffix:
        system = system.rstrip() + "\n\n" + prompt_suffix
    user_parts = [
        f"FACIAL CUES DETECTED: {desc}",
    ]
    has_fresh_context = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle_snippet = recent_context_bundle.strip()
        if len(bundle_snippet) > _LAST_CONTEXT_CAP:
            bundle_snippet = bundle_snippet[: _LAST_CONTEXT_CAP] + "\n[...truncated]"
        user_parts.append(
            "REAL-TIME CONTEXT (captured right now—tailor your insight to these exact cues):\n" + bundle_snippet
        )
    if not has_fresh_context:
        user_parts.extend(_build_rich_context_parts(signifier_scores, composite_metrics, recent_speech_tags, acoustic_tags))
    if metrics_summary and not has_fresh_context:
        attn = metrics_summary.get("attention")
        eye = metrics_summary.get("eyeContact")
        expr = metrics_summary.get("facialExpressiveness")
        cues = []
        if attn is not None and attn > 70:
            cues.append("high attention (focused gaze)")
        elif attn is not None and attn < 30:
            cues.append("low attention (gaze drifting)")
        if eye is not None and eye > 70:
            cues.append("strong eye contact")
        elif eye is not None and eye < 30:
            cues.append("weak eye contact")
        if expr is not None and expr > 70:
            cues.append("high expressiveness")
        elif expr is not None and expr < 30:
            cues.append("low expressiveness (guarded)")
        if cues:
            user_parts.append(f"BASIC METRICS: {'; '.join(cues)}.")
    if transcript_snippet and not has_fresh_context:
        user_parts.append(f"SPEECH CONTEXT (what partner just said): \"{transcript_snippet}\"")
    if not has_fresh_context:
        acoustic_ctx = get_recent_acoustic_context()
        if acoustic_ctx:
            user_parts.append(f"VOICE / ACOUSTIC CONTEXT (how they said it): {acoustic_ctx}")
    user_parts.append(
        "Write 2–3 short sentences: brief insight + short justification + clear next step. "
        "Coach voice—what the host should do next, not a fact report."
    )

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]

    max_tokens = weights.get("max_length")
    try:
        out = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=system,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            temperature=0.85,
        )
        if out and isinstance(out, str):
            s = out.strip()
            if s:
                return s
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Insight generation failed: %s", e)
    return stock  # Reserve fallback only when Azure unavailable or fails


def generate_insight_for_aural_trigger(
    category: str,
    phrase: str,
    metrics_summary: Optional[dict] = None,
    recent_context_bundle: Optional[str] = None,
    signifier_scores: Optional[dict] = None,
    composite_metrics: Optional[dict] = None,
    recent_speech_tags: Optional[list] = None,
    acoustic_tags: Optional[list] = None,
) -> str:
    """
    Generate a one-sentence B2B insight when partner says a trigger phrase.
    Uses facial signifiers, composite metrics, speech/acoustic context, and meeting context.
    """
    weights = get_insight_weights()
    stock = AURAL_STOCK_MESSAGES.get(
        category,
        "They just said something that deserves a response. "
        "Acknowledge it briefly and ask what they need or what would help.",
    )
    transcript = get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript

    system = (
        "You are a human business coach watching the meeting in real time. They just said something that matters—"
        "objection, interest, confusion, commitment, concern, or a question—and you need to whisper a short note to the host.\n\n"
        "Do NOT list facts. Lead with insight: what this means and a brief justification. "
        "Always end with a clear next step—a specific phrase or question the host can use now. "
        "Sound like a coach helping them, not a system reporting data. 2–3 sentences max.\n\n"
        f"{_POPUP_BREVITY}\n\n"
        f"{_TONE_INSTRUCTIONS}"
    )
    prompt_suffix = (weights.get("prompt_suffix") or "").strip()
    if prompt_suffix:
        system = system.rstrip() + "\n\n" + prompt_suffix
    category_context = {
        "objection": "pushing back, raising a concern or disagreement",
        "interest": "expressing curiosity or positive reception",
        "confusion": "signaling uncertainty or lack of understanding",
        "commitment": "signaling readiness to move forward or agree",
        "concern": "voicing worry or hesitation",
        "timeline": "asking about timing, deadlines, or delivery",
        "budget": "raising cost, pricing, or investment questions",
        "urgency": "signaling time sensitivity or priority",
        "skepticism": "expressing doubt or need for proof",
        "enthusiasm": "showing strong positive reaction",
        "authority": "deferring to others or decision makers",
        "hesitation": "using hedging or uncertain language",
        "confirmation": "affirming or agreeing",
    }
    user_parts = [
        f"SPEECH CUE: Partner said \"{phrase}\" — they're {category_context.get(category, category)}.",
    ]
    has_fresh_context = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle_snippet = recent_context_bundle.strip()
        if len(bundle_snippet) > _LAST_CONTEXT_CAP:
            bundle_snippet = bundle_snippet[: _LAST_CONTEXT_CAP] + "\n[...truncated]"
        user_parts.append(
            "REAL-TIME CONTEXT (captured right now—tailor your insight to these exact cues):\n" + bundle_snippet
        )
    if not has_fresh_context:
        user_parts.extend(_build_rich_context_parts(signifier_scores, composite_metrics, recent_speech_tags, acoustic_tags))
        if transcript_snippet:
            user_parts.append(f"CONVERSATION CONTEXT: \"{transcript_snippet}\"")
        acoustic_ctx = get_recent_acoustic_context()
        if acoustic_ctx:
            user_parts.append(f"VOICE / ACOUSTIC CONTEXT (how they said it): {acoustic_ctx}")
    if metrics_summary and not has_fresh_context:
        # Add specific metric context
        attn = metrics_summary.get("attention")
        eye = metrics_summary.get("eyeContact")
        expr = metrics_summary.get("facialExpressiveness")
        cues = []
        if attn is not None and attn > 70:
            cues.append("high attention")
        elif attn is not None and attn < 30:
            cues.append("low attention (distracted)")
        if eye is not None and eye > 70:
            cues.append("strong eye contact")
        elif eye is not None and eye < 30:
            cues.append("avoiding eye contact")
        if expr is not None and expr > 70:
            cues.append("animated expression")
        elif expr is not None and expr < 30:
            cues.append("guarded/flat expression")
        if cues:
            user_parts.append(f"FACIAL CUES: {'; '.join(cues)}.")
    user_parts.append(
        "Write 2–3 short sentences: brief insight + short justification + clear next step. "
        "Coach voice—what the host should do next, not a fact report."
    )

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]
    max_tokens = weights.get("max_length")

    try:
        out = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=system,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            temperature=0.85,
        )
        if out and isinstance(out, str):
            s = out.strip()
            if s:
                return s
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Aural insight failed: %s", e)
    return stock  # Reserve fallback only when Azure unavailable or fails


def generate_insight_for_opportunity(
    opportunity_id: str,
    context: Optional[dict] = None,
    metrics_summary: Optional[dict] = None,
    recent_transcript: Optional[str] = None,
    recent_context_bundle: Optional[str] = None,
    signifier_scores: Optional[dict] = None,
    composite_metrics: Optional[dict] = None,
    recent_speech_tags: Optional[list] = None,
    acoustic_tags: Optional[list] = None,
    timeout_sec: float = 4.0,
) -> str:
    """
    Generate a one-sentence B2B insight when an opportunity is detected from engagement metrics.
    Uses facial signifiers, composite metrics, speech/acoustic context, and meeting context.
    """
    weights = get_insight_weights()
    stock = OPPORTUNITY_STOCK_MESSAGES.get(
        opportunity_id,
        "Something just shifted—their expression or tone suggests a moment worth noticing. "
        "Take a beat to name it or ask what's on their mind, then suggest a clear next step so you don't lose the thread.",
    )
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript

    system = (
        "You are a human business coach watching the meeting in real time—face, words, and voice. You noticed a moment of "
        "opportunity (interest, readiness, confusion, resistance, etc.) and need to whisper a short note to the host.\n\n"
        "Do NOT list facts or metrics. Lead with insight: what this moment means and a brief justification. "
        "Always end with a clear next step—a specific phrase or action the host can use now. "
        "Sound like a coach helping them, not a system reporting data. 2–3 sentences max.\n\n"
        f"{_POPUP_BREVITY}\n\n"
        f"{_TONE_INSTRUCTIONS}"
    )
    if opportunity_id in NEGATIVE_OPPORTUNITY_IDS:
        system += (
            "\n\nFor confusion or resistance: brief insight + short justification + one clear, kind next step "
            "(e.g. clarify, acknowledge their concern, or invite them back in). Coach voice—what to do next."
        )
    prompt_suffix = (weights.get("prompt_suffix") or "").strip()
    if prompt_suffix:
        system = system.rstrip() + "\n\n" + prompt_suffix
    # Map opportunity IDs to psychology-grounded descriptions (including multimodal)
    opportunity_context = {
        "decision_readiness_multimodal": "partner's face and words both signal readiness—relaxed gaze, positive expression, commitment/interest language",
        "cognitive_overload_multimodal": "face and speech both show overload—furrowed brow, gaze shifting, confusion or concern in what they said",
        "skepticism_objection_multimodal": "they voiced objection or concern and face backs it—tension, lip compression, averted gaze",
        "aha_insight_multimodal": "expression shifted and they used realization language (e.g. got it, that makes sense)—insight just landed",
        "disengagement_multimodal": "low engagement cues and no recent positive language—attention may be drifting",
        "confusion_multimodal": "face and speech show confusion or concern; voice may signal uncertainty—clarify and simplify",
        "tension_objection_multimodal": "resistance plus objection/concern in words or tense voice—validate then address",
        "loss_of_interest": "interest dropping, no commitment language, flat or withdrawn voice—re-engage",
        "acoustic_disengagement_risk": "low vocal energy, flat pitch—possible withdrawal; re-engage",
        "acoustic_uncertainty": "voice signals uncertainty; may pair with confusion or overload—clarify",
        "acoustic_tension": "vocal tension with resistance cues—acknowledge and address",
        "closing_window": "partner's face has relaxed, tension released—internal decision appears made",
        "decision_ready": "sustained eye contact, relaxed brow, subtle agreement cues—they're ready to commit",
        "ready_to_sign": "open posture, genuine smile, forward lean—all signals say yes",
        "buying_signal": "Duchenne smile (eyes crinkling), raised brows, engaged posture—genuine interest",
        "commitment_cue": "nodding, steady gaze, relaxed face—agreement signals detected",
        "cognitive_overload_risk": "furrowed brow (AU4), gaze aversion, stillness—cognitive load is spiking",
        "confusion_moment": "brow furrow, squint, processing expression—they're lost",
        "need_clarity": "subtle uncertainty signals—something isn't landing",
        "skepticism_surface": "asymmetric lip pull, narrowed eyes—skepticism is showing",
        "objection_moment": "lip compression, tense jaw—an objection is brewing",
        "resistance_peak": "guarded face, averted gaze, closed posture—resistance is high",
        "hesitation_moment": "micro-freeze, tension flicker—they're on the fence",
        "disengagement_risk": "flat affect, gaze drifting—attention is slipping",
        "objection_fading": "face softening, tension releasing—resistance is dropping",
        "aha_moment": "eyes widened, brows raised, genuine smile—insight just landed",
        "re_engagement_opportunity": "attention just snapped back—gaze refocused, head turned",
        "alignment_cue": "nodding, mirroring posture, warm eye contact—they're with you",
        "genuine_interest": "authentic engagement markers—Duchenne smile, forward lean, active listening",
        "listening_active": "eyes locked, body still, face open—absorbing every word",
        "trust_building_moment": "open expression, steady gaze, approachable demeanor—trust is forming",
        "urgency_sensitive": "alertness around timing—urgency is on their mind",
        "processing_deep": "inward gaze, stillness, slight brow furrow—deep processing mode",
        "attention_peak": "wide eyes, engaged face, oriented body—attention is maxed",
        "rapport_moment": "genuine smile, posture mirroring, warm eye contact—rapport is high",
    }
    user_parts = [
        f"OPPORTUNITY DETECTED: {opportunity_id}",
        f"WHAT YOU'RE SEEING: {opportunity_context.get(opportunity_id, context or 'notable shift in expression')}",
    ]
    has_fresh_context = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle_snippet = recent_context_bundle.strip()
        if len(bundle_snippet) > _LAST_CONTEXT_CAP:
            bundle_snippet = bundle_snippet[: _LAST_CONTEXT_CAP] + "\n[...truncated]"
        user_parts.append(
            "REAL-TIME CONTEXT (captured right now—tailor your insight to these exact cues):\n" + bundle_snippet
        )
    if not has_fresh_context:
        sig = signifier_scores or (context.get("signifier_scores") if context else None)
        comp = composite_metrics or (context.get("composite_metrics") if context else None)
        speech = recent_speech_tags or (context.get("recent_speech_tags") if context else None)
        ac = acoustic_tags if acoustic_tags is not None else get_recent_acoustic_tags()
        user_parts.extend(_build_rich_context_parts(sig, comp, speech, ac if ac else None))
    if metrics_summary and not has_fresh_context:
        attn = metrics_summary.get("attention")
        eye = metrics_summary.get("eyeContact")
        expr = metrics_summary.get("facialExpressiveness")
        cues = []
        if attn is not None:
            cues.append(f"attention: {attn:.0f}/100")
        if eye is not None:
            cues.append(f"eye contact: {eye:.0f}/100")
        if expr is not None:
            cues.append(f"expressiveness: {expr:.0f}/100")
        if cues:
            user_parts.append(f"METRICS: {', '.join(cues)}.")
    if transcript_snippet and not has_fresh_context:
        user_parts.append(f"SPEECH CONTEXT: \"{transcript_snippet}\"")
    ac_tags = get_recent_acoustic_tags()
    if not has_fresh_context:
        acoustic_ctx = get_recent_acoustic_context()
        if acoustic_ctx:
            user_parts.append(f"VOICE / ACOUSTIC CONTEXT (how they said it): {acoustic_ctx}")
    if opportunity_id in NEGATIVE_OPPORTUNITY_IDS:
        neg_lines = ["NEGATIVE SIGNALS (use to suggest how to relieve root causes):"]
        if context:
            g1 = context.get("g1")
            g2 = context.get("g2")
            g3 = context.get("g3")
            if g2 is not None and g2 >= 50:
                neg_lines.append(f"  - G2 (cognitive load) elevated: {g2:.0f}/100")
            if g3 is not None:
                r = 100.0 - float(g3)
                if r >= 44:
                    neg_lines.append(f"  - G3 raw (resistance): {r:.0f}/100")
            sig = context.get("signifier_scores") or {}
            elevated = [k for k, v in sig.items() if isinstance(v, (int, float)) and float(v) >= 55]
            if elevated:
                neg_lines.append(f"  - Elevated signifiers: {', '.join(elevated[:8])}")
            speech_tags = context.get("recent_speech_tags") or []
            neg_cats = [t.get("category") for t in speech_tags[-5:] if t.get("category") in ("objection", "concern", "confusion")]
            if neg_cats:
                neg_lines.append(f"  - Recent speech: {', '.join(neg_cats)}")
        if ac_tags:
            neg_lines.append(f"  - Voice tags: {', '.join(ac_tags)}")
        neg_lines.append("  Suggest how to relieve the root causes: e.g. clarify if confusion, validate then address if objection, re-engage if disengagement.")
        user_parts.append("\n".join(neg_lines))
    if context and isinstance(context.get("recent_speech_tags"), list) and context["recent_speech_tags"]:
        tags = context["recent_speech_tags"]
        recent = [f"{t.get('category', '')}: \"{t.get('phrase', '')}\"" for t in tags[-5:] if t.get("category")]
        if recent and opportunity_id not in NEGATIVE_OPPORTUNITY_IDS:
            user_parts.append(f"RECENT SPEECH TAGS (corroborating): {'; '.join(recent)}.")
    user_parts.append(
        "Write 2–3 short sentences: brief insight + short justification + clear next step. "
        "Coach voice—what the host should do next, not a fact report."
    )

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]
    max_tokens = weights.get("max_length")

    svc = get_foundry_service()
    last_err = None
    for api_version_override in [None, "2024-08-01-preview"]:
        try:
            out = svc.chat_completion(
                messages=messages,
                system_prompt=system,
                max_tokens=int(max_tokens) if max_tokens is not None else None,
                temperature=0.85,
                api_version_override=api_version_override,
            )
            if out and isinstance(out, str):
                s = out.strip()
                if s:
                    # #region agent log
                    try:
                        import json as _j, os as _o, time as _t
                        _d = _o.path.dirname(_o.path.abspath(__file__))
                        _p = _o.path.join(_o.path.dirname(_d), ".cursor", "debug.log")
                        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                        with open(_p, "a") as _f:
                            _f.write(_j.dumps({"id": "log_insight_azure_ok", "timestamp": int(_t.time() * 1000), "location": "insight_generator.py:opp", "message": "Azure returned insight", "data": {"source": "azure"}, "hypothesisId": "A"}) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    return s
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "404" in err_str or "Resource not found" in err_str:
                continue
            break
    if last_err:
        import logging
        logging.getLogger(__name__).warning("Opportunity insight failed: %s", last_err)
        # #region agent log
        try:
            import json as _j, os as _o, time as _t
            _d = _o.path.dirname(_o.path.abspath(__file__))
            _p = _o.path.join(_o.path.dirname(_d), ".cursor", "debug.log")
            _o.makedirs(_o.path.dirname(_p), exist_ok=True)
            with open(_p, "a") as _f:
                _f.write(_j.dumps({"id": "log_insight_azure_err", "timestamp": int(_t.time() * 1000), "location": "insight_generator.py:opp", "message": "Azure failed, returning stock", "data": {"err": str(last_err)[:120]}, "hypothesisId": "A"}) + "\n")
        except Exception:
            pass
        # #endregion
    # #region agent log
    try:
        import json as _j, os as _o, time as _t
        _d = _o.path.dirname(_o.path.abspath(__file__))
        _p = _o.path.join(_o.path.dirname(_d), ".cursor", "debug.log")
        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
        with open(_p, "a") as _f:
            _f.write(_j.dumps({"id": "log_insight_stock", "timestamp": int(_t.time() * 1000), "location": "insight_generator.py:opp", "message": "Returning stock fallback", "data": {"opportunity_id": opportunity_id, "stock_preview": (stock or "")[:80]}, "hypothesisId": "A"}) + "\n")
    except Exception:
        pass
    # #endregion
    return stock  # Reserve fallback only when Azure unavailable or fails
