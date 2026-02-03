"""
Insight Generator Service

Generates real-time, actionable B2B meeting insights by combining:
- Visual cues: engagement detector metric spikes (facial expression groups G1–G4)
- Speech cues: recent transcript from meeting partner audio (POST /engagement/transcript)
- Phrase-triggered: when partner says B2B-relevant phrases (objections, interest, etc.)

Flow: Spike / phrase / opportunity -> backend fetches context -> Azure OpenAI -> insight.
Falls back to stock message on failure. See docs/DOCUMENTATION.md.
"""

import re
import threading
import time
from collections import deque
from typing import Optional, Tuple

from services.azure_openai import openai_service

# -----------------------------------------------------------------------------
# Phrase-triggered insights (aural cues) — B2B meeting relevance
# -----------------------------------------------------------------------------
# Categories and phrase patterns; match is case-insensitive.
# Order matters: first match wins; put more specific phrases earlier if needed.
PHRASE_CATEGORIES = {
    "objection": [
        r"\bI'm not sure\b", r"\bI have concerns\b", r"\bthat doesn't work\b",
        r"\bwe can't\b", r"\bnot interested\b", r"\bpush back\b", r"\bI disagree\b",
        r"\bthat won't work\b", r"\bno way\b", r"\bnot convinced\b",
    ],
    "interest": [
        r"\bthat's interesting\b", r"\btell me more\b", r"\bI like that\b",
        r"\bmakes sense\b", r"\bsounds good\b", r"\bI see\b", r"\bgot it\b",
        r"\bthat could work\b", r"\binteresting point\b", r"\bgood point\b",
    ],
    "confusion": [
        r"\bI don't understand\b", r"\bcan you clarify\b", r"\bwhat do you mean\b",
        r"\bI'm confused\b", r"\bcan you explain\b", r"\bnot following\b",
        r"\bunclear\b", r"\bcould you repeat\b",
    ],
    "commitment": [
        r"\blet's do it\b", r"\bI'm in\b", r"\bwe'll move forward\b",
        r"\bnext steps\b", r"\bsign off\b", r"\bwe're ready\b", r"\bdeal\b",
        r"\bagreed\b", r"\blet's proceed\b", r"\bmove ahead\b",
    ],
    "concern": [
        r"\bworried about\b", r"\bconcerned\b", r"\bhesitant\b",
        r"\bneed to think\b", r"\bnot ready yet\b", r"\brisky\b",
    ],
    "timeline": [
        r"\bwhen can we\b", r"\bby when\b", r"\btimeline\b", r"\bdeadline\b",
        r"\bhow soon\b", r"\bdelivery date\b",
    ],
    "budget": [
        r"\bbudget\b", r"\bcost\b", r"\bpricing\b", r"\bexpensive\b",
        r"\bafford\b", r"\bprice point\b", r"\binvestment\b",
    ],
    "realization": [
        r"\baha\b", r"\boh I see\b", r"\bthat makes sense\b", r"\bgot it\b",
        r"\bI get it\b", r"\bnow I understand\b", r"\bthat clicks\b", r"\bclicked\b",
    ],
}

# Cooldown between phrase-triggered alerts (seconds)
AURAL_COOLDOWN_SEC = 25
_pending_aural_alert: Optional[dict] = None
_aural_alert_lock = threading.Lock()
_last_aural_trigger_time: float = 0.0

# Recent speech tags for multimodal composite detection (facial + speech)
# Ring buffer: (category, phrase, timestamp); kept for SPEECH_TAGS_WINDOW_SEC
SPEECH_TAGS_WINDOW_SEC = 12.0
_recent_speech_tags: deque = deque(maxlen=50)
_speech_tags_lock = threading.Lock()


def _check_for_trigger_phrases(text: str) -> Optional[Tuple[str, str]]:
    """Return (category, matched_phrase) or None. Case-insensitive."""
    if not text or not text.strip():
        return None
    lower = text.lower().strip()
    for category, patterns in PHRASE_CATEGORIES.items():
        for pat in patterns:
            m = re.search(pat, lower, re.IGNORECASE)
            if m:
                return (category, m.group(0))
    return None


def _set_pending_aural_alert(category: str, phrase: str) -> bool:
    """Set pending aural alert if cooldown has passed. Returns True if set."""
    global _pending_aural_alert, _last_aural_trigger_time
    with _aural_alert_lock:
        now = time.time()
        if now - _last_aural_trigger_time < AURAL_COOLDOWN_SEC:
            return False
        _pending_aural_alert = {"type": "aural", "category": category, "phrase": phrase}
        _last_aural_trigger_time = now
        return True


def append_speech_tag(category: str, phrase: str) -> None:
    """Append a speech tag (category, phrase, timestamp) for multimodal composite detection."""
    with _speech_tags_lock:
        _recent_speech_tags.append({"category": category, "phrase": phrase, "time": time.time()})


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

# Stock fallbacks if OpenAI fails or times out — Psychology-grounded
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


def generate_insight_for_spike(
    group: str,
    metrics_summary: Optional[dict] = None,
    recent_transcript: Optional[str] = None,
    timeout_sec: float = 4.0,
) -> str:
    """
    Generate a one-sentence, actionable meeting insight using Azure OpenAI.

    Combines the visual cue (which metric group spiked) with optional speech
    context (recent transcript from meeting partner) to produce a tailored
    suggestion (e.g. 'aha' moment, confusion, ready to close).

    Args:
        group: Metric group that spiked ('g1', 'g2', 'g3', 'g4').
        metrics_summary: Optional dict with engagement metrics for context.
        recent_transcript: Optional recent speech from meeting partner.
        timeout_sec: Max time to wait for OpenAI (then use stock message).

    Returns:
        One-sentence insight for the meeting host.
    """
    import time
    stock = STOCK_MESSAGES.get(group, "Notable change in engagement—consider checking in with them.")
    desc = GROUP_DESCRIPTIONS.get(group, "engagement")
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-800:] if len(transcript) > 800 else transcript  # last 800 chars

    system = (
        "You are an expert meeting coach trained in nonverbal communication, FACS (Facial Action Coding System), "
        "and behavioral psychology. You're observing a meeting in real time—watching the partner's face and listening "
        "to their words. You just noticed a significant shift in their facial expression that reveals their internal state.\n\n"
        "Your job: pass a quick, insightful note to the host—the kind of thing a trusted advisor would whisper. "
        "Ground your insight in what you're *actually observing*:\n"
        "- For interest/engagement: mention genuine smiling (AU6+AU12), raised brows, head tilt, eye contact\n"
        "- For cognitive load: mention furrowed brow (AU4), reduced blinking, gaze aversion, stillness\n"
        "- For resistance: mention lip compression (AU23/24), asymmetric expressions, nose wrinkle (AU9), guarded face\n"
        "- For decision-ready: mention relaxed face, tension release, sustained eye contact, nodding\n\n"
        "Connect the observed cue to psychology: Kahneman's System 1/2, Ekman's microexpressions, approach/avoidance "
        "motivation, cognitive load theory, trust signals from Cialdini. If there's speech context, weave it in—"
        "e.g. if they said 'I'm not sure' and you see confusion, suggest clarifying.\n\n"
        "Rules:\n"
        "- 1-3 short lines, conversational tone, no preamble\n"
        "- Be specific: describe what you're seeing ('their brow just furrowed,' 'that's a genuine smile')\n"
        "- Make it actionable: tell them what to do next\n"
        "- Do not use 'B2B' or jargon\n"
        "- Sound like a real person who genuinely cares about their success"
    )
    user_parts = [
        f"FACIAL CUES DETECTED: {desc}",
    ]
    if metrics_summary:
        # Add specific metric context for more grounded insights
        attn = metrics_summary.get("attention")
        eye = metrics_summary.get("eyeContact")
        expr = metrics_summary.get("facialExpressiveness")
        cues = []
        if attn is not None and attn > 70:
            cues.append("high attention (focused gaze, minimal distraction)")
        elif attn is not None and attn < 30:
            cues.append("low attention (gaze drifting, distracted)")
        if eye is not None and eye > 70:
            cues.append("strong eye contact (engaged, present)")
        elif eye is not None and eye < 30:
            cues.append("weak eye contact (avoidant or processing internally)")
        if expr is not None and expr > 70:
            cues.append("high expressiveness (emotional engagement visible)")
        elif expr is not None and expr < 30:
            cues.append("low expressiveness (guarded or flat affect)")
        if cues:
            user_parts.append(f"Additional cues: {'; '.join(cues)}.")
    if transcript_snippet:
        user_parts.append(f"SPEECH CONTEXT (what partner just said): \"{transcript_snippet}\"")
    user_parts.append("What would you whisper to the host right now? Be specific about what you're seeing.")

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]

    try:
        out = openai_service.chat_completion(
            messages=messages,
            system_prompt=system,
        )
        if out and isinstance(out, str):
            s = out.strip()
            if s:
                return s
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Insight generation failed: %s", e)
    return stock


def generate_insight_for_aural_trigger(
    category: str,
    phrase: str,
    metrics_summary: Optional[dict] = None,
) -> str:
    """
    Generate a one-sentence B2B insight when partner says a trigger phrase.
    Uses full conversation context (recent transcript) and current engagement metrics.
    """
    stock = AURAL_STOCK_MESSAGES.get(
        category, "Notable comment from the partner—consider responding to their point."
    )
    transcript = get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript

    system = (
        "You are an expert meeting coach trained in verbal and nonverbal communication, behavioral psychology, and "
        "persuasion science. You're observing a meeting in real time—listening to the partner's words and watching "
        "their face. They just said something significant that reveals their state: an objection, expression of "
        "interest, confusion, commitment signal, concern, timeline question, or budget mention.\n\n"
        "Your job: pass a quick, insightful note to the host—the kind of thing a trusted advisor would whisper. "
        "Ground your insight in both what they *said* and (if available) what their face is showing:\n"
        "- Objection + tense face → validate first, then address\n"
        "- Interest + genuine smile → momentum is building, capitalize\n"
        "- Confusion + furrowed brow → they're overloaded, simplify and clarify\n"
        "- Commitment language + relaxed face → they're ready, close it\n"
        "- Concern + guarded expression → unmet need, explore what they need to feel safe\n\n"
        "Draw on psychology: validation before persuasion (Rogers), cognitive load theory (Sweller), reciprocity "
        "and commitment (Cialdini), emotional labeling (FBI negotiation). If they said something specific, echo it "
        "back in your insight to make it feel tailored.\n\n"
        "Rules:\n"
        "- 1-3 short lines, conversational tone, no preamble\n"
        "- Be specific: quote or paraphrase what they said, describe what you see\n"
        "- Make it actionable: tell them what to do next\n"
        "- Do not use 'B2B' or jargon\n"
        "- Sound like a real person who genuinely cares about their success"
    )
    category_context = {
        "objection": "pushing back, raising a concern or disagreement",
        "interest": "expressing curiosity or positive reception",
        "confusion": "signaling uncertainty or lack of understanding",
        "commitment": "signaling readiness to move forward or agree",
        "concern": "voicing worry or hesitation",
        "timeline": "asking about timing, deadlines, or delivery",
        "budget": "raising cost, pricing, or investment questions",
    }
    user_parts = [
        f"SPEECH CUE: Partner said \"{phrase}\" — they're {category_context.get(category, category)}.",
    ]
    if transcript_snippet:
        user_parts.append(f"CONVERSATION CONTEXT: \"{transcript_snippet}\"")
    if metrics_summary:
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
    user_parts.append("What would you whisper to the host? Be specific—connect what they said to what you see.")

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]

    try:
        out = openai_service.chat_completion(
            messages=messages,
            system_prompt=system,
        )
        if out and isinstance(out, str):
            s = out.strip()
            if s:
                return s
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Aural insight failed: %s", e)
    return stock


def generate_insight_for_opportunity(
    opportunity_id: str,
    context: Optional[dict] = None,
    metrics_summary: Optional[dict] = None,
    recent_transcript: Optional[str] = None,
    timeout_sec: float = 4.0,
) -> str:
    """
    Generate a one-sentence B2B insight when an opportunity is detected from engagement metrics.
    Uses Azure OpenAI with opportunity type, context (G1–G4, signifiers), and optional transcript.
    """
    stock = OPPORTUNITY_STOCK_MESSAGES.get(
        opportunity_id,
        "Opportunity detected—consider acting on it.",
    )
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript

    system = (
        "You are an expert meeting coach trained in FACS (Facial Action Coding System), behavioral psychology, and "
        "sales/negotiation science. You're observing a meeting in real time—watching the partner's face and listening "
        "to their words. You just detected a significant *moment of opportunity* based on their facial expressions "
        "and behavioral patterns.\n\n"
        "Your job: pass a quick, insightful note to the host—the kind of thing a trusted advisor would whisper. "
        "Ground your insight in what you're *actually observing* and the specific opportunity type:\n"
        "- Closing window / decision-ready: relaxed face, tension release, sustained eye contact, subtle nods\n"
        "- Buying signal / genuine interest: Duchenne smile (AU6+AU12), raised brows, forward lean\n"
        "- Cognitive overload: furrowed brow (AU4), gaze aversion, stillness, overwhelmed expression\n"
        "- Skepticism / resistance: asymmetric lip pull (contempt), lip compression (AU23/24), nose wrinkle (AU9)\n"
        "- Aha moment: eyes widen, brows raise, genuine smile—insight just landed\n"
        "- Objection fading: tension releasing, face softening, guarded expression opening up\n"
        "- Disengagement risk: flat affect, gaze drifting, reduced expressiveness\n\n"
        "Draw on psychology: Kahneman's System 1/2, Ekman's microexpressions, Cialdini's commitment/consistency, "
        "decision fatigue, approach-avoidance motivation. If there's speech context, weave it in.\n\n"
        "Rules:\n"
        "- 1-3 short lines, conversational tone, no preamble\n"
        "- Be specific: describe what you're seeing and why it matters\n"
        "- Make it actionable: tell them what to do in this moment\n"
        "- Do not use 'B2B' or jargon\n"
        "- Sound like a real person who genuinely cares about their success"
    )
    # Map opportunity IDs to psychology-grounded descriptions (including multimodal)
    opportunity_context = {
        "decision_readiness_multimodal": "partner's face and words both signal readiness—relaxed gaze, positive expression, commitment/interest language",
        "cognitive_overload_multimodal": "face and speech both show overload—furrowed brow, gaze shifting, confusion or concern in what they said",
        "skepticism_objection_multimodal": "they voiced objection or concern and face backs it—tension, lip compression, averted gaze",
        "aha_insight_multimodal": "expression shifted and they used realization language (e.g. got it, that makes sense)—insight just landed",
        "disengagement_multimodal": "low engagement cues and no recent positive language—attention may be drifting",
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
    if metrics_summary:
        # Add specific metric context
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
    if transcript_snippet:
        user_parts.append(f"SPEECH CONTEXT: \"{transcript_snippet}\"")
    # For multimodal opportunities, include recent speech tags that triggered corroboration
    if context and isinstance(context.get("recent_speech_tags"), list) and context["recent_speech_tags"]:
        tags = context["recent_speech_tags"]
        recent = [f"{t.get('category', '')}: \"{t.get('phrase', '')}\"" for t in tags[-5:] if t.get("category")]
        if recent:
            user_parts.append(f"RECENT SPEECH TAGS (corroborating): {'; '.join(recent)}.")
    user_parts.append("What would you whisper to the host? Be specific about the opportunity and what to do.")

    user_content = " ".join(user_parts)
    messages = [{"role": "user", "content": user_content}]

    try:
        out = openai_service.chat_completion(
            messages=messages,
            system_prompt=system,
        )
        if out and isinstance(out, str):
            s = out.strip()
            if s:
                return s
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Opportunity insight failed: %s", e)
    return stock
