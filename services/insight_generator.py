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
}

# Cooldown between phrase-triggered alerts (seconds)
AURAL_COOLDOWN_SEC = 25
_pending_aural_alert: Optional[dict] = None
_aural_alert_lock = threading.Lock()
_last_aural_trigger_time: float = 0.0


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
# Group descriptions for prompt (visual cue context) — B2B meeting focus
# -----------------------------------------------------------------------------
GROUP_DESCRIPTIONS = {
    "g1": "Interest & engagement (e.g. stronger attention, positive facial cues—buying signals)",
    "g2": "Cognitive load (e.g. thinking hard, processing information—e.g. evaluating proposal)",
    "g3": "Resistance or discomfort (e.g. skepticism, objections, unease)",
    "g4": "Decision-ready (e.g. ready to commit, closing signals—ready to say yes or next step)",
}

# Stock fallbacks if OpenAI fails or times out — B2B meeting context
STOCK_MESSAGES = {
    "g1": "They're showing stronger interest—good moment to deepen the value proposition or ask for their view.",
    "g2": "They look like they're thinking hard—consider pausing or clarifying to avoid overload before your next ask.",
    "g3": "Signs of resistance or discomfort—try acknowledging concerns, addressing objections, or shifting approach.",
    "g4": "They appear ready to decide—offer a clear next step or ask for commitment.",
}

# Stock fallbacks for phrase-triggered (aural) insights
AURAL_STOCK_MESSAGES = {
    "objection": "They raised an objection—acknowledge it directly and offer to address their concern before moving on.",
    "interest": "They expressed interest—capitalize on this moment to deepen the discussion or ask for their perspective.",
    "confusion": "They seem confused—pause and clarify before proceeding; check for understanding.",
    "commitment": "They're signaling commitment—offer a clear next step or ask for a concrete yes.",
    "concern": "They voiced a concern—acknowledge it and explore what would help them feel comfortable.",
    "timeline": "They're asking about timing—provide clarity on milestones or next steps.",
    "budget": "Budget or pricing came up—address it directly and link value to their investment.",
}

# Fallbacks for B2B opportunity-triggered insights (visual/temporal cues)
OPPORTUNITY_STOCK_MESSAGES = {
    "closing_window": "They're in a closing window—offer a clear next step or ask for commitment now.",
    "decision_ready": "They appear decision-ready—present options or ask for a concrete yes.",
    "ready_to_sign": "Signals suggest readiness to move forward—propose the next step or close.",
    "buying_signal": "Strong buying signal—deepen value and ask for their view or commitment.",
    "commitment_cue": "Commitment cues detected—offer a clear next step or ask for agreement.",
    "cognitive_overload_risk": "Cognitive overload risk—pause, simplify, or recap before adding more.",
    "confusion_moment": "Confusion detected—clarify and check understanding before continuing.",
    "need_clarity": "They may need clarity—pause and summarize or ask what to clarify.",
    "skepticism_surface": "Skepticism surfacing—address concerns directly and offer evidence or reassurance.",
    "objection_moment": "Objection moment—acknowledge and address their concern before moving on.",
    "resistance_peak": "Resistance is high—acknowledge their view and pivot or reframe.",
    "hesitation_moment": "Hesitation detected—offer reassurance or a smaller commitment step.",
    "disengagement_risk": "Disengagement risk—re-engage with a question or shift of topic.",
    "objection_fading": "Objections fading—gently reinforce value and suggest next step.",
    "aha_moment": "Aha moment—capitalize on this insight and tie it to your proposal.",
    "re_engagement_opportunity": "Re-engagement opportunity—ask a direct question or invite their input.",
    "alignment_cue": "Alignment cues—reinforce shared goals and propose next step.",
    "genuine_interest": "Genuine interest—go deeper on value and ask for their perspective.",
    "listening_active": "They're actively listening—deliver your key point and ask for reaction.",
    "trust_building_moment": "Trust-building moment—be transparent and invite their concerns.",
    "urgency_sensitive": "They're sensitive to urgency—offer a clear timeline or next step.",
    "processing_deep": "They're processing deeply—pause briefly then reinforce one key point.",
    "attention_peak": "Attention is high—deliver your most important message now.",
    "rapport_moment": "Rapport moment—strengthen connection then steer toward outcome.",
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
        "You are an expert B2B meeting coach observing the meeting in real time. You've just noticed a shift in the "
        "meeting partner's expression (client, prospect, or stakeholder)—a moment that signals interest, cognitive "
        "load/confusion, resistance, or decision-readiness. You're about to pass a quick note to the host. What do "
        "you say?\n\n"
        "Say one to three short lines—the kind of thing you'd whisper to someone sitting next to you. Natural, "
        "conversational, specific to what you're seeing and the business context. Make it actionable so they know "
        "the next step. No preamble, no 'You should' or 'Consider.' Do not use the term B2B. Keep it concise—"
        "2–3 lines max, nothing overwhelming. If you have recent speech from the partner, weave it in—e.g. they "
        "said 'I'm not sure' and you're seeing confusion, so you might say to pause and clarify. Draw on psychology: "
        "cognitive load research, emotional contagion, trust signals. Sound like a real person who cares, not a script."
    )
    user_parts = [
        f"In this meeting, the meeting partner (client/prospect) just showed a sudden shift in: {desc}.",
    ]
    if metrics_summary:
        user_parts.append(f"What you're seeing in the room: {metrics_summary}.")
    if transcript_snippet:
        user_parts.append(f"Recent speech from partner: \"{transcript_snippet}\".")
    user_parts.append("What would you say to the host right now? One to three short lines.")

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
        "You are an expert B2B meeting coach observing the meeting in real time. The meeting partner (client, "
        "prospect, or stakeholder) just said something that caught your attention—an objection, expression of "
        "interest, confusion, commitment, concern, timeline, or budget. You're about to pass a quick note to the "
        "host. What do you say?\n\n"
        "Say one to three short lines—the kind of thing you'd whisper to someone next to you. Weave in the "
        "conversation context so it feels tailored, not generic. Be specific to business meetings. Make it actionable. "
        "Do not use the term B2B. Keep it concise—2–3 lines max, nothing overwhelming. If you know how engaged "
        "they seem (e.g. they're checked out but raised an objection—acknowledge first, then address; they're "
        "leaning in with interest—capitalize on momentum), factor that in. Draw on psychology: validation before "
        "persuasion, cognitive load, trust-building. Sound like a real person who cares, not a script."
    )
    user_parts = [
        f"The partner said: \"{phrase}\" (category: {category}).",
    ]
    if transcript_snippet:
        user_parts.append(f"Recent conversation context: \"{transcript_snippet}\".")
    if metrics_summary:
        user_parts.append(f"What you're seeing in the room: {metrics_summary}.")
    user_parts.append("What would you say to the host right now? One to three short lines.")

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
        "You are an expert B2B meeting coach observing the meeting in real time. You've just noticed something "
        "important—a moment of opportunity. The partner might be ready to close, decision-ready, showing cognitive "
        "overload, surfacing skepticism, having an aha moment, or shifting from objection to openness. These are "
        "psychology-based signals from their expressions and demeanor. You're about to pass a quick note to the host. "
        "What do you say?\n\n"
        "Say one to three short lines—the kind of thing you'd whisper to someone next to you. Tie it to what you're "
        "seeing: the specific opportunity and the broader context. Be specific to business meetings. Make it "
        "actionable so they know the next step. Do not use the term B2B. Keep it concise—2–3 lines max, nothing "
        "overwhelming. "
        "If you have recent speech from the partner, weave it in—e.g. confusion moment plus 'can you clarify' might "
        "lead you to suggest pausing and clarifying. Draw on psychology: decision fatigue, cognitive load, trust "
        "signals, buying signals. Sound like a real person who cares, not a script."
    )
    user_parts = [
        f"What you're noticing: {opportunity_id}.",
        f"Context: {context or {}}.",
    ]
    if metrics_summary:
        user_parts.append(f"What you're seeing in the room: {metrics_summary}.")
    if transcript_snippet:
        user_parts.append(f"Recent partner speech: \"{transcript_snippet}\".")
    user_parts.append("What would you say to the host right now? One to three short lines.")

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
