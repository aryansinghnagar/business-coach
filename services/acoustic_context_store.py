"""
In-memory store for acoustic context from meeting partner (or mic) audio.

Accepts windowed acoustic features via append_acoustic_windows(); provides
get_recent_acoustic_context() for insight generation and chat stream.
Uses utils.acoustic_interpreter to turn raw windows into NL summary.
"""

import threading
from collections import deque
from typing import Any, Dict, List

from utils.acoustic_interpreter import interpret_acoustic_windows

# Keep last N windows (e.g. ~1–2 s each -> 15–30 s of context)
ACOUSTIC_MAX_WINDOWS = 20
_acoustic_windows: deque = deque(maxlen=ACOUSTIC_MAX_WINDOWS)
_acoustic_lock = threading.Lock()


def append_acoustic_windows(windows: List[Dict[str, Any]]) -> None:
    """
    Append one or more acoustic feature windows. Call from POST /engagement/acoustic-context.
    """
    if not windows:
        return
    with _acoustic_lock:
        for w in windows:
            if isinstance(w, dict):
                _acoustic_windows.append(w)


def get_recent_acoustic_context() -> str:
    """
    Return a natural-language summary of recent acoustics for prompts (thread-safe).
    """
    with _acoustic_lock:
        snap = list(_acoustic_windows)
    summary, _ = interpret_acoustic_windows(snap)
    return summary or ""


def get_recent_acoustic_tags() -> List[str]:
    """
    Return the list of acoustic tags from the interpreter (e.g. acoustic_uncertainty,
    acoustic_tension, acoustic_disengagement_risk). Thread-safe.
    """
    with _acoustic_lock:
        snap = list(_acoustic_windows)
    _, tags = interpret_acoustic_windows(snap)
    return list(tags) if tags else []


# Negative tags used for acoustic_negative_strength (disengagement, uncertainty, tension, roughness)
_ACOUSTIC_NEGATIVE_TAGS = frozenset({
    "acoustic_disengagement_risk",
    "acoustic_uncertainty",
    "acoustic_tension",
    "acoustic_roughness_proxy",
})


def get_acoustic_negative_strength() -> float:
    """
    Return 0-1 strength from count of negative acoustic tags (for composites).
    Caps at 1.0 when 3+ negative tags present.
    """
    tags = get_recent_acoustic_tags()
    count = sum(1 for t in tags if t in _ACOUSTIC_NEGATIVE_TAGS)
    return min(1.0, count / 3.0)


def clear_acoustic_context() -> None:
    """Clear stored windows (e.g. when engagement stops)."""
    with _acoustic_lock:
        _acoustic_windows.clear()
