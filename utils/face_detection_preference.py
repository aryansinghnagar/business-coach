"""
Runtime face detection method preference.

Stores the user-selected backend (MediaPipe or Azure Face API) so only one
is used at a time. Falls back to config.FACE_DETECTION_METHOD when unset.
"""

from typing import Optional

import config

_PREFERENCE: Optional[str] = None

VALID_METHODS = ("mediapipe", "azure_face_api")


def get_face_detection_method() -> str:
    """Return the current face detection method (preference or config default)."""
    # Default to MediaPipe if nothing is set
    method = (_PREFERENCE or config.FACE_DETECTION_METHOD or "mediapipe").lower()
    return method


def set_face_detection_method(method: str) -> str:
    """
    Set the face detection method. Valid: 'mediapipe', 'azure_face_api'.
    Azure is only allowed when config reports it as available.
    Returns the validated method that was set.
    """
    global _PREFERENCE
    m = (method or "").strip().lower()
    if m not in VALID_METHODS:
        raise ValueError("method must be 'mediapipe' or 'azure_face_api'")
    if m == "azure_face_api" and not config.is_azure_face_api_enabled():
        raise ValueError("Azure Face API is not configured or available")
    _PREFERENCE = m
    return _PREFERENCE
