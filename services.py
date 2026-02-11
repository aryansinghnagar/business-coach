"""
Business Meeting Copilot — consolidated services.

Single module containing: acoustic context store, Azure Face API, Azure Foundry,
Azure Speech, insight generator, and engagement API. All service logic lives here.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
import config

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Acoustic context store and interpretation
# =============================================================================

LOUDNESS_LOW = 0.15
LOUDNESS_HIGH = 0.6
PITCH_VARIABILITY_HIGH_ST = 2.0
PITCH_VARIABILITY_LOW_ST = 0.5
VOICING_CONFIDENCE_THRESHOLD = 0.5
TONE_ROUGHNESS_THRESHOLD = 0.65
PITCH_CREAKY_PROXY_HZ = 120
PITCH_RISING = "rising"
PITCH_FALLING = "falling"


def interpret_acoustic_windows(windows: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Turn acoustic feature windows into a short NL summary and tags."""
    if not windows:
        return "", []
    active = [w for w in windows if w.get("speech_active", True)]
    if not active:
        return "", []

    tags: List[str] = []
    parts: List[str] = []
    loudness_vals = [float(w["loudness_norm"]) for w in active if "loudness_norm" in w and w["loudness_norm"] is not None]
    pitch_vals = [float(w["pitch_hz"]) for w in active if "pitch_hz" in w and w["pitch_hz"] and w["pitch_hz"] > 0]
    contours = [w.get("pitch_contour") for w in active if w.get("pitch_contour")]
    variability_vals = [float(w["pitch_variability"]) for w in active if "pitch_variability" in w and w["pitch_variability"] is not None]
    tone_vals = [float(w["tone_proxy"]) for w in active if "tone_proxy" in w and w["tone_proxy"] is not None]
    voicing_vals = [float(w["voicing_confidence"]) for w in active if "voicing_confidence" in w and w["voicing_confidence"] is not None]

    avg_loudness = sum(loudness_vals) / len(loudness_vals) if loudness_vals else 0.0
    avg_pitch = sum(pitch_vals) / len(pitch_vals) if pitch_vals else None
    avg_variability = sum(variability_vals) / len(variability_vals) if variability_vals else 0.0
    avg_tone = sum(tone_vals) / len(tone_vals) if tone_vals else None
    avg_voicing = sum(voicing_vals) / len(voicing_vals) if voicing_vals else None
    pitch_trusted = avg_voicing is None or avg_voicing >= VOICING_CONFIDENCE_THRESHOLD

    rising_count = sum(1 for c in contours if c == PITCH_RISING)
    falling_count = sum(1 for c in contours if c == PITCH_FALLING)
    n_contours = len(contours)

    if avg_loudness < LOUDNESS_LOW and (not contours or (falling_count + rising_count) < n_contours / 2):
        parts.append("Low vocal energy and relatively flat pitch suggest possible disengagement or withdrawal; consider re-engaging.")
        tags.append("acoustic_disengagement_risk")
    if pitch_trusted and avg_loudness >= LOUDNESS_HIGH and avg_pitch and avg_pitch > 180:
        parts.append("Elevated loudness and higher pitch indicate heightened arousal; good moment for emphasis or closing.")
        tags.append("acoustic_arousal_high")
    if pitch_trusted and (avg_variability >= PITCH_VARIABILITY_HIGH_ST or (n_contours and rising_count >= max(1, n_contours // 2))):
        parts.append("Rising pitch contour and/or high pitch variability suggest possible uncertainty or questioning; consider clarifying.")
        tags.append("acoustic_uncertainty")
    if pitch_trusted and avg_tone is not None and avg_tone > 0.6 and avg_pitch and avg_pitch > 160:
        parts.append("Vocal tension and elevated pitch may signal objection or stress; acknowledge and address.")
        tags.append("acoustic_tension")
    if avg_tone is not None and avg_tone > TONE_ROUGHNESS_THRESHOLD and avg_variability >= PITCH_VARIABILITY_HIGH_ST:
        parts.append("Vocal roughness proxy suggests strain; may indicate tension or discomfort.")
        tags.append("acoustic_roughness_proxy")
    if pitch_trusted and n_contours and falling_count >= max(1, n_contours // 2):
        if "acoustic_uncertainty" not in tags:
            parts.append("Falling pitch contour can signal finality or resolution; watch for closing cues.")
        tags.append("acoustic_falling_contour")
    if pitch_trusted and avg_variability < PITCH_VARIABILITY_LOW_ST and n_contours:
        flat_count = n_contours - rising_count - falling_count
        if flat_count >= max(1, n_contours // 2):
            parts.append("Very flat pitch contour suggests monotone delivery; may indicate disengagement or cognitive load.")
            tags.append("acoustic_monotone")
    if pitch_trusted and avg_loudness >= LOUDNESS_HIGH and n_contours and rising_count >= max(1, n_contours // 2):
        if "acoustic_arousal_high" not in tags:
            tags.append("acoustic_emphasis_proxy")
    if pitch_trusted and avg_pitch and avg_pitch < PITCH_CREAKY_PROXY_HZ and avg_variability < PITCH_VARIABILITY_LOW_ST:
        parts.append("Low pitch with minimal variation may suggest creaky voice; can signal hesitation or low energy.")
        tags.append("acoustic_creakiness_proxy")
    if avg_tone is not None and avg_tone > 0.7 and avg_variability < 1.0 and pitch_trusted:
        tags.append("acoustic_breathiness_proxy")
    words_per_sec_vals = [float(w["words_per_sec"]) for w in active if "words_per_sec" in w and w["words_per_sec"] is not None]
    if words_per_sec_vals:
        avg_wps = sum(words_per_sec_vals) / len(words_per_sec_vals)
        if avg_wps > 3.0:
            tags.append("acoustic_speech_rate_high")
        elif avg_wps < 1.5:
            tags.append("acoustic_speech_rate_low")

    summary = " ".join(parts).strip() if parts else ""
    return summary, tags


ACOUSTIC_MAX_WINDOWS = 20
_acoustic_windows: deque = deque(maxlen=ACOUSTIC_MAX_WINDOWS)
_acoustic_lock = threading.Lock()
_ACOUSTIC_NEGATIVE_TAGS = frozenset({
    "acoustic_disengagement_risk", "acoustic_uncertainty", "acoustic_tension", "acoustic_roughness_proxy",
})


def append_acoustic_windows(windows: List[Dict[str, Any]]) -> None:
    """Append one or more acoustic feature windows."""
    if not windows:
        return
    with _acoustic_lock:
        for w in windows:
            if isinstance(w, dict):
                _acoustic_windows.append(w)


def get_recent_acoustic_context() -> str:
    """Return a natural-language summary of recent acoustics (thread-safe)."""
    with _acoustic_lock:
        snap = list(_acoustic_windows)
    summary, _ = interpret_acoustic_windows(snap)
    return summary or ""


def get_recent_acoustic_tags() -> List[str]:
    """Return the list of acoustic tags (thread-safe)."""
    with _acoustic_lock:
        snap = list(_acoustic_windows)
    _, tags = interpret_acoustic_windows(snap)
    return list(tags) if tags else []


def get_recent_acoustic_tags_and_negative_strength() -> Tuple[List[str], float]:
    """Return (tags, negative_strength) in one lock and one interpret pass."""
    with _acoustic_lock:
        snap = list(_acoustic_windows)
    _, tags = interpret_acoustic_windows(snap)
    tag_list = list(tags) if tags else []
    count = sum(1 for t in tag_list if t in _ACOUSTIC_NEGATIVE_TAGS)
    return (tag_list, min(1.0, count / 3.0))


def get_acoustic_negative_strength() -> float:
    """Return 0-1 strength from count of negative acoustic tags."""
    tags = get_recent_acoustic_tags()
    count = sum(1 for t in tags if t in _ACOUSTIC_NEGATIVE_TAGS)
    return min(1.0, count / 3.0)


def clear_acoustic_context() -> None:
    """Clear stored windows (e.g. when engagement stops)."""
    with _acoustic_lock:
        _acoustic_windows.clear()


# =============================================================================
# 2. Azure Face API
# =============================================================================

def _optional_cv2_numpy():
    import cv2
    import numpy as np
    return cv2, np


class AzureFaceAPIService:
    """Service for Azure Face API: face detection, landmarks, emotion/head pose."""

    def __init__(self):
        if not config.is_azure_face_api_enabled():
            raise ValueError(
                "Azure Face API is not configured. "
                "Set AZURE_FACE_API_KEY and AZURE_FACE_API_ENDPOINT."
            )
        self.api_key = config.AZURE_FACE_API_KEY.strip() if config.AZURE_FACE_API_KEY else ""
        self.endpoint = config.AZURE_FACE_API_ENDPOINT.strip().rstrip("/") if config.AZURE_FACE_API_ENDPOINT else ""
        self.region = config.AZURE_FACE_API_REGION
        if not self.endpoint or not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("Invalid Azure Face API endpoint")
        if not self.api_key:
            raise ValueError("Azure Face API key is empty")
        self.api_version = "v1.0"
        if "/face/" in self.endpoint.lower():
            if self.endpoint.endswith("/face") or self.endpoint.endswith("/face/"):
                self.detect_url = f"{self.endpoint}/{self.api_version}/detect"
            else:
                base_endpoint = self.endpoint.split("/face")[0].rstrip("/")
                self.detect_url = f"{base_endpoint}/face/{self.api_version}/detect"
        else:
            self.detect_url = f"{self.endpoint}/face/{self.api_version}/detect"
        self.headers = {"Ocp-Apim-Subscription-Key": self.api_key, "Content-Type": "application/octet-stream"}

    def detect_faces(
        self,
        image: Any,
        return_face_landmarks: bool = True,
        return_face_attributes: bool = True,
    ) -> List[Dict[str, Any]]:
        cv2, np = _optional_cv2_numpy()
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Invalid image")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected BGR image (H, W, 3)")
        h, w = image.shape[:2]
        if h < 36 or w < 36:
            raise ValueError("Image too small (min 36x36)")
        if h > 4096 or w > 4096:
            scale = min(4096 / w, 4096 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            h, w = new_h, new_w
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success or buffer is None:
            raise ValueError("Failed to encode image")
        image_data = buffer.tobytes()
        if len(image_data) > 6 * 1024 * 1024:
            success, buffer = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success or buffer is None:
                raise ValueError("Image too large (max 6MB)")
            image_data = buffer.tobytes()
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "true" if return_face_landmarks else "false",
            "returnFaceAttributes": (
                "age,gender,emotion,headPose,smile,facialHair,glasses,hair,makeup,occlusion,accessories,blur,exposure,noise"
                if return_face_attributes else "false"
            ),
        }
        try:
            response = requests.post(self.detect_url, headers=self.headers, params=params, data=image_data, timeout=10)
            if response.status_code != 200:
                err = response.json().get("error", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
                raise requests.RequestException(err.get("message", f"Status {response.status_code}"))
            faces = response.json()
            return faces if isinstance(faces, list) else []
        except requests.RequestException:
            raise
        except Exception as e:
            raise ValueError(f"Azure Face API error: {e}") from e

    def extract_landmarks_from_face(self, face_data: Dict[str, Any]) -> Optional[Any]:
        """Extract facial landmarks from Azure Face API response (N, 2) or (N, 3)."""
        if "faceLandmarks" not in face_data:
            return None
        landmarks = face_data["faceLandmarks"]
        if not isinstance(landmarks, dict):
            return None
        cv2, np = _optional_cv2_numpy()
        landmark_keys = [
            "pupilLeft", "pupilRight", "noseTip", "mouthLeft", "mouthRight",
            "eyebrowLeftOuter", "eyebrowLeftInner", "eyeLeftOuter", "eyeLeftTop", "eyeLeftBottom", "eyeLeftInner",
            "eyebrowRightInner", "eyebrowRightOuter", "eyeRightInner", "eyeRightTop", "eyeRightBottom", "eyeRightOuter",
            "noseRootLeft", "noseRootRight", "noseLeftAlarTop", "noseRightAlarTop",
            "noseLeftAlarOutTip", "noseRightAlarOutTip", "upperLipTop", "upperLipBottom", "underLipTop", "underLipBottom",
        ]
        points = []
        for key in landmark_keys:
            if key in landmarks:
                pt = landmarks[key]
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    try:
                        points.append([float(pt["x"]), float(pt["y"])])
                    except (ValueError, TypeError):
                        continue
        if len(points) < 10:
            return None
        return np.array(points, dtype=np.float32)

    def extract_emotion_from_face(self, face_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        attrs = face_data.get("faceAttributes") or {}
        return attrs.get("emotion")

    def extract_head_pose_from_face(self, face_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        attrs = face_data.get("faceAttributes") or {}
        return attrs.get("headPose")

    def get_face_rectangle(self, face_data: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        rect = face_data.get("faceRectangle")
        if not rect:
            return None
        return (rect["left"], rect["top"], rect["width"], rect["height"])


_azure_face_api_service: Optional[AzureFaceAPIService] = None


def get_azure_face_api_service() -> Optional[AzureFaceAPIService]:
    """Get or create the global Azure Face API service instance."""
    global _azure_face_api_service
    if _azure_face_api_service is None and config.is_azure_face_api_enabled():
        try:
            _azure_face_api_service = AzureFaceAPIService()
        except Exception as e:
            logger.warning("Azure Face API init failed: %s", e)
            return None
    return _azure_face_api_service


# =============================================================================
# 3. Azure AI Foundry
# =============================================================================

def _optional_openai():
    from openai import AzureOpenAI
    return AzureOpenAI


class AzureFoundryService:
    """Service for Azure AI Foundry: chat completions and streaming."""

    def __init__(self):
        AzureOpenAI = _optional_openai()
        self.client = AzureOpenAI(
            azure_endpoint=config.AZURE_FOUNDRY_ENDPOINT,
            api_key=config.AZURE_FOUNDRY_KEY,
            api_version=config.AZURE_FOUNDRY_API_VERSION,
        )
        self.endpoint = config.AZURE_FOUNDRY_ENDPOINT
        self.deployment_name = config.FOUNDRY_DEPLOYMENT_NAME
        self.api_key = config.AZURE_FOUNDRY_KEY

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_version_override: Optional[str] = None,
    ) -> str:
        client = self.client
        if api_version_override:
            AzureOpenAI = _optional_openai()
            client = AzureOpenAI(
                azure_endpoint=config.AZURE_FOUNDRY_ENDPOINT,
                api_key=config.AZURE_FOUNDRY_KEY,
                api_version=api_version_override,
            )
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + list(messages)
        kwargs = {"model": self.deployment_name, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        enable_oyd: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        if system_prompt is None:
            system_prompt = config.SYSTEM_PROMPT
        if not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + list(messages)
        base = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment_name}/"
        if enable_oyd and config.is_cognitive_search_enabled():
            url = base + "extensions/chat/completions?api-version=2023-06-01-preview"
        else:
            url = base + "chat/completions?api-version=2023-06-01-preview"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        body = {"messages": messages, "stream": True}
        if enable_oyd and config.is_cognitive_search_enabled():
            body["dataSources"] = [{
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": config.AZURE_COG_SEARCH_ENDPOINT,
                    "key": config.AZURE_COG_SEARCH_API_KEY,
                    "indexName": config.AZURE_COG_SEARCH_INDEX_NAME,
                    "semanticConfiguration": "",
                    "queryType": "simple",
                    "fieldsMapping": {
                        "contentFieldsSeparator": "\n",
                        "contentFields": ["content"],
                        "filepathField": None,
                        "titleField": "title",
                        "urlField": None,
                    },
                    "inScope": True,
                    "roleInformation": system_prompt,
                },
            }]
        response = requests.post(url, headers=headers, json=body, stream=True)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                elif line.startswith("data:"):
                    yield line + "\n\n"
                else:
                    yield "data: " + line + "\n\n"
        yield "data: [DONE]\n\n"


_foundry_service: Optional[AzureFoundryService] = None


def get_foundry_service() -> AzureFoundryService:
    """Return the Azure AI Foundry service (lazy singleton)."""
    global _foundry_service
    if _foundry_service is None:
        _foundry_service = AzureFoundryService()
    return _foundry_service


def get_openai_service() -> AzureFoundryService:
    """Deprecated: use get_foundry_service()."""
    return get_foundry_service()


# =============================================================================
# 4. Azure Speech
# =============================================================================

class AzureSpeechService:
    """Service for Azure Speech: STT token only."""

    def __init__(self):
        self.speech_key = config.SPEECH_KEY
        self.speech_region = config.SPEECH_REGION

    def get_speech_token(self) -> Dict[str, str]:
        url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {"Ocp-Apim-Subscription-Key": self.speech_key}
        resp = requests.post(url, headers=headers, timeout=5)
        resp.raise_for_status()
        token = resp.text.strip()
        if not token:
            raise ValueError("Empty token from Azure Speech")
        return {"token": token, "region": self.speech_region}


_speech_service: Optional[AzureSpeechService] = None


def get_speech_service() -> AzureSpeechService:
    """Return the Speech service (lazy singleton)."""
    global _speech_service
    if _speech_service is None:
        _speech_service = AzureSpeechService()
    return _speech_service


# =============================================================================
# 5. Insight generator (phrase cues, transcript, spike/opportunity/aural insights)
# =============================================================================

from helpers import (
    NEGATIVE_OPPORTUNITY_IDS,
    METRIC_CRITICAL,
    METRIC_HIGH,
    ContextGenerator,
    COMPOSITE_LABELS as CONTEXT_COMPOSITE_LABELS,
)

DEFAULT_INSIGHT_WEIGHTS: Dict[str, Any] = {"prompt_suffix": "", "max_length": 100, "opportunity_thresholds": {}}


def get_insight_weights() -> Dict[str, Any]:
    return dict(DEFAULT_INSIGHT_WEIGHTS)


PHRASE_CATEGORIES = {
    "objection": [r"\bI'm not sure\b", r"\bI have concerns\b", r"\bthat doesn't work\b", r"\bwe can't\b", r"\bnot interested\b", r"\bpush back\b", r"\bI disagree\b", r"\bthat won't work\b", r"\bno way\b", r"\bnot convinced\b", r"\bwe might need to reconsider\b", r"\bthat's a non-starter\b", r"\bdoesn't fit\b", r"\bcan't agree\b", r"\bnot on board\b", r"\bthat won't fly\b", r"\bwe'll have to pass\b"],
    "interest": [r"\bthat's interesting\b", r"\btell me more\b", r"\bI like that\b", r"\bI'd like to hear more\b", r"\bmakes sense\b", r"\bsounds good\b", r"\bI see\b", r"\bgot it\b", r"\bthat could work\b", r"\binteresting point\b", r"\bgood point\b", r"\bthat resonates\b", r"\bworth exploring\b"],
    "confusion": [r"\bI don't understand\b", r"\bcan you clarify\b", r"\bwhat do you mean\b", r"\bI'm confused\b", r"\bcan you explain\b", r"\bnot following\b", r"\bunclear\b", r"\bcould you repeat\b", r"\blost me\b", r"\bnot sure I follow\b", r"\bcan you walk me through\b", r"\bwhat does that mean\b", r"\bhow so\b"],
    "commitment": [r"\blet's do it\b", r"\bI'm in\b", r"\bI'm ready to move forward\b", r"\bwe'll move forward\b", r"\bnext steps\b", r"\bwhat's the next step\b", r"\bsign off\b", r"\bwe're ready\b", r"\bdeal\b", r"\bagreed\b", r"\blet's proceed\b", r"\bmove ahead\b", r"\bwe're good to go\b", r"\blet's lock this in\b", r"\bcount me in\b"],
    "concern": [r"\bworried about\b", r"\bconcerned\b", r"\bhesitant\b", r"\bneed to think\b", r"\bnot ready yet\b", r"\brisky\b", r"\bI'll need to run this by\b", r"\blet me circle back\b", r"\bwe're still evaluating\b", r"\bneed to discuss with\b", r"\bsome reservations\b", r"\bnot quite there\b", r"\bholding off\b"],
    "timeline": [r"\bwhen can we\b", r"\bby when\b", r"\btimeline\b", r"\bdeadline\b", r"\bhow soon\b", r"\bdelivery date\b", r"\bwhen would we\b", r"\bstart date\b", r"\blaunch date\b", r"\bhow long will it take\b"],
    "budget": [r"\bbudget\b", r"\bcost\b", r"\bpricing\b", r"\bexpensive\b", r"\bafford\b", r"\bprice point\b", r"\binvestment\b", r"\bROI\b", r"\btotal cost\b", r"\bwhat's the price\b", r"\bpaying for\b"],
    "realization": [r"\baha\b", r"\boh I see\b", r"\bthat makes sense\b", r"\bgot it\b", r"\bI get it\b", r"\bnow I understand\b", r"\bthat clicks\b", r"\bclicked\b", r"\bthat clarifies it\b", r"\bmakes more sense now\b", r"\bI see what you mean\b"],
    "urgency": [r"\basap\b", r"\bneed it soon\b", r"\btime[- ]?sensitive\b", r"\burgent\b", r"\bpriority\b", r"\brush\b", r"\bdeadline\b", r"\bsoon(?:er)?\s+than\b"],
    "skepticism": [r"\bsounds too good\b", r"\bprove it\b", r"\bshow me\b", r"\bwe'll see\b", r"\bI'll believe it when\b", r"\btoo good to be true\b", r"\bneed to see\b", r"\bconvince me\b", r"\breally\?\b"],
    "enthusiasm": [r"\blove it\b", r"\bexcited about\b", r"\bcan't wait\b", r"\bgreat idea\b", r"\bthat's great\b", r"\bawesome\b", r"\bfantastic\b", r"\bperfect\b", r"\bexactly what we need\b", r"\bbrilliant\b"],
    "authority": [r"\bI'll have to check with\b", r"\bmy team\b", r"\bthe board\b", r"\bdecision makers\b", r"\bneed to run this by\b", r"\bhave to get approval\b", r"\bmy manager\b", r"\bstakeholders\b", r"\bneed sign[- ]?off\b"],
    "hesitation": [r"\bum\b", r"\buh\b", r"\bwell\b", r"\bI guess\b", r"\bsort of\b", r"\bmaybe\b", r"\bI'm not sure\b", r"\bkind of\b", r"\bperhaps\b", r"\bit depends\b", r"\bmight\b", r"\bcould be\b"],
    "confirmation": [r"\bexactly\b", r"\bright\b", r"\bcorrect\b", r"\bthat's it\b", r"\bprecisely\b", r"\byes\b", r"\babsolutely\b", r"\bthat's right\b", r"\bspot on\b", r"\bexactly right\b"],
}

DISCOURSE_MARKERS = {
    "positive": [r"\bfortunately\b", r"\bthankfully\b", r"\bactually\b(?!\s+(not|don't|can't|won't))", r"\bdefinitely\b", r"\babsolutely\b"],
    "negative": [r"\bunfortunately\b", r"\bfrankly\b", r"\bhonestly\b", r"\bactually\s+(not|don't|can't|won't)", r"\bto be honest\b", r"\bthe problem is\b"],
    "hedging": [r"\bbasically\b", r"\bI mean\b", r"\byou know\b", r"\banyway\b", r"\bkind of\b", r"\bsort of\b", r"\bperhaps\b", r"\bmaybe\b"],
}

AURAL_COOLDOWN_SEC = 15
_pending_aural_alert: Optional[dict] = None
_aural_alert_lock = threading.Lock()
_last_aural_trigger_time: float = 0.0
_last_aural_trigger_category: Optional[str] = None
_llm_fallback_calls: deque = deque(maxlen=10)
_llm_fallback_lock = threading.Lock()
SPEECH_TAGS_WINDOW_SEC = 12.0
_recent_speech_tags: deque = deque(maxlen=50)
_speech_tags_lock = threading.Lock()


def _check_for_trigger_phrases(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not text or not text.strip():
        return out
    lower = text.lower().strip()
    seen = set()
    for category, patterns in PHRASE_CATEGORIES.items():
        if category in seen:
            continue
        for pat in patterns:
            m = re.search(pat, lower, re.IGNORECASE)
            if m:
                out.append((category, m.group(0)))
                seen.add(category)
                break
    return out


def _check_discourse_marker(text: str) -> Optional[Tuple[str, str]]:
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
    if not text or not text.strip():
        return False
    if len(text.strip().split()) > 15:
        return True
    return _check_discourse_marker(text) is not None


def _llm_classify_phrase(text: str) -> Optional[Tuple[str, str]]:
    if not getattr(config, "SPEECH_CUE_LLM_FALLBACK_ENABLED", False):
        return None
    rate = getattr(config, "SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN", 2)
    with _llm_fallback_lock:
        now = time.time()
        recent = [t for t in list(_llm_fallback_calls) if t >= now - 60.0]
        if len(recent) >= rate:
            return None
        _llm_fallback_calls.clear()
        _llm_fallback_calls.extend(recent)
        _llm_fallback_calls.append(now)
    try:
        svc = get_foundry_service()
        prompt = (
            "Classify this B2B meeting transcript into exactly one category or 'none'. "
            "Categories: objection, interest, confusion, commitment, concern, timeline, budget, "
            "realization, urgency, skepticism, enthusiasm, authority, hesitation, confirmation.\n\n"
            f"Snippet: {text[:500].strip()}"
        )
        resp = svc.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=10, temperature=0.0)
        if not resp or not isinstance(resp, str):
            return None
        cat = resp.strip().lower().replace(".", "")
        if cat == "none" or cat not in PHRASE_CATEGORIES:
            return None
        return (cat, text[:80].strip() + ("..." if len(text) > 80 else ""))
    except Exception:
        return None


def _discourse_aligns_with_category(polarity: str, category: str) -> bool:
    neg = {"objection", "concern", "confusion"}
    pos = {"interest", "commitment", "realization"}
    if polarity == "negative" and category in neg:
        return True
    if polarity == "positive" and category in pos:
        return True
    if polarity == "hedging" and category in {"concern", "confusion"}:
        return True
    return False


def get_discourse_boost(text: str, category: str) -> bool:
    dm = _check_discourse_marker(text)
    return dm is not None and _discourse_aligns_with_category(dm[0], category)


def check_speech_cues(text: str) -> List[Tuple[str, str]]:
    matches = _check_for_trigger_phrases(text)
    if matches:
        return matches
    if _should_try_llm_fallback(text):
        llm_result = _llm_classify_phrase(text)
        if llm_result:
            return [llm_result]
    return []


def append_speech_tag(category: str, phrase: str, discourse_boost: bool = False) -> None:
    with _speech_tags_lock:
        _recent_speech_tags.append({"category": category, "phrase": phrase, "time": time.time(), "discourse_boost": discourse_boost})


def get_recent_speech_tags(within_sec: float = SPEECH_TAGS_WINDOW_SEC) -> list:
    now = time.time()
    cutoff = now - within_sec
    with _speech_tags_lock:
        return [t for t in list(_recent_speech_tags) if t["time"] >= cutoff]


def clear_recent_speech_tags() -> None:
    with _speech_tags_lock:
        _recent_speech_tags.clear()


def get_pending_aural_alert() -> Optional[dict]:
    with _aural_alert_lock:
        return _pending_aural_alert


def get_and_clear_pending_aural_alert() -> Optional[dict]:
    global _pending_aural_alert
    with _aural_alert_lock:
        a = _pending_aural_alert
        _pending_aural_alert = None
        return a


def clear_pending_aural_alert() -> None:
    global _pending_aural_alert
    with _aural_alert_lock:
        _pending_aural_alert = None


_TRANSCRIPT_MAX_CHARS = 1200
_transcript_buffer: deque = deque(maxlen=1)
_transcript_lock = threading.Lock()


def append_transcript(text: str) -> None:
    if not text or not text.strip():
        return
    with _transcript_lock:
        current = (_transcript_buffer[0] if _transcript_buffer else "") or ""
        current = (current + " " + text.strip()).strip()
        if len(current) > _TRANSCRIPT_MAX_CHARS:
            current = current[-_TRANSCRIPT_MAX_CHARS:]
        _transcript_buffer.clear()
        _transcript_buffer.append(current)


def get_recent_transcript() -> str:
    with _transcript_lock:
        return (_transcript_buffer[0] if _transcript_buffer else "") or ""


def clear_transcript() -> None:
    with _transcript_lock:
        _transcript_buffer.clear()


GROUP_DESCRIPTIONS = {
    "g1": "Interest & Engagement — Duchenne smile, head tilt, eye contact, eyebrow raises, forward lean.",
    "g2": "Cognitive Load — Furrowed brow, reduced blinking, lip compression, gaze aversion, stillness.",
    "g3": "Resistance & Discomfort — Asymmetric expressions, compressed lips, gaze aversion, guarded.",
    "g4": "Decision-Ready — Sustained eye contact, relaxed face, genuine smile, forward lean, nodding.",
}

STOCK_MESSAGES = {
    "g1": "Their face just lit up—genuine smile, eyes engaged. Lean in: ask what's resonating or go deeper.",
    "g2": "Furrowed brow, stillness—they're processing hard. Give them a beat. Pause, then ask what's on their mind.",
    "g3": "Tension in the face—compressed lips, maybe a slight pull. Acknowledge: 'I'm sensing some hesitation—what's your concern?'",
    "g4": "Relaxed face, steady eye contact, slight nod—they've made up their mind. Offer a clear next step or ask for commitment.",
}

AURAL_STOCK_MESSAGES = {
    "objection": "They just pushed back. Validate first, then address the concern directly.",
    "interest": "They're leaning in verbally—go deeper or ask what caught their attention.",
    "confusion": "They're lost. Simplify, recap, or ask 'What would help clarify this?'",
    "commitment": "They signaled readiness. Lock it in: confirm next step and timeline.",
    "concern": "They voiced a worry. Name it and address it directly.",
    "timeline": "Be specific: give dates, milestones, or ask what timeline works.",
    "budget": "Reframe cost as investment. Tie to outcomes they care about.",
    "urgency": "Be concrete: specific dates or ask what timeline works.",
    "skepticism": "Address head-on: offer proof points, demos, or references.",
    "enthusiasm": "Build on it: ask what's resonating or propose next step.",
    "authority": "Help them sell internally: arm with one-pagers, talking points.",
    "hesitation": "Lower the stakes: offer a smaller step or ask what would help.",
    "confirmation": "Reinforce the point and move toward commitment.",
}

OPPORTUNITY_STOCK_MESSAGES = {
    "decision_readiness_multimodal": "They're showing commitment in words and expression. Ask for next step or confirmation now.",
    "cognitive_overload_multimodal": "Face and words point to overload. Pause, simplify, or ask what would help clarify.",
    "skepticism_objection_multimodal": "They voiced concern and face backs it up. Address directly: 'What's your main concern?'",
    "aha_insight_multimodal": "Something just landed—expression shifted, 'got it' or 'that makes sense.' Capitalize.",
    "disengagement_multimodal": "Low engagement and no recent positive language. Re-engage with a direct question.",
    "loss_of_interest": "Interest dropping. Re-engage: ask what matters to them.",
    "closing_window": "Face has settled—tension released. Ask for commitment or next step now.",
    "decision_ready": "Relaxed brow, sustained gaze. Ask: 'Does this work for you?' or 'What do you need to move forward?'",
    "ready_to_sign": "All signals say yes. Propose the next step clearly.",
    "buying_signal": "Duchenne smile, raised brows. Go deeper or reinforce value.",
    "commitment_cue": "Nodding, eye contact, relaxed face. Summarize and ask for confirmation.",
    "cognitive_overload_risk": "Furrowed brow, gaze aversion. Stop adding. Pause or simplify.",
    "confusion_moment": "Brow furrow, squint. Ask: 'What would help clarify this?'",
    "need_clarity": "Subtle uncertainty. Pause and summarize, or ask 'What's unclear?'",
    "skepticism_surface": "Name it: 'You look unconvinced—what's your concern?'",
    "objection_moment": "Preempt: 'I'm sensing something—what's on your mind?'",
    "resistance_peak": "Don't push. Acknowledge: 'What would help?'",
    "hesitation_moment": "Lower the stakes: offer a smaller step.",
    "disengagement_risk": "Re-engage now: ask a direct question.",
    "objection_fading": "Don't reargue. Reinforce value and suggest next step.",
    "aha_moment": "Capitalize: 'How does this fit with what you need?'",
    "re_engagement_opportunity": "Strike now: ask for input or deliver key point.",
    "alignment_cue": "Reinforce shared goals and propose next step.",
    "genuine_interest": "Go deeper or ask what's most interesting.",
    "listening_active": "Deliver your most important point and pause.",
    "trust_building_moment": "Be transparent; invite concerns.",
    "urgency_sensitive": "Be specific: give clear dates.",
    "processing_deep": "Give space. Reinforce one key point.",
    "attention_peak": "Deliver your most important message now.",
    "rapport_moment": "Leverage it; steer toward outcome.",
}

_TONE_INSTRUCTIONS = (
    "VOICE: One sentence only. Simple, concise, actionable. Base advice strictly on context. No jargon, no raw metrics."
)
_POPUP_BREVITY = "LENGTH: Exactly one sentence. One clear, actionable next step. No filler, no preamble."
_INSIGHT_RESEARCH_SNIPPET = (
    "BASE ADVICE ON CONTEXT ONLY. One sentence: insight + next step. "
    "Resistance → validate then address. Confusion → clarify. Decision readiness → suggest next step."
)
_INSIGHT_PRINCIPLE_SUMMARY = "One sentence: simple, concise, actionable. One insight, one next step."
_LAST_CONTEXT_CAP = 2800

_SIGNIFIER_LABELS: Dict[str, str] = {
    "g1_duchenne": "Duchenne smile", "g1_eye_contact": "Eye contact", "g1_head_tilt": "Head tilt", "g1_forward_lean": "Forward lean",
    "g2_look_up_lr": "Look up", "g2_lip_pucker": "Lip pucker", "g2_thinking_brow": "Thinking brow", "g2_stillness": "Stillness",
    "g3_contempt": "Contempt", "g3_lip_compression": "Lip compression", "g3_gaze_aversion": "Gaze aversion",
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


def _build_metrics_cues(metrics_summary: Optional[dict], label_prefix: str = "BASIC METRICS") -> Optional[str]:
    if not metrics_summary:
        return None
    attn = metrics_summary.get("attention")
    eye = metrics_summary.get("eyeContact")
    expr = metrics_summary.get("facialExpressiveness")
    cues = []
    if attn is not None and attn >= METRIC_HIGH:
        cues.append("high attention (focused gaze)")
    elif attn is not None and attn < METRIC_CRITICAL:
        cues.append("low attention (gaze drifting)")
    if eye is not None and eye >= METRIC_HIGH:
        cues.append("strong eye contact")
    elif eye is not None and eye < METRIC_CRITICAL:
        cues.append("weak eye contact")
    if expr is not None and expr >= METRIC_HIGH:
        cues.append("high expressiveness")
    elif expr is not None and expr < METRIC_CRITICAL:
        cues.append("low expressiveness (guarded)")
    if not cues:
        return None
    return f"{label_prefix}: {'; '.join(cues)}."


def _build_rich_context_parts(
    signifier_scores: Optional[Dict[str, float]] = None,
    composite_metrics: Optional[Dict[str, float]] = None,
    recent_speech_tags: Optional[list] = None,
    acoustic_tags: Optional[list] = None,
) -> list:
    parts = []
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
        recent = [f"{t.get('category', '')}: \"{t.get('phrase', '')}\"" for t in recent_speech_tags[-5:] if t.get("category")]
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
    weights = get_insight_weights()
    stock = STOCK_MESSAGES.get(group, "Something just shifted—check in with a short question or pause.")
    desc = GROUP_DESCRIPTIONS.get(group, "engagement")
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-800:] if len(transcript) > 800 else transcript
    system = (
        "You are a real-time meeting coach. Reply in exactly one sentence: simple, concise, actionable. Base ONLY on context below.\n\n"
        f"{_POPUP_BREVITY}\n\n{_TONE_INSTRUCTIONS}\n\n{_INSIGHT_RESEARCH_SNIPPET}"
    )
    if (weights.get("prompt_suffix") or "").strip():
        system = system.rstrip() + "\n\n" + (weights.get("prompt_suffix") or "").strip()
    user_parts = [f"FACIAL CUES DETECTED: {desc}"]
    has_fresh = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle = recent_context_bundle.strip()[: _LAST_CONTEXT_CAP] + ("\n[...truncated]" if len(recent_context_bundle) > _LAST_CONTEXT_CAP else "")
        user_parts.append("REAL-TIME CONTEXT:\n" + _INSIGHT_PRINCIPLE_SUMMARY + "\n\n" + bundle)
    if not has_fresh:
        user_parts.extend(_build_rich_context_parts(signifier_scores, composite_metrics, recent_speech_tags, acoustic_tags))
    if metrics_summary and not has_fresh:
        cue = _build_metrics_cues(metrics_summary, "BASIC METRICS")
        if cue:
            user_parts.append(cue)
    if transcript_snippet and not has_fresh:
        user_parts.append(f"SPEECH CONTEXT: \"{transcript_snippet}\"")
    if not has_fresh:
        ac = get_recent_acoustic_context()
        if ac:
            user_parts.append(f"VOICE / ACOUSTIC: {ac}")
    user_parts.append("Reply in one sentence: simple, concise, actionable (insight + next step).")
    messages = [{"role": "user", "content": " ".join(user_parts)}]
    try:
        out = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=system,
            max_tokens=min(int(weights.get("max_length") or 80), 80),
            temperature=0.85,
        )
        if out and isinstance(out, str) and out.strip():
            return out.strip()
    except Exception as e:
        logger.warning("Insight generation failed: %s", e)
    return stock


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
    weights = get_insight_weights()
    stock = AURAL_STOCK_MESSAGES.get(category, "They said something that deserves a response. Acknowledge and ask what would help.")
    transcript = get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript
    system = (
        "You are a real-time meeting coach. Reply in exactly one sentence: simple, concise, actionable. They just said something that matters.\n\n"
        + _POPUP_BREVITY + "\n\n" + _TONE_INSTRUCTIONS + "\n\n" + _INSIGHT_RESEARCH_SNIPPET
    )
    if (weights.get("prompt_suffix") or "").strip():
        system = system.rstrip() + "\n\n" + (weights.get("prompt_suffix") or "").strip()
    user_parts = [f"SPEECH CUE: Partner said \"{phrase}\" — category: {category}."]
    has_fresh = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle = recent_context_bundle.strip()[: _LAST_CONTEXT_CAP] + ("\n[...truncated]" if len(recent_context_bundle) > _LAST_CONTEXT_CAP else "")
        user_parts.append("REAL-TIME CONTEXT:\n" + _INSIGHT_PRINCIPLE_SUMMARY + "\n\n" + bundle)
    if not has_fresh:
        user_parts.extend(_build_rich_context_parts(signifier_scores, composite_metrics, recent_speech_tags, acoustic_tags))
        if transcript_snippet:
            user_parts.append(f"CONVERSATION: \"{transcript_snippet}\"")
        ac = get_recent_acoustic_context()
        if ac:
            user_parts.append(f"VOICE / ACOUSTIC: {ac}")
    if metrics_summary and not has_fresh:
        cue = _build_metrics_cues(metrics_summary, "FACIAL CUES")
        if cue:
            user_parts.append(cue)
    user_parts.append("Reply in one sentence: simple, concise, actionable (insight + next step).")
    messages = [{"role": "user", "content": " ".join(user_parts)}]
    try:
        out = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=system,
            max_tokens=min(int(weights.get("max_length") or 80), 80),
            temperature=0.85,
        )
        if out and isinstance(out, str) and out.strip():
            return out.strip()
    except Exception as e:
        logger.warning("Aural insight failed: %s", e)
    return stock


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
    weights = get_insight_weights()
    stock = OPPORTUNITY_STOCK_MESSAGES.get(
        opportunity_id,
        "Something shifted—take a beat to name it or ask what's on their mind, then suggest a clear next step.",
    )
    transcript = (recent_transcript or "").strip() or get_recent_transcript()
    transcript_snippet = transcript[-600:] if len(transcript) > 600 else transcript
    system = (
        "You are a real-time meeting coach. Reply in exactly one sentence: simple, concise, actionable. A moment of opportunity was detected.\n\n"
        + _POPUP_BREVITY + "\n\n" + _TONE_INSTRUCTIONS + "\n\n" + _INSIGHT_RESEARCH_SNIPPET
    )
    if opportunity_id in NEGATIVE_OPPORTUNITY_IDS:
        system += "\n\nOne sentence: insight + clear next step."
    if (weights.get("prompt_suffix") or "").strip():
        system = system.rstrip() + "\n\n" + (weights.get("prompt_suffix") or "").strip()
    user_parts = [f"OPPORTUNITY DETECTED: {opportunity_id}", f"WHAT YOU'RE SEEING: {context or 'notable shift'}"]
    has_fresh = recent_context_bundle and "[CURRENT ENGAGEMENT STATE]" in (recent_context_bundle or "")
    if recent_context_bundle and recent_context_bundle.strip():
        bundle = recent_context_bundle.strip()[: _LAST_CONTEXT_CAP] + ("\n[...truncated]" if len(recent_context_bundle) > _LAST_CONTEXT_CAP else "")
        user_parts.append("REAL-TIME CONTEXT:\n" + _INSIGHT_PRINCIPLE_SUMMARY + "\n\n" + bundle)
    if not has_fresh:
        sig = signifier_scores or (context.get("signifier_scores") if context else None)
        comp = composite_metrics or (context.get("composite_metrics") if context else None)
        speech = recent_speech_tags or (context.get("recent_speech_tags") if context else None)
        ac = acoustic_tags if acoustic_tags is not None else get_recent_acoustic_tags()
        user_parts.extend(_build_rich_context_parts(sig, comp, speech, ac or None))
    if metrics_summary and not has_fresh:
        cue = _build_metrics_cues(metrics_summary, "METRICS")
        if cue:
            user_parts.append(cue)
    if transcript_snippet and not has_fresh:
        user_parts.append(f"SPEECH CONTEXT: \"{transcript_snippet}\"")
    ac_tags = get_recent_acoustic_tags()
    if not has_fresh:
        ac_ctx = get_recent_acoustic_context()
        if ac_ctx:
            user_parts.append(f"VOICE / ACOUSTIC: {ac_ctx}")
    if opportunity_id in NEGATIVE_OPPORTUNITY_IDS and context:
        neg_lines = ["NEGATIVE SIGNALS:"]
        g2, g3 = context.get("g2"), context.get("g3")
        if g2 is not None and g2 >= 50:
            neg_lines.append(f"  G2 (cognitive load): {g2:.0f}/100")
        if g3 is not None and (100.0 - float(g3)) >= 44:
            neg_lines.append(f"  G3 raw (resistance): {100.0 - float(g3):.0f}/100")
        if ac_tags:
            neg_lines.append("  Voice tags: " + ", ".join(ac_tags))
        neg_lines.append("  Suggest how to relieve root causes.")
        user_parts.append("\n".join(neg_lines))
    user_parts.append("Reply in one sentence: simple, concise, actionable (insight + next step).")
    messages = [{"role": "user", "content": " ".join(user_parts)}]
    max_tok = min(int(weights.get("max_length") or 80), 80)
    svc = get_foundry_service()
    last_err = None
    for api_ver in [None, "2024-08-01-preview"]:
        try:
            out = svc.chat_completion(
                messages=messages,
                system_prompt=system,
                max_tokens=max_tok,
                temperature=0.85,
                api_version_override=api_ver,
            )
            if out and isinstance(out, str) and out.strip():
                return out.strip()
        except Exception as e:
            last_err = e
            if "404" in str(e) or "Resource not found" in str(e):
                continue
            break
    if last_err:
        logger.warning("Opportunity insight failed: %s", last_err)
    return stock


# =============================================================================
# 6. Engagement API (detector singleton, context store, idle tracking, context bundle)
# =============================================================================


_last_request_time: float = 0.0
_request_lock = threading.Lock()
_IDLE_THRESHOLD_SEC = 60.0


def update_last_request() -> None:
    global _last_request_time
    with _request_lock:
        _last_request_time = time.time()


def is_idle(threshold_sec: float = _IDLE_THRESHOLD_SEC) -> bool:
    with _request_lock:
        if _last_request_time == 0.0:
            return False
        return (time.time() - _last_request_time) > threshold_sec


engagement_detector = None  # type: Optional[Any]
_context_generator: Optional[ContextGenerator] = None

engagement_context_store: Dict[str, Any] = {
    "last_context_sent": None,
    "last_response": None,
    "last_response_timestamp": None,
    "pending_user_context": None,
}

SIGNIFIER_LABELS = {
    "g1_duchenne": "Duchenne smile", "g1_eye_contact": "Eye contact", "g1_head_tilt": "Head tilt", "g1_forward_lean": "Forward lean",
    "g2_look_up_lr": "Look up", "g2_lip_pucker": "Lip pucker", "g2_thinking_brow": "Thinking brow", "g2_stillness": "Stillness",
    "g3_contempt": "Contempt", "g3_lip_compression": "Lip compression", "g3_gaze_aversion": "Gaze aversion",
    "g4_fixed_gaze": "Fixed gaze", "g4_smile_transition": "Smile transition",
}
COMPOSITE_LABELS = {k: (v.split("(")[0].strip() if "(" in v else v) for k, v in CONTEXT_COMPOSITE_LABELS.items()}
ACOUSTIC_LABELS = {
    "acoustic_uncertainty": "Voice uncertainty",
    "acoustic_tension": "Vocal tension",
    "acoustic_disengagement_risk": "Vocal disengagement",
}


def get_detector():
    return engagement_detector


def get_context_store() -> Dict[str, Any]:
    return engagement_context_store


def get_context_generator() -> ContextGenerator:
    global _context_generator
    if _context_generator is None:
        _context_generator = ContextGenerator()
    return _context_generator


def build_engagement_context_bundle(additional_context: Optional[str] = None) -> str:
    effective_additional = additional_context
    if not (effective_additional and isinstance(effective_additional, str) and effective_additional.strip()):
        effective_additional = engagement_context_store.pop("pending_user_context", None)
    state = None
    det = get_detector()
    if det:
        try:
            state = det.get_current_state()
        except Exception:
            pass
    cg = get_context_generator()
    if not state or not state.face_detected:
        ctx = cg.generate_context_no_face()
    else:
        ctx = state.context
    acoustic_summary = get_recent_acoustic_context() or ""
    acoustic_tags = get_recent_acoustic_tags() or []
    composite_metrics = getattr(state, "composite_metrics", None) or {} if state else {}
    return cg.build_context_bundle_for_foundry(
        ctx,
        acoustic_summary=acoustic_summary,
        acoustic_tags=acoustic_tags,
        additional_context=effective_additional,
        persistently_low_line=None,
        composite_metrics=composite_metrics or None,
    )


def build_fresh_insight_context(
    state: Any,
    signifier_scores: dict,
    composite_metrics: dict,
    metrics_summary: dict,
    recent_speech_tags: list,
    acoustic_tags_list: list,
) -> str:
    sections = []
    cg = get_context_generator()
    if state and state.face_detected:
        ctx = state.context
        sections.append(
            "[CURRENT ENGAGEMENT STATE]\n"
            f"Score: {float(state.score):.0f}/100 | Level: {state.level.name if state.level else 'UNKNOWN'}\n"
            f"Summary: {ctx.summary}\n"
            f"Key indicators: {'; '.join(ctx.key_indicators[:5])}\n"
            f"Suggested actions: {'; '.join(ctx.suggested_actions[:3])}\n"
            f"Risks: {'; '.join(ctx.risk_factors[:3]) if ctx.risk_factors else 'None'}\n"
            f"Opportunities: {'; '.join(ctx.opportunities[:3]) if ctx.opportunities else 'None'}\n"
            "[/CURRENT ENGAGEMENT STATE]"
        )
    else:
        sections.append("[CURRENT ENGAGEMENT STATE]\nNo face detected. Assume neutral engagement.\n[/CURRENT ENGAGEMENT STATE]")
    if metrics_summary:
        cues = []
        if metrics_summary.get("attention") is not None:
            cues.append(f"attention={metrics_summary['attention']:.0f}")
        if metrics_summary.get("eyeContact") is not None:
            cues.append(f"eye_contact={metrics_summary['eyeContact']:.0f}")
        if metrics_summary.get("facialExpressiveness") is not None:
            cues.append(f"expressiveness={metrics_summary['facialExpressiveness']:.0f}")
        if cues:
            sections.append(f"[BASIC METRICS]\n{', '.join(cues)}\n[/BASIC METRICS]")
    if signifier_scores:
        elevated = [(k, v) for k, v in signifier_scores.items() if isinstance(v, (int, float)) and float(v) >= 50]
        elevated.sort(key=lambda x: -float(x[1]))
        if elevated:
            items = [f"{SIGNIFIER_LABELS.get(k, k)}={int(v)}" for k, v in elevated[:10]]
            sections.append(f"[ELEVATED FACIAL SIGNIFIERS]\n{'; '.join(items)}\n[/ELEVATED FACIAL SIGNIFIERS]")
    if composite_metrics:
        notable = [(k, v) for k, v in composite_metrics.items() if isinstance(v, (int, float)) and 35 <= float(v) <= 100]
        notable.sort(key=lambda x: -abs(float(x[1]) - 50))
        if notable:
            items = [f"{COMPOSITE_LABELS.get(k, k)}={int(v)}" for k, v in notable[:10]]
            sections.append(f"[COMPOSITE SIGNALS]\n{'; '.join(items)}\n[/COMPOSITE SIGNALS]")
    if recent_speech_tags:
        recent = [f"{t.get('category', '')}: \"{t.get('phrase', '')}\"" for t in recent_speech_tags[-5:] if t.get("category")]
        if recent:
            sections.append(f"[RECENT SPEECH CUES]\n{'; '.join(recent)}\n[/RECENT SPEECH CUES]")
    ac_summary = get_recent_acoustic_context() or ""
    if acoustic_tags_list or ac_summary:
        voice_parts = []
        if ac_summary:
            voice_parts.append(ac_summary)
        if acoustic_tags_list:
            voice_parts.append("Tags: " + ", ".join(ACOUSTIC_LABELS.get(t, t) for t in acoustic_tags_list[:5]))
        sections.append(f"[VOICE / ACOUSTIC]\n{' '.join(voice_parts)}\n[/VOICE / ACOUSTIC]")
    transcript = get_recent_transcript()
    if transcript and transcript.strip():
        snippet = transcript[-600:] if len(transcript) > 600 else transcript
        sections.append(f"[RECENT TRANSCRIPT]\n{snippet.strip()}\n[/RECENT TRANSCRIPT]")
    user_ctx = engagement_context_store.get("pending_user_context")
    if user_ctx and isinstance(user_ctx, str) and user_ctx.strip():
        sections.append("[USER-PROVIDED CONTEXT]\n" + "\n".join(user_ctx.strip().split("\n")[:15]) + "\n[/USER-PROVIDED CONTEXT]")
    sections.append(
        "[CRITICAL]\nThis context was captured RIGHT NOW. Generate an insight SPECIFIC to these signals. Do NOT return a generic message.\n[/CRITICAL]"
    )
    return "\n\n".join(sections)


def start_detection(source_type, source_path=None):
    global engagement_detector
    from engagement_detector import EngagementStateDetector
    from helpers import load_weights

    if engagement_detector:
        engagement_detector.stop_detection()
    lightweight = config.LIGHTWEIGHT_MODE
    detection_method = (config.FACE_DETECTION_METHOD or "auto").lower()
    if lightweight:
        detection_method = "mediapipe"
    load_weights()
    det = EngagementStateDetector(detection_method=detection_method, lightweight_mode=lightweight)
    if not det.start_detection(source_type, source_path):
        return False, "Failed to start detection. Check video source."
    engagement_detector = det
    update_last_request()
    return True, {"detectionMethod": det.detection_method, "lightweightMode": det.lightweight_mode}


def stop_detection():
    global engagement_detector
    from helpers import clear_opportunity_state

    if engagement_detector:
        engagement_detector.stop_detection()
        engagement_detector = None
    clear_transcript()
    clear_pending_aural_alert()
    clear_recent_speech_tags()
    clear_opportunity_state()
    clear_acoustic_context()
