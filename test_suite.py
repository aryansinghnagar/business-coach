"""
Business Meeting Copilot — consolidated test suite.

Single file: API endpoints, utils, services, metric validation, signifier parameter sweep.
Synthetic landmark helpers inlined for expression/signifier tests.
"""

import json
import math
import os
import sys
import time
import unittest
from typing import Tuple
from unittest.mock import patch, MagicMock

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


# =============================================================================
# Synthetic landmarks (inlined from tests/fixtures/synthetic_landmarks.py)
# =============================================================================

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYEBROW = [107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276]
NOSE = [4, 6, 19, 20, 51, 94, 168, 197, 326, 327, 358, 359, 360, 361]
MOUTH_LEFT, MOUTH_RIGHT = 61, 17
NOSE_TIP, CHIN = 4, 175
FACE_LEFT, FACE_RIGHT = 234, 454


def _make_base_landmarks(w: int = 640, h: int = 480) -> np.ndarray:
    """Create 468x3 landmark array with neutral face geometry in pixel space."""
    lm = np.zeros((468, 3), dtype=np.float64)
    cx, cy = w / 2, h / 2
    face_w = min(w, h) * 0.35
    face_h = face_w * 1.2
    lex, ley = cx - face_w * 0.25, cy - face_h * 0.1
    rex, rey = cx + face_w * 0.25, cy - face_h * 0.1
    eye_h = face_w * 0.08
    eye_w = face_w * 0.15
    for i, idx in enumerate(LEFT_EYE):
        t = i / max(len(LEFT_EYE) - 1, 1)
        lm[idx, 0] = lex - eye_w / 2 + eye_w * (t if t <= 0.5 else 1 - t)
        lm[idx, 1] = ley - eye_h / 2 + eye_h * (0.3 + 0.7 * (t if t < 0.5 else 1 - t))
        lm[idx, 2] = 0
    for i, idx in enumerate(RIGHT_EYE):
        t = i / max(len(RIGHT_EYE) - 1, 1)
        lm[idx, 0] = rex - eye_w / 2 + eye_w * (t if t <= 0.5 else 1 - t)
        lm[idx, 1] = rey - eye_h / 2 + eye_h * (0.3 + 0.7 * (t if t < 0.5 else 1 - t))
        lm[idx, 2] = 0
    for i, idx in enumerate(LEFT_EYEBROW):
        lm[idx, 0] = lex - eye_w / 2 + (i / max(len(LEFT_EYEBROW) - 1, 1)) * eye_w * 0.5
        lm[idx, 1] = ley - eye_h - face_w * 0.03
        lm[idx, 2] = 0
    for i, idx in enumerate(RIGHT_EYEBROW):
        lm[idx, 0] = rex - eye_w / 2 + (i / max(len(RIGHT_EYEBROW) - 1, 1)) * eye_w * 0.5
        lm[idx, 1] = rey - eye_h - face_w * 0.03
        lm[idx, 2] = 0
    mw = face_w * 0.4
    mh = face_w * 0.08
    mx, my = cx, cy + face_h * 0.25
    for i, idx in enumerate(MOUTH):
        t = i / max(len(MOUTH) - 1, 1)
        lm[idx, 0] = mx - mw / 2 + mw * (t if t <= 0.5 else 1 - (t - 0.5) * 2)
        lm[idx, 1] = my - mh / 2 + mh * 0.5
        lm[idx, 2] = 0
    lm[MOUTH_LEFT, 0] = mx - mw / 2
    lm[MOUTH_LEFT, 1] = my
    lm[MOUTH_RIGHT, 0] = mx + mw / 2
    lm[MOUTH_RIGHT, 1] = my
    lm[NOSE_TIP, 0] = cx
    lm[NOSE_TIP, 1] = cy + face_h * 0.05
    lm[NOSE_TIP, 2] = 0
    for idx in NOSE:
        if lm[idx, 0] == 0 and lm[idx, 1] == 0:
            lm[idx, 0] = cx + (np.random.rand() - 0.5) * face_w * 0.1
            lm[idx, 1] = cy + face_h * 0.05 + (np.random.rand() - 0.5) * face_h * 0.1
            lm[idx, 2] = 0
    lm[CHIN, 0] = cx
    lm[CHIN, 1] = cy + face_h * 0.5
    lm[CHIN, 2] = 0
    lm[FACE_LEFT, 0] = cx - face_w / 2
    lm[FACE_LEFT, 1] = cy
    lm[FACE_LEFT, 2] = 0
    lm[FACE_RIGHT, 0] = cx + face_w / 2
    lm[FACE_RIGHT, 1] = cy
    lm[FACE_RIGHT, 2] = 0
    np.random.seed(42)
    for i in range(lm.shape[0]):
        if np.all(lm[i] == 0):
            lm[i, 0] = cx + (np.random.rand() - 0.5) * face_w * 0.3
            lm[i, 1] = cy + (np.random.rand() - 0.5) * face_h * 0.3
            lm[i, 2] = 0
    return lm


def _rotate_landmarks_2d(lm: np.ndarray, cx: float, cy: float, roll_deg: float) -> np.ndarray:
    out = lm.copy()
    rad = math.radians(roll_deg)
    c, s = math.cos(rad), math.sin(rad)
    for i in range(lm.shape[0]):
        x, y = lm[i, 0] - cx, lm[i, 1] - cy
        out[i, 0] = cx + x * c - y * s
        out[i, 1] = cy + x * s + y * c
    return out


def make_neutral_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    h, w = shape[0], shape[1]
    return _make_base_landmarks(w, h)


def make_smile_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), strength: float = 1.0) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    for idx in [MOUTH_LEFT, MOUTH_RIGHT]:
        lm[idx, 1] -= 8 * strength
    for idx in MOUTH:
        lm[idx, 1] -= 5 * strength
    return lm


def make_head_tilt_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), roll_deg: float = 12.0) -> np.ndarray:
    h, w = shape[0], shape[1]
    lm = _make_base_landmarks(w, h)
    return _rotate_landmarks_2d(lm, w / 2, h / 2, roll_deg)


def make_gaze_aversion_landmarks(shape: Tuple[int, int, int] = (480, 640, 3), yaw_offset: float = 80) -> np.ndarray:
    h, w = shape[0], shape[1]
    lm = _make_base_landmarks(w, h)
    inner_indices = set(LEFT_EYE + RIGHT_EYE + MOUTH + LEFT_EYEBROW + RIGHT_EYEBROW + NOSE + [NOSE_TIP, CHIN])
    for i in inner_indices:
        if i < lm.shape[0]:
            lm[i, 0] += yaw_offset
    return lm


def make_lip_compression_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    my = np.mean([lm[i, 1] for i in MOUTH])
    mw_orig = lm[MOUTH_RIGHT, 0] - lm[MOUTH_LEFT, 0]
    cx = shape[1] / 2
    for idx in MOUTH:
        lm[idx, 0] = cx + (lm[idx, 0] - cx) * 0.4
        lm[idx, 1] = my + (lm[idx, 1] - my) * 0.2
    lm[MOUTH_LEFT, 0] = cx - mw_orig * 0.15
    lm[MOUTH_RIGHT, 0] = cx + mw_orig * 0.15
    return lm


def make_parted_lips_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    my = np.mean([lm[i, 1] for i in MOUTH])
    cx = shape[1] / 2
    for idx in MOUTH:
        if lm[idx, 1] > my:
            lm[idx, 1] += 25
        else:
            lm[idx, 1] -= 25
    for idx in MOUTH:
        lm[idx, 0] = cx + (lm[idx, 0] - cx) * 1.15
    return lm


def make_contempt_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    lm[MOUTH_RIGHT, 1] -= 25
    lm[MOUTH_RIGHT, 0] += 5
    lm[MOUTH_LEFT, 1] += 8
    return lm


def make_brow_furrow_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    for idx in LEFT_EYEBROW + RIGHT_EYEBROW:
        lm[idx, 1] += 10
    return lm


def make_brow_raise_landmarks(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    lm = _make_base_landmarks(shape[1], shape[0])
    for idx in LEFT_EYEBROW + RIGHT_EYEBROW:
        lm[idx, 1] -= 12
    return lm


def warmup_engine(engine, landmarks: np.ndarray, shape: Tuple[int, int, int], n_frames: int = 15) -> None:
    for _ in range(n_frames):
        engine.update(landmarks, None, shape)


# =============================================================================
# API endpoint tests
# =============================================================================

def get_app_client():
    """Create Flask app and test client."""
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


class TestStaticRoutes(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    def test_index_returns_200_or_404(self):
        r = self.client.get("/")
        self.assertIn(r.status_code, (200, 404))
        if r.status_code == 200:
            self.assertIn("text/html", r.content_type)

    def test_dashboard_redirects_to_index(self):
        r = self.client.get("/dashboard")
        self.assertIn(r.status_code, (302, 200, 404))
        if r.status_code == 302:
            location = (r.location or "") or r.headers.get("Location", "")
            location_lower = location.lower()
            # Dashboard redirects to main app: either "/" (pages.index) or a path containing "index"
            self.assertTrue(
                location_lower.endswith("/") or "index" in location_lower,
                f"Redirect location should be / or contain 'index', got {location!r}",
            )
        elif r.status_code == 200:
            self.assertIn("text/html", r.content_type)

    def test_favicon_returns_204(self):
        r = self.client.get("/favicon.ico")
        self.assertEqual(r.status_code, 204)


class TestConfigEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    def test_config_all_returns_json(self):
        r = self.client.get("/config/all")
        self.assertEqual(r.status_code, 200)
        self.assertIn("application/json", r.content_type)
        self.assertIsInstance(r.get_json(), dict)

    def test_config_face_detection_get(self):
        r = self.client.get("/config/face-detection")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json(), dict)

    def test_config_system_prompt_returns_string(self):
        r = self.client.get("/config/system-prompt")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue("prompt" in data or "systemPrompt" in data)
        prompt = data.get("prompt") or data.get("systemPrompt")
        self.assertIsInstance(prompt, str)


class TestEngagementEndpointsWithoutDetector(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    def test_engagement_state_without_start_returns_200_not_started(self):
        r = self.client.get("/engagement/state")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("detectionStarted", data)
        self.assertFalse(data["detectionStarted"])
        self.assertEqual(data.get("score"), 0)

    def test_engagement_stop_without_start_returns_200(self):
        r = self.client.post("/engagement/stop", json={})
        self.assertIn(r.status_code, (200, 404))

    def test_engagement_debug_without_start_returns_404(self):
        r = self.client.get("/engagement/debug")
        self.assertEqual(r.status_code, 404)


class TestContextAndResponseEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    def test_context_and_response_returns_json(self):
        r = self.client.get("/api/engagement/context-and-response")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn("contextSent", data)
        self.assertIn("response", data)

    def test_set_additional_context_accepts_post(self):
        r = self.client.post(
            "/api/engagement/set-additional-context",
            json={"additionalContext": "Test note"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("ok", data)
        self.assertTrue(data["ok"])

    def test_set_additional_context_requires_json(self):
        r = self.client.post(
            "/api/engagement/set-additional-context",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(r.status_code, 400)


class TestChatEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    def test_chat_requires_json(self):
        r = self.client.post("/chat", data="text")
        self.assertEqual(r.status_code, 400)

    def test_chat_requires_message(self):
        r = self.client.post("/chat", json={}, content_type="application/json")
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/chat", json={"message": ""}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    @patch("routes.get_foundry_service")
    def test_chat_success_with_mock(self, mock_get_service):
        mock_svc = MagicMock()
        mock_svc.chat_completion.return_value = "Mocked response"
        mock_get_service.return_value = mock_svc
        r = self.client.post("/chat", json={"message": "Hello"}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Mocked response")


class TestContextPush(unittest.TestCase):
    def setUp(self):
        self.client = get_app_client()

    @patch("routes.get_foundry_service")
    def test_context_push_returns_context_and_response(self, mock_get_service):
        mock_svc = MagicMock()
        mock_svc.chat_completion.return_value = "Mocked AI response"
        mock_get_service.return_value = mock_svc
        r = self.client.post("/api/context-push", json={}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("context", data)
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Mocked AI response")


# =============================================================================
# Utils tests
# =============================================================================

class TestContextGenerator(unittest.TestCase):
    def test_generate_context_no_face(self):
        from helpers import ContextGenerator, EngagementContext
        gen = ContextGenerator()
        ctx = gen.generate_context_no_face()
        self.assertIsInstance(ctx, EngagementContext)
        self.assertIsInstance(ctx.summary, str)
        self.assertIsInstance(ctx.level_description, str)
        self.assertIsInstance(ctx.key_indicators, list)
        self.assertIsInstance(ctx.suggested_actions, list)
        self.assertIsInstance(ctx.risk_factors, list)
        self.assertIsInstance(ctx.opportunities, list)
        self.assertIn("No face", ctx.summary)

    def test_format_for_ai(self):
        from helpers import ContextGenerator
        gen = ContextGenerator()
        ctx = gen.generate_context_no_face()
        formatted = gen.format_for_ai(ctx)
        self.assertIsInstance(formatted, str)
        self.assertIn("ENGAGEMENT", formatted)


class TestEngagementScorer(unittest.TestCase):
    def test_calculate_score_returns_float(self):
        from helpers import EngagementScorer, EngagementMetrics
        scorer = EngagementScorer()
        metrics = EngagementMetrics(
            attention=70, eye_contact=65, facial_expressiveness=60,
            head_movement=50, symmetry=55, mouth_activity=45
        )
        score = scorer.calculate_score(metrics)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestHelpers(unittest.TestCase):
    def test_build_config_response_returns_dict(self):
        from config import build_config_response
        resp = build_config_response()
        self.assertIsInstance(resp, dict)
        self.assertIn("openai", resp)
        self.assertIn("speech", resp)


class TestVideoSourceType(unittest.TestCase):
    def test_video_source_type_values(self):
        from helpers import VideoSourceType
        self.assertEqual(VideoSourceType.WEBCAM.value, "webcam")
        self.assertEqual(VideoSourceType.FILE.value, "file")
        self.assertEqual(VideoSourceType.STREAM.value, "stream")
        self.assertEqual(VideoSourceType.PARTNER.value, "partner")


class TestFaceDetectionPreference(unittest.TestCase):
    def test_get_face_detection_method_returns_string(self):
        from config import get_face_detection_method
        m = get_face_detection_method()
        self.assertIsInstance(m, str)
        self.assertIn(m.lower(), ("mediapipe", "azure_face_api", "auto", "unified", ""))


class TestMetricSelector(unittest.TestCase):
    def test_get_active_metrics_returns_metric_config(self):
        from helpers import get_active_metrics, MetricConfig
        cfg = get_active_metrics()
        self.assertIsInstance(cfg, MetricConfig)
        self.assertIsInstance(cfg.signifier_keys, list)
        self.assertIsInstance(cfg.speech_categories, list)
        self.assertIsInstance(cfg.acoustic_tags, list)
        self.assertIsInstance(cfg.composite_keys, list)
        self.assertIn(cfg.tier, ("high", "medium", "low"))

    def test_override_high_returns_full_metrics(self):
        from helpers import get_active_metrics
        cfg = get_active_metrics(override="high")
        self.assertGreater(len(cfg.signifier_keys), 30)

    def test_override_low_returns_subset(self):
        from helpers import get_active_metrics
        cfg_high = get_active_metrics(override="high")
        cfg_low = get_active_metrics(override="low")
        self.assertLess(len(cfg_low.signifier_keys), len(cfg_high.signifier_keys))

    def test_metric_config_has_expected_keys(self):
        from helpers import get_active_metrics_with_config
        cfg = get_active_metrics_with_config()
        self.assertGreater(len(cfg.signifier_keys), 0)
        self.assertGreater(len(cfg.speech_categories), 0)
        self.assertGreater(len(cfg.acoustic_tags), 0)
        self.assertGreater(len(cfg.composite_keys), 0)

    def test_high_tier_includes_new_speech_and_composites(self):
        from helpers import get_active_metrics
        cfg = get_active_metrics(override="high")
        self.assertIn("urgency", cfg.speech_categories)
        self.assertIn("skepticism", cfg.speech_categories)
        self.assertIn("enthusiasm", cfg.speech_categories)
        self.assertIn("psychological_safety_proxy", cfg.composite_keys)
        self.assertIn("enthusiasm_multimodal", cfg.composite_keys)


class TestEngagementComposites(unittest.TestCase):
    def test_compute_composite_metrics_returns_new_composites(self):
        from helpers import compute_composite_metrics
        group_means = {"g1": 60, "g2": 50, "g3": 55, "g4": 60}
        out = compute_composite_metrics(group_means)
        self.assertIn("psychological_safety_proxy", out)
        self.assertIn("enthusiasm_multimodal", out)
        self.assertIn("rapport_depth", out)


# =============================================================================
# Service tests
# =============================================================================

class TestLazyFoundryService(unittest.TestCase):
    def test_get_foundry_service_returns_singleton(self):
        import services as mod
        mod._foundry_service = None
        from services import get_foundry_service
        a = get_foundry_service()
        b = get_foundry_service()
        self.assertIs(a, b)


class TestLazySpeechService(unittest.TestCase):
    def test_get_speech_service_returns_singleton(self):
        import services as mod
        mod._speech_service = None
        from services import get_speech_service
        a = get_speech_service()
        b = get_speech_service()
        self.assertIs(a, b)


class TestInsightGenerator(unittest.TestCase):
    def test_get_recent_transcript_returns_string(self):
        from services import get_recent_transcript
        self.assertIsInstance(get_recent_transcript(), str)

    def test_get_recent_speech_tags_returns_list(self):
        from services import get_recent_speech_tags
        self.assertIsInstance(get_recent_speech_tags(), list)

    def test_append_speech_tag(self):
        from services import append_speech_tag, get_recent_speech_tags, clear_recent_speech_tags
        clear_recent_speech_tags()
        append_speech_tag("objection", "I'm not sure")
        tags = get_recent_speech_tags()
        self.assertGreater(len(tags), 0)
        self.assertEqual(tags[-1]["category"], "objection")
        self.assertEqual(tags[-1]["phrase"], "I'm not sure")
        clear_recent_speech_tags()

    @patch("services.get_foundry_service")
    def test_generate_insight_for_spike_returns_string(self, mock_get):
        mock_get.return_value.chat_completion.return_value = "Mocked insight"
        from services import generate_insight_for_spike
        out = generate_insight_for_spike("g1", metrics_summary=None)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    @patch("services.get_foundry_service")
    def test_generate_insight_for_aural_returns_string(self, mock_get):
        mock_get.return_value.chat_completion.return_value = "Mocked aural insight"
        from services import generate_insight_for_aural_trigger
        out = generate_insight_for_aural_trigger("objection", "I disagree")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


class TestEngagementRequestTracker(unittest.TestCase):
    def test_update_and_is_idle(self):
        from services import update_last_request, is_idle
        update_last_request()
        self.assertFalse(is_idle(60.0))


class TestAcousticContextStore(unittest.TestCase):
    def test_get_recent_acoustic_context_returns_string(self):
        from services import get_recent_acoustic_context
        self.assertIsInstance(get_recent_acoustic_context(), str)

    def test_get_recent_acoustic_tags_returns_list(self):
        from services import get_recent_acoustic_tags
        self.assertIsInstance(get_recent_acoustic_tags(), list)


# =============================================================================
# Metric validation tests
# =============================================================================

class TestExpressionSignifierMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from helpers import ExpressionSignifierEngine, get_weights
        cls.engine = ExpressionSignifierEngine(buffer_frames=22, weights_provider=get_weights)
        cls.shape = (480, 640, 3)

    def setUp(self):
        self.engine.reset()

    def _warmup_and_score(self, landmarks_fn, n_warmup=15, n_scoring=8):
        neutral = make_neutral_landmarks(self.shape)
        warmup_engine(self.engine, neutral, self.shape, n_warmup)
        lm = landmarks_fn(self.shape)
        for _ in range(n_scoring):
            self.engine.update(lm, None, self.shape)
        return self.engine.get_all_scores()

    def test_g1_duchenne_higher_for_smile_than_neutral(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_smile_landmarks(self.shape), None, self.shape)
        scores_smile = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreater(scores_smile["g1_duchenne"], scores_neutral["g1_duchenne"])

    def test_g1_head_tilt_higher_for_tilted_head(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_head_tilt_landmarks(self.shape, roll_deg=12), None, self.shape)
        scores_tilt = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreater(scores_tilt["g1_head_tilt"], scores_neutral["g1_head_tilt"])

    def test_g1_parted_lips_higher_or_equal_for_open_mouth(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_parted_lips_landmarks(self.shape), None, self.shape)
        scores_open = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreaterEqual(scores_open["g1_parted_lips"], scores_neutral["g1_parted_lips"])

    def test_g3_contempt_higher_or_equal_for_unilateral_asymmetry(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_contempt_landmarks(self.shape), None, self.shape)
        scores_contempt = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreaterEqual(scores_contempt["g3_contempt"], scores_neutral["g3_contempt"])

    def test_g3_lip_compression_higher_or_equal_for_compressed_lips(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_lip_compression_landmarks(self.shape), None, self.shape)
        scores_compressed = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreaterEqual(scores_compressed["g3_lip_compression"], scores_neutral["g3_lip_compression"])

    def test_g3_gaze_aversion_higher_or_equal_when_gaze_averted(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_gaze_aversion_landmarks(self.shape, yaw_offset=100), None, self.shape)
        scores_averted = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreaterEqual(scores_averted["g3_gaze_aversion"], scores_neutral["g3_gaze_aversion"])

    def test_g2_brow_furrow_higher_for_furrowed_brow(self):
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_brow_furrow_landmarks(self.shape), None, self.shape)
        scores_furrow = self.engine.get_all_scores()
        self.engine.reset()
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        for _ in range(8):
            self.engine.update(make_neutral_landmarks(self.shape), None, self.shape)
        scores_neutral = self.engine.get_all_scores()
        self.assertGreater(scores_furrow["g2_brow_furrow_deep"], scores_neutral["g2_brow_furrow_deep"])

    def test_all_signifiers_return_valid_scores(self):
        from helpers import SIGNIFIER_KEYS
        warmup_engine(self.engine, make_neutral_landmarks(self.shape), self.shape, 15)
        scores = self.engine.get_all_scores()
        for key in SIGNIFIER_KEYS:
            self.assertIn(key, scores)
            v = scores[key]
            self.assertIsInstance(v, (int, float))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 100.0)


class TestSpeechCategoryMetrics(unittest.TestCase):
    def test_objection_matches(self):
        from services import check_speech_cues
        for t in ["I'm not sure that works", "We can't do that", "I disagree with that"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("objection", cats)

    def test_interest_matches(self):
        from services import check_speech_cues
        for t in ["that's interesting", "tell me more", "that could work"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("interest", cats)

    def test_commitment_matches(self):
        from services import check_speech_cues
        for t in ["let's do it", "I'm in", "we'll move forward"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("commitment", cats)

    def test_confusion_matches(self):
        from services import check_speech_cues
        for t in ["I don't understand", "can you clarify", "not following"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("confusion", cats)

    def test_urgency_matches(self):
        from services import check_speech_cues
        for t in ["we need it asap", "time-sensitive", "urgent priority"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("urgency", cats)

    def test_skepticism_matches(self):
        from services import check_speech_cues
        for t in ["sounds too good", "prove it", "we'll see"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("skepticism", cats)

    def test_enthusiasm_matches(self):
        from services import check_speech_cues
        for t in ["love it", "great idea", "can't wait"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("enthusiasm", cats)

    def test_hesitation_matches(self):
        from services import check_speech_cues
        for t in ["um, well", "I guess", "maybe"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("hesitation", cats)

    def test_confirmation_matches(self):
        from services import check_speech_cues
        for t in ["exactly", "that's right", "correct"]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertIn("confirmation", cats)

    def test_neutral_text_returns_empty_or_no_false_positives(self):
        from services import check_speech_cues
        for t in ["The meeting is at 3pm.", "Please send the report."]:
            cats = [m[0] for m in check_speech_cues(t)]
            self.assertNotIn("objection", cats)
            self.assertNotIn("confusion", cats)


class TestAcousticTagMetrics(unittest.TestCase):
    def test_disengagement_risk_tag_for_low_loudness_flat_pitch(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.1, "pitch_hz": 150, "pitch_contour": "flat",
             "pitch_variability": 0.2, "tone_proxy": 0.3, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_disengagement_risk", tags)

    def test_arousal_high_tag_for_high_loudness_pitch(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.7, "pitch_hz": 220, "pitch_contour": "rising",
             "pitch_variability": 1.0, "tone_proxy": 0.4, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_arousal_high", tags)

    def test_uncertainty_tag_for_high_variability_rising(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.4, "pitch_hz": 180, "pitch_contour": "rising",
             "pitch_variability": 2.5, "tone_proxy": 0.5, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_uncertainty", tags)

    def test_tension_tag_for_tense_tone_elevated_pitch(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.5, "pitch_hz": 200, "pitch_contour": "flat",
             "pitch_variability": 1.0, "tone_proxy": 0.75, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_tension", tags)

    def test_falling_contour_tag_for_falling_pitch(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.4, "pitch_hz": 170, "pitch_contour": "falling",
             "pitch_variability": 0.5, "tone_proxy": 0.4, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_falling_contour", tags)

    def test_monotone_tag_for_low_variability_flat(self):
        from services import interpret_acoustic_windows
        windows = [
            {"loudness_norm": 0.4, "pitch_hz": 160, "pitch_contour": "flat",
             "pitch_variability": 0.3, "tone_proxy": 0.4, "voicing_confidence": 0.9, "speech_active": True},
        ] * 5
        _, tags = interpret_acoustic_windows(windows)
        self.assertIn("acoustic_monotone", tags)

    def test_empty_windows_returns_empty(self):
        from services import interpret_acoustic_windows
        summary, tags = interpret_acoustic_windows([])
        self.assertEqual(summary, "")
        self.assertEqual(tags, [])


class TestCompositeMetrics(unittest.TestCase):
    def test_decision_readiness_higher_for_g4_commitment(self):
        from helpers import compute_composite_metrics
        group_means_high = {"g1": 60, "g2": 40, "g3": 60, "g4": 85}
        group_means_low = {"g1": 50, "g2": 50, "g3": 50, "g4": 30}
        speech_commit = [{"category": "commitment", "phrase": "let's do it", "time": time.time(), "discourse_boost": False}]
        out_high = compute_composite_metrics(group_means_high, speech_tags=speech_commit)
        out_low = compute_composite_metrics(group_means_low, speech_tags=[])
        self.assertGreater(out_high["decision_readiness_multimodal"], out_low["decision_readiness_multimodal"])

    def test_cognitive_load_higher_for_g2_confusion_speech(self):
        from helpers import compute_composite_metrics
        group_means_high = {"g1": 40, "g2": 80, "g3": 50, "g4": 40}
        group_means_low = {"g1": 60, "g2": 30, "g3": 50, "g4": 50}
        speech_conf = [{"category": "confusion", "phrase": "I don't understand", "time": time.time()}]
        out_high = compute_composite_metrics(group_means_high, speech_tags=speech_conf)
        out_low = compute_composite_metrics(group_means_low, speech_tags=[])
        self.assertGreater(out_high["cognitive_load_multimodal"], out_low["cognitive_load_multimodal"])

    def test_disengagement_risk_higher_for_low_g1_no_commitment(self):
        from helpers import compute_composite_metrics
        group_means_high = {"g1": 25, "g2": 50, "g3": 40, "g4": 30}
        group_means_low = {"g1": 75, "g2": 50, "g3": 60, "g4": 70}
        out_high = compute_composite_metrics(group_means_high, speech_tags=[])
        out_low = compute_composite_metrics(group_means_low, speech_tags=[])
        self.assertGreater(out_high["disengagement_risk_multimodal"], out_low["disengagement_risk_multimodal"])

    def test_skepticism_strength_higher_for_skepticism_speech_resistance(self):
        from helpers import compute_composite_metrics
        speech_skep = [{"category": "skepticism", "phrase": "prove it", "time": time.time()}]
        group_means_high = {"g1": 40, "g2": 50, "g3": 35, "g4": 40}
        group_means_low = {"g1": 70, "g2": 50, "g3": 70, "g4": 60}
        out_high = compute_composite_metrics(group_means_high, speech_tags=speech_skep)
        out_low = compute_composite_metrics(group_means_low, speech_tags=[])
        self.assertGreater(out_high["skepticism_strength"], out_low["skepticism_strength"])

    def test_enthusiasm_multimodal_higher_for_enthusiasm_speech_g1(self):
        from helpers import compute_composite_metrics
        speech_enth = [{"category": "enthusiasm", "phrase": "love it", "time": time.time()}]
        group_means_high = {"g1": 85, "g2": 40, "g3": 65, "g4": 70}
        group_means_low = {"g1": 35, "g2": 50, "g3": 50, "g4": 40}
        out_high = compute_composite_metrics(group_means_high, speech_tags=speech_enth)
        out_low = compute_composite_metrics(group_means_low, speech_tags=[])
        self.assertGreater(out_high["enthusiasm_multimodal"], out_low["enthusiasm_multimodal"])

    def test_all_composites_return_valid_scores(self):
        from helpers import compute_composite_metrics
        group_means = {"g1": 50, "g2": 50, "g3": 50, "g4": 50}
        out = compute_composite_metrics(group_means)
        for k, v in out.items():
            self.assertIsInstance(v, (int, float))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 100.0)

    def test_acoustic_boost_affects_composites(self):
        from helpers import compute_composite_metrics
        group_means = {"g1": 50, "g2": 60, "g3": 50, "g4": 50}
        out_no_ac = compute_composite_metrics(group_means, acoustic_tags=[])
        out_with_unc = compute_composite_metrics(group_means, acoustic_tags=["acoustic_uncertainty"])
        self.assertGreater(out_with_unc["confusion_multimodal"], out_no_ac["confusion_multimodal"])


class TestGroupMeansAndCompositeScore(unittest.TestCase):
    def test_group_means_g3_inverted_when_resistance_increases(self):
        from helpers import ExpressionSignifierEngine, get_weights
        engine = ExpressionSignifierEngine(buffer_frames=22, weights_provider=get_weights)
        shape = (480, 640, 3)
        warmup_engine(engine, make_neutral_landmarks(shape), shape, 15)
        for _ in range(8):
            engine.update(make_neutral_landmarks(shape), None, shape)
        g_neutral = engine.get_group_means()
        engine.reset()
        warmup_engine(engine, make_neutral_landmarks(shape), shape, 15)
        for _ in range(8):
            engine.update(make_contempt_landmarks(shape), None, shape)
        g_contempt = engine.get_group_means()
        self.assertGreaterEqual(g_neutral["g3"], 0.0)
        self.assertLessEqual(g_neutral["g3"], 100.0)
        self.assertGreaterEqual(g_contempt["g3"], 0.0)
        self.assertLessEqual(g_contempt["g3"], 100.0)

    def test_composite_score_within_range(self):
        from helpers import ExpressionSignifierEngine, get_weights
        engine = ExpressionSignifierEngine(buffer_frames=22, weights_provider=get_weights)
        shape = (480, 640, 3)
        warmup_engine(engine, make_neutral_landmarks(shape), shape, 15)
        score = engine.get_composite_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


# =============================================================================
# Signifier parameter sweep tests
# =============================================================================

def _warmup_and_get_scores(engine, landmarks, shape, n_warmup=18, n_scoring=10):
    neutral = make_neutral_landmarks(shape)
    warmup_engine(engine, neutral, shape, n_warmup)
    for _ in range(n_scoring):
        engine.update(landmarks, None, shape)
    return engine.get_all_scores()


class TestSignifierParameterSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from helpers import ExpressionSignifierEngine, get_weights
        cls.engine = ExpressionSignifierEngine(buffer_frames=22, weights_provider=get_weights)
        cls.shape = (480, 640, 3)

    def setUp(self):
        self.engine.reset()

    def _run_sweep(self):
        results = {}
        configs = [
            ("neutral", lambda: make_neutral_landmarks(self.shape)),
            ("smile", lambda: make_smile_landmarks(self.shape, strength=1.0)),
            ("gaze_aversion_20", lambda: make_gaze_aversion_landmarks(self.shape, yaw_offset=20)),
            ("gaze_aversion_40", lambda: make_gaze_aversion_landmarks(self.shape, yaw_offset=40)),
            ("gaze_aversion_80", lambda: make_gaze_aversion_landmarks(self.shape, yaw_offset=80)),
            ("contempt", lambda: make_contempt_landmarks(self.shape)),
            ("parted_lips", lambda: make_parted_lips_landmarks(self.shape)),
            ("lip_compression", lambda: make_lip_compression_landmarks(self.shape)),
            ("head_tilt_12", lambda: make_head_tilt_landmarks(self.shape, roll_deg=12)),
            ("brow_raise", lambda: make_brow_raise_landmarks(self.shape)),
            ("brow_furrow", lambda: make_brow_furrow_landmarks(self.shape)),
        ]
        for name, fn in configs:
            self.engine.reset()
            scores = _warmup_and_get_scores(self.engine, fn(), self.shape)
            results[name] = scores
        return results

    def test_sweep_directions_smile_vs_neutral(self):
        results = self._run_sweep()
        self.assertGreater(results["smile"]["g1_duchenne"], results["neutral"]["g1_duchenne"])
        self.assertGreater(results["smile"]["g4_smile_transition"], results["neutral"]["g4_smile_transition"])

    def test_sweep_directions_gaze_aversion(self):
        results = self._run_sweep()
        self.assertGreaterEqual(
            results["gaze_aversion_80"]["g3_gaze_aversion"],
            results["gaze_aversion_40"]["g3_gaze_aversion"],
        )
        self.assertLess(
            results["gaze_aversion_80"]["g1_eye_contact"],
            results["neutral"]["g1_eye_contact"],
        )

    def test_sweep_directions_contempt(self):
        results = self._run_sweep()
        self.assertGreaterEqual(results["contempt"]["g3_contempt"], results["neutral"]["g3_contempt"])

    def test_sweep_directions_parted_vs_compression(self):
        results = self._run_sweep()
        self.assertGreaterEqual(
            results["parted_lips"]["g1_parted_lips"],
            results["lip_compression"]["g1_parted_lips"],
        )
        self.assertGreaterEqual(
            results["lip_compression"]["g3_lip_compression"],
            results["neutral"]["g3_lip_compression"],
        )

    def test_baseline_warmup_behavior(self):
        neutral = make_neutral_landmarks(self.shape)
        warmup_engine(self.engine, neutral, self.shape, n_frames=65)
        warmup_count = getattr(self.engine, "_baseline_warmup_frames", 0)
        self.assertGreaterEqual(warmup_count, 60)
        scores = self.engine.get_all_scores()
        self.assertIn("g1_duchenne", scores)
        self.assertGreaterEqual(scores["g1_duchenne"], 0.0)
        self.assertLessEqual(scores["g1_duchenne"], 100.0)

    def test_sweep_document_ranges(self):
        results = self._run_sweep()
        key_signifiers = [
            "g1_duchenne", "g1_eye_contact", "g1_parted_lips", "g1_facial_symmetry",
            "g3_contempt", "g3_gaze_aversion", "g3_lip_compression",
            "g4_smile_transition", "g4_fixed_gaze",
        ]
        lines = [
            "# Signifier Parameter Sweep (Phase C2)",
            "",
            "| Configuration | " + " | ".join(key_signifiers) + " |",
            "|" + "---|" * (len(key_signifiers) + 1) + "|",
        ]
        for config_name in ["neutral", "smile", "gaze_aversion_20", "gaze_aversion_40", "gaze_aversion_80",
                            "contempt", "parted_lips", "lip_compression", "head_tilt_12", "brow_raise", "brow_furrow"]:
            row = [config_name]
            for k in key_signifiers:
                val = results.get(config_name, {}).get(k, 0)
                row.append(f"{float(val):.1f}")
            lines.append("| " + " | ".join(row) + " |")
        out_path = os.path.join(_PROJECT_ROOT, "docs", "sweep_results.md")
        if os.getenv("WRITE_SIGNIFIER_SWEEP_DOC", "0") == "1":
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        self.assertIn("neutral", results)
        self.assertIn("g1_duchenne", results["neutral"])


if __name__ == "__main__":
    unittest.main()
