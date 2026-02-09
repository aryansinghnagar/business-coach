"""
API endpoint tests.

Uses Flask test client. Does not require a running server.
External services (OpenAI, Azure Speech) may be mocked or skipped when unavailable.
"""

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock


def get_app_client():
    """Create Flask app and test client. Lazy to avoid import-time side effects."""
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


class TestStaticRoutes(unittest.TestCase):
    """Test static file and page routes."""

    def setUp(self):
        self.client = get_app_client()

    def test_index_returns_200_or_404(self):
        """GET / should return 200 with HTML or 404 if index.html missing."""
        r = self.client.get("/")
        self.assertIn(r.status_code, (200, 404))
        if r.status_code == 200:
            self.assertIn("text/html", r.content_type)

    def test_dashboard_returns_200_or_404(self):
        """GET /dashboard should return 200 or 404."""
        r = self.client.get("/dashboard")
        self.assertIn(r.status_code, (200, 404))
        if r.status_code == 200:
            self.assertIn("text/html", r.content_type)

    def test_favicon_returns_204(self):
        """GET /favicon.ico should return 204."""
        r = self.client.get("/favicon.ico")
        self.assertEqual(r.status_code, 204)


class TestConfigEndpoints(unittest.TestCase):
    """Test config-related endpoints."""

    def setUp(self):
        self.client = get_app_client()

    def test_config_all_returns_json(self):
        """GET /config/all should return JSON."""
        r = self.client.get("/config/all")
        self.assertEqual(r.status_code, 200)
        self.assertIn("application/json", r.content_type)
        data = r.get_json()
        self.assertIsInstance(data, dict)

    def test_config_face_detection_get(self):
        """GET /config/face-detection should return JSON."""
        r = self.client.get("/config/face-detection")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, dict)

    def test_config_system_prompt_returns_string(self):
        """GET /config/system-prompt should return JSON with prompt."""
        r = self.client.get("/config/system-prompt")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue("prompt" in data or "systemPrompt" in data, msg="Expected 'prompt' or 'systemPrompt' key")
        prompt = data.get("prompt") or data.get("systemPrompt")
        self.assertIsInstance(prompt, str)


class TestEngagementEndpointsWithoutDetector(unittest.TestCase):
    """Test engagement endpoints when detector is not started (expect 404 or error)."""

    def setUp(self):
        self.client = get_app_client()

    def test_engagement_state_without_start_returns_404(self):
        """GET /engagement/state without starting detection should return 404."""
        r = self.client.get("/engagement/state")
        self.assertEqual(r.status_code, 404)
        data = r.get_json()
        self.assertIn("error", data)

    def test_engagement_stop_without_start_returns_200(self):
        """POST /engagement/stop without detector should not crash."""
        r = self.client.post("/engagement/stop", json={})
        self.assertIn(r.status_code, (200, 404))

    def test_engagement_debug_without_start_returns_404(self):
        """GET /engagement/debug without detector should return 404."""
        r = self.client.get("/engagement/debug")
        self.assertEqual(r.status_code, 404)


class TestContextAndResponseEndpoints(unittest.TestCase):
    """Test context-and-response and set-additional-context."""

    def setUp(self):
        self.client = get_app_client()

    def test_context_and_response_returns_json(self):
        """GET /api/engagement/context-and-response should return JSON."""
        r = self.client.get("/api/engagement/context-and-response")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn("contextSent", data)
        self.assertIn("response", data)

    def test_set_additional_context_accepts_post(self):
        """POST /api/engagement/set-additional-context should accept JSON."""
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
        """POST without JSON should return 400."""
        r = self.client.post(
            "/api/engagement/set-additional-context",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(r.status_code, 400)


class TestChatEndpoints(unittest.TestCase):
    """Test chat endpoints. OpenAI is mocked to avoid network calls."""

    def setUp(self):
        self.client = get_app_client()

    def test_chat_requires_json(self):
        """POST /chat without JSON should return 400."""
        r = self.client.post("/chat", data="text")
        self.assertEqual(r.status_code, 400)

    def test_chat_requires_message(self):
        """POST /chat with empty message should return 400."""
        r = self.client.post("/chat", json={}, content_type="application/json")
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/chat", json={"message": ""}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    @patch("routes.get_foundry_service")
    def test_chat_success_with_mock(self, mock_get_service):
        """POST /chat with valid message and mocked OpenAI returns 200."""
        mock_svc = MagicMock()
        mock_svc.chat_completion.return_value = "Mocked response"
        mock_get_service.return_value = mock_svc
        r = self.client.post(
            "/chat",
            json={"message": "Hello"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Mocked response")


class TestContextPush(unittest.TestCase):
    """Test context-push endpoint."""

    def setUp(self):
        self.client = get_app_client()

    @patch("routes.get_foundry_service")
    def test_context_push_returns_context_and_response(self, mock_get_service):
        """POST /api/context-push should return context and response."""
        mock_svc = MagicMock()
        mock_svc.chat_completion.return_value = "Mocked AI response"
        mock_get_service.return_value = mock_svc
        r = self.client.post("/api/context-push", json={}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("context", data)
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Mocked AI response")


class TestInsightWeights(unittest.TestCase):
    """Test insight weights endpoint."""

    def setUp(self):
        self.client = get_app_client()

    def test_insight_weights_get(self):
        """GET /weights/insight should return JSON."""
        r = self.client.get("/weights/insight")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, dict)
