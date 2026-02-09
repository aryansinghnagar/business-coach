"""
Service layer tests.

Tests insight generator, acoustic context store, lazy initialization, etc.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock


class TestLazyFoundryService(unittest.TestCase):
    """Test lazy Azure AI Foundry service initialization."""

    def test_get_foundry_service_returns_singleton(self):
        """get_foundry_service should return same instance on subsequent calls."""
        import services.azure_foundry as mod
        mod._foundry_service = None
        from services.azure_foundry import get_foundry_service
        a = get_foundry_service()
        b = get_foundry_service()
        self.assertIs(a, b)


class TestLazySpeechService(unittest.TestCase):
    """Test lazy Speech service initialization."""

    def test_get_speech_service_returns_singleton(self):
        """get_speech_service should return same instance on subsequent calls."""
        import services.azure_speech as mod
        mod._speech_service = None
        from services.azure_speech import get_speech_service
        a = get_speech_service()
        b = get_speech_service()
        self.assertIs(a, b)


class TestInsightGenerator(unittest.TestCase):
    """Test insight generator functions."""

    def test_get_recent_transcript_returns_string(self):
        """get_recent_transcript should return a string."""
        from services.insight_generator import get_recent_transcript
        t = get_recent_transcript()
        self.assertIsInstance(t, str)

    def test_get_recent_speech_tags_returns_list(self):
        """get_recent_speech_tags should return a list."""
        from services.insight_generator import get_recent_speech_tags
        tags = get_recent_speech_tags()
        self.assertIsInstance(tags, list)

    def test_append_speech_tag(self):
        """append_speech_tag should add a tag."""
        from services.insight_generator import append_speech_tag, get_recent_speech_tags, clear_recent_speech_tags
        clear_recent_speech_tags()
        append_speech_tag("objection", "I'm not sure")
        tags = get_recent_speech_tags()
        self.assertGreater(len(tags), 0)
        self.assertEqual(tags[-1]["category"], "objection")
        self.assertEqual(tags[-1]["phrase"], "I'm not sure")
        clear_recent_speech_tags()

    @patch("services.insight_generator.get_foundry_service")
    def test_generate_insight_for_spike_returns_string(self, mock_get):
        """generate_insight_for_spike should return a string (or stock on failure)."""
        mock_get.return_value.chat_completion.return_value = "Mocked insight"
        from services.insight_generator import generate_insight_for_spike
        out = generate_insight_for_spike("g1", metrics_summary=None)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    @patch("services.insight_generator.get_foundry_service")
    def test_generate_insight_for_aural_returns_string(self, mock_get):
        """generate_insight_for_aural_trigger should return a string."""
        mock_get.return_value.chat_completion.return_value = "Mocked aural insight"
        from services.insight_generator import generate_insight_for_aural_trigger
        out = generate_insight_for_aural_trigger("objection", "I disagree")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


class TestEngagementRequestTracker(unittest.TestCase):
    """Test engagement request tracker."""

    def test_update_and_is_idle(self):
        """update_last_request should affect is_idle."""
        from services.engagement_request_tracker import update_last_request, is_idle
        update_last_request()
        # Right after update, should not be idle
        self.assertFalse(is_idle(60.0))


class TestAcousticContextStore(unittest.TestCase):
    """Test acoustic context store."""

    def test_get_recent_acoustic_context_returns_string(self):
        """get_recent_acoustic_context should return a string."""
        from services.acoustic_context_store import get_recent_acoustic_context
        ctx = get_recent_acoustic_context()
        self.assertIsInstance(ctx, str)

    def test_get_recent_acoustic_tags_returns_list(self):
        """get_recent_acoustic_tags should return a list."""
        from services.acoustic_context_store import get_recent_acoustic_tags
        tags = get_recent_acoustic_tags()
        self.assertIsInstance(tags, list)
