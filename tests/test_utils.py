"""
Utility module tests.

Tests context generator, engagement scorer, helpers, etc.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest


class TestContextGenerator(unittest.TestCase):
    """Test context generator."""

    def test_generate_context_no_face(self):
        """generate_context_no_face should return EngagementContext with expected fields."""
        from utils.context_generator import ContextGenerator, EngagementContext
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
        """format_for_ai should return a string."""
        from utils.context_generator import ContextGenerator, EngagementContext
        gen = ContextGenerator()
        ctx = gen.generate_context_no_face()
        formatted = gen.format_for_ai(ctx)
        self.assertIsInstance(formatted, str)
        self.assertIn("ENGAGEMENT", formatted)


class TestEngagementScorer(unittest.TestCase):
    """Test engagement scorer."""

    def test_calculate_score_returns_float(self):
        """EngagementScorer.calculate_score should return a float 0-100."""
        from utils.engagement_scorer import EngagementScorer, EngagementMetrics
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
    """Test helpers module."""

    def test_build_config_response_returns_dict(self):
        """build_config_response should return a dict with expected keys."""
        from utils.helpers import build_config_response
        resp = build_config_response()
        self.assertIsInstance(resp, dict)
        self.assertIn("openai", resp)
        self.assertIn("speech", resp)


class TestVideoSourceType(unittest.TestCase):
    """Test VideoSourceType enum."""

    def test_video_source_type_values(self):
        """VideoSourceType should have expected values."""
        from utils.video_source_handler import VideoSourceType
        self.assertEqual(VideoSourceType.WEBCAM.value, "webcam")
        self.assertEqual(VideoSourceType.FILE.value, "file")
        self.assertEqual(VideoSourceType.STREAM.value, "stream")
        self.assertEqual(VideoSourceType.PARTNER.value, "partner")


class TestFaceDetectionPreference(unittest.TestCase):
    """Test face detection preference."""

    def test_get_face_detection_method_returns_string(self):
        """get_face_detection_method should return a string."""
        from utils.face_detection_preference import get_face_detection_method
        m = get_face_detection_method()
        self.assertIsInstance(m, str)
        self.assertIn(m.lower(), ("mediapipe", "azure_face_api", "auto", "unified", ""))


class TestMetricSelector(unittest.TestCase):
    """Test dynamic metric selection (metric_selector)."""

    def test_get_active_metrics_returns_metric_config(self):
        """get_active_metrics should return MetricConfig with signifier_keys, speech_categories, acoustic_tags, composite_keys."""
        from utils.metric_selector import get_active_metrics, MetricConfig
        cfg = get_active_metrics()
        self.assertIsInstance(cfg, MetricConfig)
        self.assertIsInstance(cfg.signifier_keys, list)
        self.assertIsInstance(cfg.speech_categories, list)
        self.assertIsInstance(cfg.acoustic_tags, list)
        self.assertIsInstance(cfg.composite_keys, list)
        self.assertIn(cfg.tier, ("high", "medium", "low"))

    def test_override_high_returns_full_metrics(self):
        """get_active_metrics(override='high') should return high tier with full signifier set."""
        from utils.metric_selector import get_active_metrics
        cfg = get_active_metrics(override="high")
        self.assertGreater(len(cfg.signifier_keys), 30)

    def test_override_low_returns_subset(self):
        """get_active_metrics(override='low') should return fewer signifiers than high."""
        from utils.metric_selector import get_active_metrics
        cfg_high = get_active_metrics(override="high")
        cfg_low = get_active_metrics(override="low")
        self.assertLess(len(cfg_low.signifier_keys), len(cfg_high.signifier_keys))

    def test_metric_config_has_expected_keys(self):
        """MetricConfig should have non-empty lists for each metric type."""
        from utils.metric_selector import get_active_metrics_with_config
        cfg = get_active_metrics_with_config()
        self.assertGreater(len(cfg.signifier_keys), 0)
        self.assertGreater(len(cfg.speech_categories), 0)
        self.assertGreater(len(cfg.acoustic_tags), 0)
        self.assertGreater(len(cfg.composite_keys), 0)

    def test_high_tier_includes_new_speech_and_composites(self):
        """High tier should include new speech categories and composite metrics."""
        from utils.metric_selector import get_active_metrics
        cfg = get_active_metrics(override="high")
        self.assertIn("urgency", cfg.speech_categories)
        self.assertIn("skepticism", cfg.speech_categories)
        self.assertIn("enthusiasm", cfg.speech_categories)
        self.assertIn("psychological_safety_proxy", cfg.composite_keys)
        self.assertIn("enthusiasm_multimodal", cfg.composite_keys)


class TestEngagementComposites(unittest.TestCase):
    """Test engagement composites."""

    def test_compute_composite_metrics_returns_new_composites(self):
        """compute_composite_metrics should include new composites."""
        from utils.engagement_composites import compute_composite_metrics
        group_means = {"g1": 60, "g2": 50, "g3": 55, "g4": 60}
        out = compute_composite_metrics(group_means)
        self.assertIn("psychological_safety_proxy", out)
        self.assertIn("enthusiasm_multimodal", out)
        self.assertIn("rapport_depth", out)
