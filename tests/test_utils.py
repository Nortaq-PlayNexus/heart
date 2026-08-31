"""Tests for heart.utils module."""

from heart.utils.helpers import clamp, lerp, normalize_valence, normalize_arousal, emotional_distance, intensity_label


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5) == 0.5

    def test_below_min(self):
        assert clamp(-0.5) == 0.0

    def test_above_max(self):
        assert clamp(1.5) == 1.0

    def test_custom_range(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10


class TestLerp:
    def test_midpoint(self):
        assert abs(lerp(0.0, 10.0, 0.5) - 5.0) < 0.01

    def test_start(self):
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_end(self):
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_clamped_t(self):
        assert lerp(0.0, 10.0, 2.0) == 10.0


class TestNormalizeValence:
    def test_positive(self):
        assert normalize_valence(0.5) == 0.5

    def test_negative(self):
        assert normalize_valence(-0.5) == -0.5

    def test_clamped(self):
        assert normalize_valence(2.0) == 1.0
        assert normalize_valence(-2.0) == -1.0


class TestNormalizeArousal:
    def test_valid(self):
        assert normalize_arousal(0.5) == 0.5

    def test_clamped(self):
        assert normalize_arousal(-0.5) == 0.0
        assert normalize_arousal(1.5) == 1.0


class TestEmotionalDistance:
    def test_same_point(self):
        assert emotional_distance(0.5, 0.5, 0.5, 0.5) == 0.0

    def test_known_distance(self):
        d = emotional_distance(0.0, 0.0, 1.0, 0.0)
        assert abs(d - 1.0) < 0.01


class TestIntensityLabel:
    def test_minimal(self):
        assert intensity_label(0.05) == "minimal"

    def test_low(self):
        assert intensity_label(0.2) == "low"

    def test_moderate(self):
        assert intensity_label(0.4) == "moderate"

    def test_high(self):
        assert intensity_label(0.7) == "high"

    def test_intense(self):
        assert intensity_label(0.9) == "intense"
