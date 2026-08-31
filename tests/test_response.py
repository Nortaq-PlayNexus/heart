"""Tests for heart.response module."""

from heart.response.generator import ResponseGenerator
from heart.core.state import EmotionalState, EmotionCategory
from heart.models.personality import PersonalityProfile


def _make_state(emotion: EmotionCategory = EmotionCategory.JOY, intensity: float = 0.5):
    return EmotionalState(
        primary=emotion,
        intensity=intensity,
        valence=0.5,
        arousal=0.5,
        dominance=0.5,
        confidence=0.8,
    )


class TestResponseGenerator:
    def test_generate_joy(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.JOY, 0.8)
        response = gen.generate(state)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_sadness(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.SADNESS, 0.6)
        response = gen.generate(state)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_anger(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.ANGER, 0.9)
        response = gen.generate(state)
        assert isinstance(response, str)

    def test_generate_fear(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.FEAR, 0.7)
        response = gen.generate(state)
        assert isinstance(response, str)

    def test_generate_with_personality(self):
        gen = ResponseGenerator()
        personality = PersonalityProfile()
        personality.traits.neuroticism = 0.9
        state = _make_state(EmotionCategory.SADNESS, 0.8)
        response = gen.generate(state, personality)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_blended(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.JOY, 0.7)
        state.blended = True
        state.blended_components = [(EmotionCategory.TRUST, 0.4)]
        response = gen.generate(state)
        assert isinstance(response, str)

    def test_high_intensity_response(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.JOY, 0.95)
        response = gen.generate(state)
        assert len(response) > 0

    def test_low_intensity_response(self):
        gen = ResponseGenerator()
        state = _make_state(EmotionCategory.SADNESS, 0.1)
        response = gen.generate(state)
        assert len(response) > 0
