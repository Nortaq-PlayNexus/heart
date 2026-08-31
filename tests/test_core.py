"""Tests for heart.core modules: config, state, engine."""

import json
from datetime import datetime, timezone

from heart.core.config import HeartConfig
from heart.core.state import EmotionalState, EmotionCategory, CompositeEmotion
from heart.core.engine import HeartEngine, state_to_emotion_scores
from heart.models.emotion_model import PlutchikEmotion, EmotionScores


class TestHeartConfig:
    def test_defaults(self):
        config = HeartConfig()
        assert config.decay_rate == 0.95
        assert config.memory_capacity == 1000
        assert config.working_memory_size == 7
        assert config.min_confidence == 0.1
        assert config.emotional_blend_enabled is True

    def test_custom_config(self):
        config = HeartConfig(memory_capacity=500, working_memory_size=3)
        assert config.memory_capacity == 500
        assert config.working_memory_size == 3


class TestEmotionalState:
    def test_creation(self):
        state = EmotionalState(
            primary=EmotionCategory.JOY,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
        )
        assert state.primary == EmotionCategory.JOY
        assert state.intensity == 0.8
        assert state.blended is False
        assert state.blended_components == []

    def test_to_dict(self):
        state = EmotionalState(
            primary=EmotionCategory.JOY,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
        )
        d = state.to_dict()
        assert d["primary"] == "joy"
        assert d["intensity"] == 0.8
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        state = EmotionalState(
            primary=EmotionCategory.SADNESS,
            intensity=0.5,
            valence=-0.6,
            arousal=0.3,
            dominance=-0.2,
            confidence=0.7,
            triggers=["loss"],
        )
        d = state.to_dict()
        restored = EmotionalState.from_dict(d)
        assert restored.primary == EmotionCategory.SADNESS
        assert restored.intensity == 0.5
        assert restored.valence == -0.6
        assert restored.triggers == ["loss"]

    def test_composite_emotion(self):
        comp = CompositeEmotion(
            emotions={EmotionCategory.JOY: 0.8, EmotionCategory.TRUST: 0.4},
            dominant=EmotionCategory.JOY,
            overall_valence=0.7,
            overall_arousal=0.6,
            overall_dominance=0.5,
        )
        assert comp.dominant == EmotionCategory.JOY
        assert len(comp.emotions) == 2


class TestStateToEmotionScores:
    def test_joy_state(self):
        state = EmotionalState(
            primary=EmotionCategory.JOY,
            intensity=0.9,
            valence=0.8,
            arousal=0.7,
            dominance=0.6,
            confidence=0.95,
        )
        scores = state_to_emotion_scores(state)
        assert scores.emotions[PlutchikEmotion.JOY] == 0.9
        assert scores.valence == 0.8
        assert scores.arousal == 0.7

    def test_sadness_state(self):
        state = EmotionalState(
            primary=EmotionCategory.SADNESS,
            intensity=0.4,
            valence=-0.5,
            arousal=0.2,
            dominance=-0.3,
            confidence=0.6,
        )
        scores = state_to_emotion_scores(state)
        assert scores.emotions[PlutchikEmotion.SADNESS] == 0.4


class TestHeartEngine:
    def test_process_returns_state(self):
        engine = HeartEngine()
        state = engine.process("I am so happy today!")
        assert isinstance(state, EmotionalState)
        assert state.confidence >= 0

    def test_process_updates_memory(self):
        engine = HeartEngine()
        engine.process("Feeling great!")
        assert engine.working_memory.size == 1
        assert engine.episodic_memory.size == 1

    def test_process_with_context(self):
        engine = HeartEngine()
        engine.process("I love this", context="work")
        assert engine.semantic_memory.size == 1

    def test_respond(self):
        engine = HeartEngine()
        result = engine.respond("I feel terrible about what happened.")
        assert "emotional_state" in result
        assert "response_text" in result
        assert "primary_emotion" in result
        assert isinstance(result["response_text"], str)
        assert len(result["response_text"]) > 0

    def test_emotional_profile_empty(self):
        engine = HeartEngine()
        profile = engine.get_emotional_profile()
        assert profile["primary"] == "neutral"

    def test_emotional_profile_after_processing(self):
        engine = HeartEngine()
        engine.process("I am joyful and excited!")
        engine.process("Feeling happy about the news.")
        profile = engine.get_emotional_profile()
        assert "average_valence" in profile
        assert "recent_emotions" in profile
        assert len(profile["recent_emotions"]) == 2

    def test_multiple_interactions(self):
        engine = HeartEngine()
        texts = [
            "I feel great!",
            "This is terrible.",
            "Wow, unexpected news!",
            "I'm a bit worried.",
        ]
        for text in texts:
            state = engine.process(text)
            assert isinstance(state, EmotionalState)
        assert engine.working_memory.size == 4
        assert engine.episodic_memory.size == 4
