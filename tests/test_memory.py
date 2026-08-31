"""Tests for heart.memory modules."""

from heart.memory.working import WorkingMemory
from heart.memory.episodic import EpisodicMemory
from heart.memory.semantic import SemanticMemory
from heart.core.state import EmotionalState, EmotionCategory


def _make_state(emotion: EmotionCategory = EmotionCategory.JOY, intensity: float = 0.5):
    return EmotionalState(
        primary=emotion,
        intensity=intensity,
        valence=0.5,
        arousal=0.5,
        dominance=0.5,
        confidence=0.8,
    )


class TestWorkingMemory:
    def test_initialization(self):
        wm = WorkingMemory(capacity=10)
        assert wm.size == 0
        assert wm.capacity == 10

    def test_update_and_retrieve(self):
        wm = WorkingMemory(capacity=5)
        state = _make_state()
        wm.update(state)
        assert wm.size == 1
        recent = wm.get_recent(1)
        assert len(recent) == 1
        assert recent[0] is state

    def test_capacity_overflow(self):
        wm = WorkingMemory(capacity=3)
        states = [_make_state(intensity=i * 0.1) for i in range(5)]
        for s in states:
            wm.update(s)
        assert wm.size == 3
        recent = wm.get_recent(3)
        assert recent[0].intensity == 0.2
        assert recent[2].intensity == 0.4

    def test_get_recent_clamped(self):
        wm = WorkingMemory(capacity=5)
        wm.update(_make_state())
        result = wm.get_recent(100)
        assert len(result) == 1

    def test_get_recent_empty(self):
        wm = WorkingMemory(capacity=5)
        result = wm.get_recent(5)
        assert result == []

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.update(_make_state())
        wm.clear()
        assert wm.size == 0

    def test_get_all(self):
        wm = WorkingMemory(capacity=5)
        wm.update(_make_state(EmotionCategory.JOY))
        wm.update(_make_state(EmotionCategory.SADNESS))
        all_states = wm.get_all()
        assert len(all_states) == 2


class TestEpisodicMemory:
    def test_initialization(self):
        em = EpisodicMemory(capacity=100)
        assert em.size == 0
        assert em.capacity == 100

    def test_store_and_recall(self):
        em = EpisodicMemory(capacity=100)
        state = _make_state()
        em.store(state, "I feel happy")
        assert em.size == 1
        entries = em.recall_recent(1)
        assert len(entries) == 1
        assert entries[0].text == "I feel happy"

    def test_recall_increments_count(self):
        em = EpisodicMemory(capacity=100)
        em.store(_make_state(), "test")
        em.recall_recent(1)
        em.recall_recent(1)
        entries = em.recall_recent(1)
        assert entries[0].recall_count == 3

    def test_capacity_overflow(self):
        em = EpisodicMemory(capacity=3)
        for i in range(5):
            em.store(_make_state(), f"text_{i}")
        assert em.size == 3
        entries = em.recall_recent(3)
        assert entries[0].text == "text_2"

    def test_recall_by_emotion(self):
        em = EpisodicMemory(capacity=100)
        em.store(_make_state(EmotionCategory.JOY), "happy")
        em.store(_make_state(EmotionCategory.SADNESS), "sad")
        em.store(_make_state(EmotionCategory.JOY), "joyful")
        matches = em.recall_by_emotion("joy")
        assert len(matches) == 2

    def test_recall_by_context(self):
        em = EpisodicMemory(capacity=100)
        em.store(_make_state(), "text", context="work")
        em.store(_make_state(), "text2", context="home")
        matches = em.recall_by_context("work")
        assert len(matches) == 1

    def test_clear(self):
        em = EpisodicMemory(capacity=100)
        em.store(_make_state(), "test")
        em.clear()
        assert em.size == 0


class TestSemanticMemory:
    def test_add_context(self):
        sm = SemanticMemory()
        state = _make_state(EmotionCategory.JOY, 0.8)
        sm.add_context("work", state)
        assert sm.size == 1

    def test_reinforce_context(self):
        sm = SemanticMemory()
        sm.add_context("work", _make_state(EmotionCategory.JOY, 0.8))
        sm.add_context("work", _make_state(EmotionCategory.JOY, 0.6))
        assocs = sm.get_context_emotions("work")
        assert len(assocs) == 1
        assert assocs[0].count == 2
        assert abs(assocs[0].intensity - 0.7) < 0.01

    def test_multiple_emotions_per_context(self):
        sm = SemanticMemory()
        sm.add_context("work", _make_state(EmotionCategory.JOY, 0.8))
        sm.add_context("work", _make_state(EmotionCategory.STRESS if hasattr(EmotionCategory, 'STRESS') else EmotionCategory.ANGER, 0.3))
        assocs = sm.get_context_emotions("work")
        assert len(assocs) == 2

    def test_get_dominant_emotion(self):
        sm = SemanticMemory()
        sm.add_context("family", _make_state(EmotionCategory.TRUST, 0.9))
        dominant = sm.get_dominant_emotion_for_context("family")
        assert dominant == EmotionCategory.TRUST

    def test_get_dominant_emotion_empty(self):
        sm = SemanticMemory()
        result = sm.get_dominant_emotion_for_context("nonexistent")
        assert result is None

    def test_get_valence(self):
        sm = SemanticMemory()
        sm.add_context("happy_place", _make_state(EmotionCategory.JOY, 0.9))
        valence = sm.get_valence_for_context("happy_place")
        assert valence is not None

    def test_clear(self):
        sm = SemanticMemory()
        sm.add_context("test", _make_state())
        sm.clear()
        assert sm.size == 0
