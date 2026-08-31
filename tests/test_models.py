"""Tests for heart.models modules."""

from heart.models.emotion_model import (
    EmotionModel,
    PlutchikModel,
    DimensionalModel,
    EmotionScores,
    PlutchikEmotion,
    PlutchikEmotionState,
    DimensionalPoint,
    EMOTION_OPPOSITES,
    EMOTION_PAIRS,
    BLEND_EMOTIONS,
)
from heart.models.personality import (
    PersonalityProfile,
    TraitVector,
    EmotionalTendency,
    BigFiveTrait,
)
import math


class TestEmotionModel:
    def test_plutchik_to_dimensional(self):
        model = EmotionModel()
        point = model.plutchik_to_dimensional(PlutchikEmotion.JOY)
        assert point.valence > 0
        assert point.arousal > 0

    def test_dimensional_to_plutchik(self):
        model = EmotionModel()
        emotion = model.dimensional_to_plutchik(0.8, 0.7)
        assert isinstance(emotion, PlutchikEmotion)

    def test_blend(self):
        model = EmotionModel()
        result = model.blend(PlutchikEmotion.JOY, PlutchikEmotion.TRUST)
        assert result is not None
        assert isinstance(result, PlutchikEmotion)

    def test_blend_unknown_pair(self):
        model = EmotionModel()
        result = model.blend(PlutchikEmotion.JOY, PlutchikEmotion.ANGER)
        assert result is None

    def test_are_opposites(self):
        model = EmotionModel()
        assert model.are_opposites(PlutchikEmotion.JOY, PlutchikEmotion.SADNESS)
        assert not model.are_opposites(PlutchikEmotion.JOY, PlutchikEmotion.TRUST)

    def test_get_color(self):
        model = EmotionModel()
        color = model.get_color(PlutchikEmotion.JOY)
        assert color.startswith("#")

    def test_intensity_scale(self):
        model = EmotionModel()
        result = model.intensity_scale(1.0, 2, 3)
        assert result == 1.5


class TestPlutchikModel:
    def test_adjacent_emotions(self):
        model = PlutchikModel()
        adjacent = model.get_adjacent_emotions(PlutchikEmotion.JOY)
        assert len(adjacent) == 2
        assert PlutchikEmotion.TRUST in adjacent
        assert PlutchikEmotion.ANTICIPATION in adjacent


class TestDimensionalModel:
    def test_locate_in_space_high_positive(self):
        model = DimensionalModel()
        result = model.locate_in_space(0.8, 0.9)
        assert result["region"] == "high_positive"

    def test_locate_in_space_low_negative(self):
        model = DimensionalModel()
        result = model.locate_in_space(-0.5, 0.2)
        assert result["region"] == "low_negative"

    def test_locate_in_space_calm(self):
        model = DimensionalModel()
        result = model.locate_in_space(0.0, 0.3)
        assert result["region"] == "calm"


class TestEmotionScores:
    def test_dominant_emotion(self):
        scores = EmotionScores()
        scores.emotions[PlutchikEmotion.JOY] = 0.9
        scores.emotions[PlutchikEmotion.SADNESS] = 0.3
        assert scores.dominant_emotion() == PlutchikEmotion.JOY

    def test_top_emotions(self):
        scores = EmotionScores()
        scores.emotions[PlutchikEmotion.JOY] = 0.9
        scores.emotions[PlutchikEmotion.ANGER] = 0.7
        scores.emotions[PlutchikEmotion.FEAR] = 0.5
        top = scores.top_emotions(2)
        assert len(top) == 2
        assert top[0][0] == PlutchikEmotion.JOY

    def test_to_dict(self):
        scores = EmotionScores()
        d = scores.to_dict()
        assert "emotions" in d
        assert "valence" in d


class TestDimensionalPoint:
    def test_distance_to(self):
        p1 = DimensionalPoint(0.0, 0.0, 0.0)
        p2 = DimensionalPoint(1.0, 0.0, 0.0)
        assert abs(p1.distance_to(p2) - 1.0) < 0.01

    def test_to_dict(self):
        p = DimensionalPoint(0.5, 0.6, 0.7)
        d = p.to_dict()
        assert d["valence"] == 0.5


class TestPlutchikEmotionState:
    def test_to_dict(self):
        state = PlutchikEmotionState(emotion=PlutchikEmotion.JOY, intensity=0.8)
        d = state.to_dict()
        assert d["emotion"] == "joy"
        assert d["intensity"] == 0.8


class TestPersonalityProfile:
    def test_default_traits(self):
        profile = PersonalityProfile()
        assert profile.traits.openness == 0.5
        assert profile.emotional_range == 0.6

    def test_adjust_neuroticism(self):
        profile = PersonalityProfile()
        profile.traits.neuroticism = 0.8
        adjusted = profile.adjust_for_trait(-0.5, BigFiveTrait.NEUROTICISM)
        assert adjusted < -0.5

    def test_adjust_extraversion(self):
        profile = PersonalityProfile()
        profile.traits.extraversion = 0.8
        adjusted = profile.adjust_for_trait(0.5, BigFiveTrait.EXTRAVERSION)
        assert adjusted > 0.5

    def test_random_variation(self):
        profile = PersonalityProfile()
        original_openness = profile.traits.openness
        profile.random_variation(noise=0.1)
        assert abs(profile.traits.openness - original_openness) <= 0.1

    def test_to_dict(self):
        profile = PersonalityProfile()
        d = profile.to_dict()
        assert "traits" in d
        assert "emotional_range" in d


class TestTraitVector:
    def test_clamp(self):
        tv = TraitVector(openness=1.5, neuroticism=-0.3)
        tv.clamp()
        assert tv.openness == 1.0
        assert tv.neuroticism == 0.0

    def test_distance_to(self):
        tv1 = TraitVector()
        tv2 = TraitVector(openness=1.0)
        dist = tv1.distance_to(tv2)
        assert dist > 0

    def test_to_dict(self):
        tv = TraitVector()
        d = tv.to_dict()
        assert "openness" in d
        assert "neuroticism" in d


class TestEmotionalTendency:
    def test_emotion_affinity_preferred(self):
        et = EmotionalTendency(
            trait=BigFiveTrait.OPENNESS,
            preferred_emotions=["joy", "surprise"],
            sensitivity=1.0,
        )
        affinity = et.emotion_affinity("joy")
        assert affinity > 0.5

    def test_emotion_affinity_avoidance(self):
        et = EmotionalTendency(
            trait=BigFiveTrait.NEUROTICISM,
            avoidance_emotions=["anger"],
            sensitivity=1.0,
        )
        affinity = et.emotion_affinity("anger")
        assert affinity < 0.5
