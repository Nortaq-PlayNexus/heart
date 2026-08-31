"""Tests for heart.cognition modules."""

from heart.cognition.appraisal import AppraisalEngine, AppraisalResult, AppraisalType
from heart.cognition.reasoning import EmotionalReasoner, ReasoningResult
from heart.cognition.inference import InferenceEngine, InferenceResult
from heart.core.state import EmotionalState, EmotionCategory


def _make_state(emotion: EmotionCategory = EmotionCategory.JOY, intensity: float = 0.7):
    return EmotionalState(
        primary=emotion,
        intensity=intensity,
        valence=0.6,
        arousal=0.5,
        dominance=0.4,
        confidence=0.8,
    )


class TestAppraisalEngine:
    def test_appraise(self):
        engine = AppraisalEngine()
        state = _make_state()
        results = engine.appraise("This is wonderful news!", state)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, AppraisalResult)

    def test_appraisal_types(self):
        engine = AppraisalEngine()
        state = _make_state()
        results = engine.appraise("I achieved my goal!", state)
        types_found = {r.appraisal_type for r in results}
        assert AppraisalType.NOVELTY in types_found or AppraisalType.GOAL_CONDUCIVENESS in types_found


class TestEmotionalReasoner:
    def test_reason(self):
        reasoner = EmotionalReasoner()
        state = _make_state(EmotionCategory.ANGER, 0.8)
        result = reasoner.reason(state, context="Someone cut me off in traffic")
        assert isinstance(result, ReasoningResult)
        assert len(result.chain) > 0
        assert result.confidence > 0

    def test_reasoning_chain(self):
        reasoner = EmotionalReasoner()
        state = _make_state(EmotionCategory.SADNESS, 0.5)
        result = reasoner.reason(state, context="Lost my job")
        for step in result.chain:
            assert step.premise
            assert step.conclusion

    def test_engagement_inference(self):
        reasoner = EmotionalReasoner()
        assert "High activation" in reasoner._infer_engagement(0.9)
        assert "Moderate activation" in reasoner._infer_engagement(0.5)
        assert "Low activation" in reasoner._infer_engagement(0.2)
        assert "Minimal activation" in reasoner._infer_engagement(0.05)

    def test_control_inference(self):
        reasoner = EmotionalReasoner()
        assert "High sense of control" in reasoner._infer_control(0.7)
        assert "Moderate sense of agency" in reasoner._infer_control(0.3)
        assert "Neutral sense of control" in reasoner._infer_control(0.0)
        assert "Low sense of control" in reasoner._infer_control(-0.5)


class TestInferenceEngine:
    def test_infer(self):
        engine = InferenceEngine()
        state = _make_state()
        result = engine.infer(state)
        assert isinstance(result, InferenceResult)
        assert result.dominant_emotion == EmotionCategory.JOY

    def test_infer_with_history(self):
        engine = InferenceEngine()
        history = [
            _make_state(EmotionCategory.JOY, 0.3),
            _make_state(EmotionCategory.JOY, 0.5),
            _make_state(EmotionCategory.SADNESS, 0.4),
        ]
        current = _make_state(EmotionCategory.ANGER, 0.6)
        result = engine.infer(current, history=history)
        assert result.emotional_trajectory != "insufficient_data"

    def test_trajectory_stable(self):
        engine = InferenceEngine()
        history = [
            _make_state(EmotionCategory.JOY, 0.5),
            _make_state(EmotionCategory.JOY, 0.51),
            _make_state(EmotionCategory.JOY, 0.49),
        ]
        current = _make_state(EmotionCategory.JOY, 0.5)
        result = engine.infer(current, history=history)
        assert result.emotional_trajectory == "stable"

    def test_composite_emotion(self):
        engine = InferenceEngine()
        state = _make_state()
        state.blended = True
        state.blended_components = [(EmotionCategory.TRUST, 0.4)]
        result = engine.infer(state)
        assert result.composite_emotion is not None
