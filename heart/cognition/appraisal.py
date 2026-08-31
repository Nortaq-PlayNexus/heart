from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from heart.core.state import EmotionalState, EmotionCategory
from heart.models.emotion_model import PlutchikEmotion, DimensionalModel


class AppraisalType(Enum):
    NOVELTY = "novelty"
    INTRINSIC_SATISFACTION = "intrinsic_satisfaction"
    GOAL_CONDUCIVENESS = "goal_conduciveness"
    MEANING_CONSEQUENCE = "meaning_consequence"
    COPING_POTENTIAL = "coping_potential"
    NORM_COMPATIBILITY = "norm_compatibility"


@dataclass
class AppraisalResult:
    appraisal_type: AppraisalType
    score: float
    label: str
    explanation: str
    emotion_implication: Optional[str] = None


@dataclass
class AppraisalContext:
    novelty: float = 0.0
    goal_conduciveness: float = 0.0
    coping_potential: float = 0.0
    norm_compatibility: float = 0.0
    intrinsic_satisfaction: float = 0.0
    meaning_consequence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "novelty": self.novelty,
            "goal_conduciveness": self.goal_conduciveness,
            "coping_potential": self.coping_potential,
            "norm_compatibility": self.norm_compatibility,
            "intrinsic_satisfaction": self.intrinsic_satisfaction,
            "meaning_consequence": self.meaning_consequence,
        }


class AppraisalEngine:
    def __init__(self):
        self.dimensional_model = DimensionalModel()

    def appraise(self, text: str, state: EmotionalState) -> list[AppraisalResult]:
        results: list[AppraisalResult] = []

        novelty = self._compute_novelty(text, state)
        results.append(AppraisalResult(
            appraisal_type=AppraisalType.NOVELTY,
            score=novelty,
            label="high" if novelty > 0.5 else "low",
            explanation=f"Novelty score of {novelty:.3f} indicates {'unfamiliar' if novelty > 0.5 else 'familiar'} stimulus",
            emotion_implication="surprise" if novelty > 0.6 else None,
        ))

        goal_conduciveness = self._compute_goal_conduciveness(state)
        results.append(AppraisalResult(
            appraisal_type=AppraisalType.GOAL_CONDUCIVENESS,
            score=goal_conduciveness,
            label="positive" if goal_conduciveness > 0.3 else ("negative" if goal_conduciveness < -0.3 else "neutral"),
            explanation=f"Goal conduciveness: {goal_conduciveness:.3f}",
            emotion_implication="joy" if goal_conduciveness > 0.5 else ("sadness" if goal_conduciveness < -0.5 else None),
        ))

        coping = self._compute_coping_potential(state)
        results.append(AppraisalResult(
            appraisal_type=AppraisalType.COPING_POTENTIAL,
            score=coping,
            label="high" if coping > 0.3 else ("low" if coping < -0.3 else "moderate"),
            explanation=f"Coping potential: {coping:.3f}",
            emotion_implication="trust" if coping > 0.5 else ("fear" if coping < -0.5 else None),
        ))

        norm_compat = self._compute_norm_compatibility(state)
        results.append(AppraisalResult(
            appraisal_type=AppraisalType.NORM_COMPATIBILITY,
            score=norm_compat,
            label="compatible" if norm_compat > 0.3 else ("violated" if norm_compat < -0.3 else "neutral"),
            explanation=f"Norm compatibility: {norm_compat:.3f}",
            emotion_implication="trust" if norm_compat > 0.5 else ("disgust" if norm_compat < -0.5 else None),
        ))

        intrinsic = self._compute_intrinsic_satisfaction(state)
        results.append(AppraisalResult(
            appraisal_type=AppraisalType.INTRINSIC_SATISFACTION,
            score=intrinsic,
            label="satisfying" if intrinsic > 0.3 else ("unsatisfying" if intrinsic < -0.3 else "neutral"),
            explanation=f"Intrinsic satisfaction: {intrinsic:.3f}",
            emotion_implication="joy" if intrinsic > 0.5 else ("sadness" if intrinsic < -0.5 else None),
        ))

        return results

    def _compute_novelty(self, text: str, state: EmotionalState) -> float:
        surprise_signals = ["surprise", "unexpected", "never", "always", "first", "once"]
        score = state.valence * 0.3 + state.arousal * 0.4
        for signal in surprise_signals:
            if signal in text.lower():
                score += 0.15
        return max(0.0, min(1.0, score))

    def _compute_goal_conduciveness(self, state: EmotionalState) -> float:
        return state.valence * state.intensity

    def _compute_coping_potential(self, state: EmotionalState) -> float:
        return state.dominance * state.intensity * 0.7 + state.valence * 0.3

    def _compute_norm_compatibility(self, state: EmotionalState) -> float:
        return state.valence * 0.6 + state.dominance * 0.4

    def _compute_intrinsic_satisfaction(self, state: EmotionalState) -> float:
        return state.valence * state.intensity * 0.8 + 0.2
