from dataclasses import dataclass, field
from typing import Optional
from heart.core.state import EmotionalState, EmotionCategory, CompositeEmotion
from heart.models.emotion_model import (
    EmotionScores,
    PlutchikEmotion,
    PlutchikModel,
    DimensionalModel,
)


@dataclass
class InferenceResult:
    dominant_emotion: EmotionCategory
    secondary_emotions: list[tuple[EmotionCategory, float]] = field(default_factory=list)
    emotional_trajectory: str = "stable"
    predicted_next_emotion: Optional[EmotionCategory] = None
    confidence: float = 0.0
    composite_emotion: Optional[CompositeEmotion] = None


class InferenceEngine:
    def __init__(self):
        self.plutchik = PlutchikModel()
        self.dimensional = DimensionalModel()

    def infer(
        self,
        current_state: EmotionalState,
        history: list[EmotionalState] = None,
    ) -> InferenceResult:
        secondary = []
        if current_state.blended and current_state.blended_components:
            for emotion, strength in current_state.blended_components:
                cat = self._emo_cat(emotion)
                secondary.append((cat, strength))

        trajectory = self._compute_trajectory(current_state, history or [])
        predicted = self._predict_next(current_state, history or [])

        composite = self._build_composite(current_state)

        confidence = current_state.confidence * 0.7 + min(
            1.0, len(history or []) * 0.1
        )

        return InferenceResult(
            dominant_emotion=current_state.primary,
            secondary_emotions=secondary[:3],
            emotional_trajectory=trajectory,
            predicted_next_emotion=predicted,
            confidence=min(confidence, 1.0),
            composite_emotion=composite,
        )

    def _compute_trajectory(
        self,
        current: EmotionalState,
        history: list[EmotionalState],
    ) -> str:
        if len(history) < 2:
            return "insufficient_data"

        recent = history[-5:]
        valence_trend = sum(
            recent[i].valence - recent[i - 1].valence for i in range(1, len(recent))
        )
        intensity_trend = sum(
            recent[i].intensity - recent[i - 1].intensity for i in range(1, len(recent))
        )

        if abs(valence_trend) < 0.1 and abs(intensity_trend) < 0.1:
            return "stable"
        elif valence_trend > 0 and intensity_trend > 0:
            return "escalating_positive"
        elif valence_trend < 0 and intensity_trend > 0:
            return "escalating_negative"
        elif valence_trend > 0 and intensity_trend < 0:
            return "calming_positive"
        elif valence_trend < 0 and intensity_trend < 0:
            return "calming_negative"
        elif valence_trend > 0:
            return "improving"
        elif valence_trend < 0:
            return "deteriorating"
        return "fluctuating"

    def _predict_next(
        self,
        current: EmotionalState,
        history: list[EmotionalState],
    ) -> Optional[EmotionCategory]:
        if not history:
            return None

        mapping = {
            EmotionCategory.JOY: PlutchikEmotion.JOY,
            EmotionCategory.TRUST: PlutchikEmotion.TRUST,
            EmotionCategory.FEAR: PlutchikEmotion.FEAR,
            EmotionCategory.SURPRISE: PlutchikEmotion.SURPRISE,
            EmotionCategory.SADNESS: PlutchikEmotion.SADNESS,
            EmotionCategory.DISGUST: PlutchikEmotion.DISGUST,
            EmotionCategory.ANGER: PlutchikEmotion.ANGER,
            EmotionCategory.ANTICIPATION: PlutchikEmotion.ANTICIPATION,
        }

        reverse_mapping = {v: k for k, v in mapping.items()}

        for prev in reversed(history[-3:]):
            prev_plutchik = mapping.get(prev.primary)
            if prev_plutchik is None:
                continue
            adjacent = self.plutchik.get_adjacent_emotions(prev_plutchik)
            for adj in adjacent:
                adj_cat = reverse_mapping.get(adj)
                if adj_cat and adj_cat != current.primary:
                    return adj_cat

        return None

    def _build_composite(self, state: EmotionalState) -> CompositeEmotion:
        emotions = {state.primary: state.intensity}
        if state.blended and state.blended_components:
            for emotion, strength in state.blended_components:
                emotions[emotion] = max(emotions.get(emotion, 0.0), strength)

        total = sum(emotions.values())
        if total > 0:
            valence = sum(
                self._emotion_valence(e) * s for e, s in emotions.items()
            ) / total
            arousal = sum(
                self._emotion_arousal(e) * s for e, s in emotions.items()
            ) / total
            dominance = sum(
                self._emotion_dominance(e) * s for e, s in emotions.items()
            ) / total
        else:
            valence = 0.0
            arousal = 0.0
            dominance = 0.0

        dominant = max(emotions, key=emotions.get)

        return CompositeEmotion(
            emotions=emotions,
            dominant=dominant,
            overall_valence=valence,
            overall_arousal=arousal,
            overall_dominance=dominance,
        )

    @staticmethod
    def _emotion_valence(emotion: EmotionCategory) -> float:
        mapping = {
            EmotionCategory.JOY: 0.8,
            EmotionCategory.TRUST: 0.6,
            EmotionCategory.FEAR: -0.6,
            EmotionCategory.SURPRISE: 0.0,
            EmotionCategory.SADNESS: -0.7,
            EmotionCategory.DISGUST: -0.7,
            EmotionCategory.ANGER: -0.5,
            EmotionCategory.ANTICIPATION: 0.3,
        }
        return mapping.get(emotion, 0.0)

    @staticmethod
    def _emotion_arousal(emotion: EmotionCategory) -> float:
        mapping = {
            EmotionCategory.JOY: 0.7,
            EmotionCategory.TRUST: 0.3,
            EmotionCategory.FEAR: 0.8,
            EmotionCategory.SURPRISE: 0.9,
            EmotionCategory.SADNESS: 0.2,
            EmotionCategory.DISGUST: 0.5,
            EmotionCategory.ANGER: 0.8,
            EmotionCategory.ANTICIPATION: 0.6,
        }
        return mapping.get(emotion, 0.5)

    @staticmethod
    def _emotion_dominance(emotion: EmotionCategory) -> float:
        mapping = {
            EmotionCategory.JOY: 0.6,
            EmotionCategory.TRUST: 0.4,
            EmotionCategory.FEAR: -0.5,
            EmotionCategory.SURPRISE: 0.1,
            EmotionCategory.SADNESS: -0.3,
            EmotionCategory.DISGUST: -0.4,
            EmotionCategory.ANGER: 0.7,
            EmotionCategory.ANTICIPATION: 0.3,
        }
        return mapping.get(emotion, 0.0)

    @staticmethod
    def _emo_cat(emotion: PlutchikEmotion) -> EmotionCategory:
        mapping = {
            PlutchikEmotion.JOY: EmotionCategory.JOY,
            PlutchikEmotion.TRUST: EmotionCategory.TRUST,
            PlutchikEmotion.FEAR: EmotionCategory.FEAR,
            PlutchikEmotion.SURPRISE: EmotionCategory.SURPRISE,
            PlutchikEmotion.SADNESS: EmotionCategory.SADNESS,
            PlutchikEmotion.DISGUST: EmotionCategory.DISGUST,
            PlutchikEmotion.ANGER: EmotionCategory.ANGER,
            PlutchikEmotion.ANTICIPATION: EmotionCategory.ANTICIPATION,
        }
        return mapping.get(emotion, EmotionCategory.JOY)