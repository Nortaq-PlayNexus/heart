from dataclasses import dataclass, field
from typing import Optional
from heart.core.state import EmotionalState, EmotionCategory
from heart.models.emotion_model import PlutchikEmotion, EmotionScores


@dataclass
class ReasoningStep:
    premise: str
    conclusion: str
    confidence: float = 0.0
    emotion_influence: Optional[str] = None


@dataclass
class ReasoningResult:
    chain: list[ReasoningStep] = field(default_factory=list)
    final_valence: float = 0.0
    final_arousal: float = 0.0
    final_dominance: float = 0.0
    inferred_emotion: Optional[str] = None
    confidence: float = 0.0


class EmotionalReasoner:
    def __init__(self):
        self._max_chain_length: int = 5

    def reason(
        self,
        current: EmotionalState,
        context: str = "",
    ) -> ReasoningResult:
        chain: list[ReasoningStep] = []
        valence = current.valence
        arousal = current.arousal
        dominance = current.dominance

        step1 = ReasoningStep(
            premise=f"Current emotion is {current.primary.value} with intensity {current.intensity:.2f}",
            conclusion=f"This suggests an underlying {'positive' if valence > 0 else 'negative'} evaluative state",
            confidence=0.8,
            emotion_influence=current.primary.value,
        )
        chain.append(step1)

        step2 = ReasoningStep(
            premise=f"Arousal level is {arousal:.2f}",
            conclusion=self._infer_engagement(arousal),
            confidence=0.7,
            emotion_influence="arousal_inference",
        )
        chain.append(step2)

        step3 = ReasoningStep(
            premise=f"Dominance level is {dominance:.2f}",
            conclusion=self._infer_control(dominance),
            confidence=0.65,
            emotion_influence="dominance_inference",
        )
        chain.append(step3)

        if current.blended and current.blended_components:
            step4 = ReasoningStep(
                premise=f"Emotion is blended across {len(current.blended_components)} components",
                conclusion="Multiple emotional processes are active simultaneously",
                confidence=0.6,
                emotion_influence="blend_inference",
            )
            chain.append(step4)

        step5 = ReasoningStep(
            premise=f"Context: {context[:100] if context else 'No context provided'}",
            conclusion=self._infer_cause(current, context),
            confidence=0.5,
            emotion_influence="context_inference",
        )
        chain.append(step5)

        chain = chain[: self._max_chain_length]

        final_valence = valence * 0.6 + sum(s.confidence for s in chain) / len(chain) * 0.4 * valence
        final_arousal = arousal
        final_dominance = dominance

        return ReasoningResult(
            chain=chain,
            final_valence=final_valence,
            final_arousal=final_arousal,
            final_dominance=final_dominance,
            inferred_emotion=current.primary.value,
            confidence=min(s.confidence for s in chain) if chain else 0.0,
        )

    def _infer_engagement(self, arousal: float) -> str:
        if arousal > 0.7:
            return "High activation - intense emotional engagement detected"
        elif arousal > 0.4:
            return "Moderate activation - emotional processing is active"
        elif arousal > 0.1:
            return "Low activation - mild emotional response"
        return "Minimal activation - emotional response is subdued"

    def _infer_control(self, dominance: float) -> str:
        if dominance > 0.5:
            return "High sense of control over situation"
        elif dominance > 0.2:
            return "Moderate sense of agency"
        elif dominance > -0.2:
            return "Neutral sense of control"
        return "Low sense of control - feeling powerless"

    def _infer_cause(self, state: EmotionalState, context: str) -> str:
        if state.intensity > 0.7:
            return "High-intensity emotion suggests strong causal trigger"
        elif state.intensity > 0.4:
            return "Moderate intensity indicates moderate causal impact"
        return "Low intensity suggests mild or background causal influence"