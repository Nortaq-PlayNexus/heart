from dataclasses import dataclass
from heart.models.emotion_model import EmotionScores


@dataclass
class SentimentAnalysis:
    score: float = 0.0
    label: str = "neutral"
    confidence: float = 0.0
    positive_prob: float = 0.0
    negative_prob: float = 0.0
    neutral_prob: float = 1.0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "positive_prob": self.positive_prob,
            "negative_prob": self.negative_prob,
            "neutral_prob": self.neutral_prob,
        }


class SentimentAnalyzer:
    def __init__(self):
        self._positive_threshold: float = 0.2
        self._negative_threshold: float = -0.2

    def analyze(
        self,
        scores: EmotionScores,
        text: str = "",
    ) -> SentimentAnalysis:
        valence = scores.valence
        arousal = scores.arousal

        if valence > self._positive_threshold:
            label = "positive"
        elif valence < self._negative_threshold:
            label = "negative"
        else:
            label = "neutral"

        positive_prob = max(0.0, min(1.0, (valence + 1.0) / 2.0))
        negative_prob = max(0.0, min(1.0, (-valence + 1.0) / 2.0))
        neutral_prob = 1.0 - max(positive_prob, negative_prob)

        confidence = (
            abs(valence) * 0.6 + arousal * 0.4
        ) * min(1.0, len(text.split()) / 5.0)

        return SentimentAnalysis(
            score=valence,
            label=label,
            confidence=min(confidence, 1.0),
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
        )
