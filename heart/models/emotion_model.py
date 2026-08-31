from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class PlutchikEmotion(Enum):
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


EMOTION_COLOR_MAP = {
    PlutchikEmotion.JOY: "#FFD700",
    PlutchikEmotion.TRUST: "#87CEEB",
    PlutchikEmotion.FEAR: "#9B59B6",
    PlutchikEmotion.SURPRISE: "#F39C12",
    PlutchikEmotion.SADNESS: "#3498DB",
    PlutchikEmotion.DISGUST: "#2ECC71",
    PlutchikEmotion.ANGER: "#E74C3C",
    PlutchikEmotion.ANTICIPATION: "#E67E22",
}

EMOTION_OPPOSITES = {
    PlutchikEmotion.JOY: PlutchikEmotion.SADNESS,
    PlutchikEmotion.TRUST: PlutchikEmotion.DISGUST,
    PlutchikEmotion.FEAR: PlutchikEmotion.ANGER,
    PlutchikEmotion.SURPRISE: PlutchikEmotion.ANTICIPATION,
}

EMOTION_PAIRS = [
    (PlutchikEmotion.JOY, PlutchikEmotion.TRUST),
    (PlutchikEmotion.TRUST, PlutchikEmotion.FEAR),
    (PlutchikEmotion.FEAR, PlutchikEmotion.SURPRISE),
    (PlutchikEmotion.SURPRISE, PlutchikEmotion.SADNESS),
    (PlutchikEmotion.SADNESS, PlutchikEmotion.DISGUST),
    (PlutchikEmotion.DISGUST, PlutchikEmotion.ANGER),
    (PlutchikEmotion.ANGER, PlutchikEmotion.ANTICIPATION),
    (PlutchikEmotion.ANTICIPATION, PlutchikEmotion.JOY),
]

BLEND_EMOTIONS: dict[tuple[PlutchikEmotion, PlutchikEmotion], PlutchikEmotion] = {
    (PlutchikEmotion.JOY, PlutchikEmotion.TRUST): PlutchikEmotion.JOY,
    (PlutchikEmotion.TRUST, PlutchikEmotion.FEAR): PlutchikEmotion.FEAR,
    (PlutchikEmotion.FEAR, PlutchikEmotion.SURPRISE): PlutchikEmotion.SURPRISE,
    (PlutchikEmotion.SURPRISE, PlutchikEmotion.SADNESS): PlutchikEmotion.SADNESS,
    (PlutchikEmotion.SADNESS, PlutchikEmotion.DISGUST): PlutchikEmotion.SADNESS,
    (PlutchikEmotion.DISGUST, PlutchikEmotion.ANGER): PlutchikEmotion.DISGUST,
    (PlutchikEmotion.ANGER, PlutchikEmotion.ANTICIPATION): PlutchikEmotion.ANTICIPATION,
    (PlutchikEmotion.ANTICIPATION, PlutchikEmotion.JOY): PlutchikEmotion.ANTICIPATION,
}

DIMENSIONAL_REGIONS = {
    "joy": {"valence": (0.5, 1.0), "arousal": (0.5, 1.0)},
    "trust": {"valence": (0.3, 1.0), "arousal": (0.0, 0.5)},
    "fear": {"valence": (-1.0, 0.0), "arousal": (0.5, 1.0)},
    "surprise": {"valence": (0.0, 0.5), "arousal": (0.7, 1.0)},
    "sadness": {"valence": (-1.0, -0.3), "arousal": (0.0, 0.4)},
    "disgust": {"valence": (-1.0, -0.2), "arousal": (0.3, 0.8)},
    "anger": {"valence": (-1.0, 0.0), "arousal": (0.6, 1.0)},
    "anticipation": {"valence": (0.0, 0.6), "arousal": (0.4, 0.9)},
}


@dataclass
class DimensionalPoint:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def distance_to(self, other: "DimensionalPoint") -> float:
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def to_dict(self) -> dict:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}


@dataclass
class PlutchikEmotionState:
    emotion: PlutchikEmotion
    intensity: float = 0.0
    color: str = "#FFFFFF"

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion.value,
            "intensity": self.intensity,
            "color": self.color,
        }


@dataclass
class EmotionScores:
    emotions: dict[PlutchikEmotion, float] = field(
        default_factory=lambda: {e: 0.0 for e in PlutchikEmotion}
    )
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    confidence: float = 0.0

    def dominant_emotion(self) -> PlutchikEmotion:
        return max(self.emotions, key=self.emotions.get)

    def top_emotions(self, n: int = 3) -> list[tuple[PlutchikEmotion, float]]:
        sorted_emotions = sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:n]

    def to_dict(self) -> dict:
        return {
            "emotions": {e.value: s for e, s in self.emotions.items()},
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


class EmotionModel:
    def __init__(self):
        self._plutchik_map: dict[PlutchikEmotion, DimensionalPoint] = {
            PlutchikEmotion.JOY: DimensionalPoint(valence=0.8, arousal=0.7, dominance=0.6),
            PlutchikEmotion.TRUST: DimensionalPoint(valence=0.7, arousal=0.3, dominance=0.4),
            PlutchikEmotion.FEAR: DimensionalPoint(valence=-0.6, arousal=0.8, dominance=-0.5),
            PlutchikEmotion.SURPRISE: DimensionalPoint(valence=0.0, arousal=0.9, dominance=0.1),
            PlutchikEmotion.SADNESS: DimensionalPoint(valence=-0.7, arousal=0.2, dominance=-0.3),
            PlutchikEmotion.DISGUST: DimensionalPoint(valence=-0.8, arousal=0.4, dominance=-0.4),
            PlutchikEmotion.ANGER: DimensionalPoint(valence=-0.5, arousal=0.8, dominance=0.7),
            PlutchikEmotion.ANTICIPATION: DimensionalPoint(valence=0.3, arousal=0.6, dominance=0.3),
        }

    def plutchik_to_dimensional(self, emotion: PlutchikEmotion) -> DimensionalPoint:
        return self._plutchik_map.get(emotion, DimensionalPoint())

    def dimensional_to_plutchik(
        self, valence: float, arousal: float, dominance: float = 0.0
    ) -> PlutchikEmotion:
        closest = PlutchikEmotion.JOY
        min_dist = float("inf")
        for emotion, point in self._plutchik_map.items():
            dist = math.sqrt(
                (valence - point.valence) ** 2
                + (arousal - point.arousal) ** 2
                + (dominance - point.dominance) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest = emotion
        return closest

    def blend(self, e1: PlutchikEmotion, e2: PlutchikEmotion) -> Optional[PlutchikEmotion]:
        pair = (e1, e2)
        if pair in BLEND_EMOTIONS:
            return BLEND_EMOTIONS[pair]
        pair_rev = (e2, e1)
        if pair_rev in BLEND_EMOTIONS:
            return BLEND_EMOTIONS[pair_rev]
        return None

    def intensity_scale(self, base: float, level: int, max_levels: int = 3) -> float:
        return base * (1.0 + (level - 1) * 0.5)

    def get_color(self, emotion: PlutchikEmotion) -> str:
        return EMOTION_COLOR_MAP.get(emotion, "#FFFFFF")

    def are_opposites(self, e1: PlutchikEmotion, e2: PlutchikEmotion) -> bool:
        return EMOTION_OPPOSITES.get(e1) == e2


class PlutchikModel(EmotionModel):
    def __init__(self):
        super().__init__()
        self.wheel_radius = 8

    def get_adjacent_emotions(self, emotion: PlutchikEmotion) -> list[PlutchikEmotion]:
        idx = list(PlutchikEmotion).index(emotion)
        adjacent = []
        for offset in [-1, 1]:
            adjacent.append(list(PlutchikEmotion)[(idx + offset) % len(PlutchikEmotion)])
        return adjacent


class DimensionalModel(EmotionModel):
    def __init__(self):
        super().__init__()
        self.circumplex: dict[str, DimensionalPoint] = {
            e.value: point for e, point in self._plutchik_map.items()
        }

    def locate_in_space(self, valence: float, arousal: float) -> dict:
        region = "neutral"
        if valence > 0.3 and arousal > 0.5:
            region = "high_positive"
        elif valence > 0.3 and arousal <= 0.5:
            region = "low_positive"
        elif valence <= 0.3 and valence >= -0.3 and arousal > 0.5:
            region = "high_arousal"
        elif valence <= 0.3 and valence >= -0.3 and arousal <= 0.5:
            region = "calm"
        elif valence < -0.3 and arousal > 0.5:
            region = "high_negative_aroused"
        elif valence < -0.3 and arousal <= 0.5:
            region = "low_negative"
        regions = self.circumplex
        closest_emotion = min(
            regions.items(),
            key=lambda item: math.sqrt(
                (valence - item[1].valence) ** 2 + (arousal - item[1].arousal) ** 2
            ),
        )
        return {"region": region, "closest_emotion": closest_emotion[0], "valence": valence, "arousal": arousal}