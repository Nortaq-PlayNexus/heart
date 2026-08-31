from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class EmotionCategory(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionalState:
    primary: EmotionCategory
    intensity: float
    valence: float
    arousal: float
    dominance: float
    confidence: float
    dimensions: dict[str, float] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_estimate: Optional[float] = None
    blended: bool = False
    blended_components: list[tuple["EmotionCategory", float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.value,
            "intensity": self.intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence,
            "dimensions": self.dimensions,
            "triggers": self.triggers,
            "timestamp": self.timestamp.isoformat(),
            "duration_estimate": self.duration_estimate,
            "blended": self.blended,
            "blended_components": [
                (e.value, s) for e, s in self.blended_components
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        return cls(
            primary=EmotionCategory(data["primary"]),
            intensity=data["intensity"],
            valence=data["valence"],
            arousal=data["arousal"],
            dominance=data["dominance"],
            confidence=data["confidence"],
            dimensions=data.get("dimensions", {}),
            triggers=data.get("triggers", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_estimate=data.get("duration_estimate"),
            blended=data.get("blended", False),
            blended_components=[
                (EmotionCategory(e), s) for e, s in data.get("blended_components", [])
            ],
        )


@dataclass
class CompositeEmotion:
    emotions: dict[EmotionCategory, float]
    dominant: EmotionCategory
    overall_valence: float
    overall_arousal: float
    overall_dominance: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))