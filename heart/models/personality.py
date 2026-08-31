from dataclasses import dataclass, field
from enum import Enum
import math
import random


class BigFiveTrait(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class TraitVector:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def normalize(self) -> None:
        traits = [
            self.openness,
            self.conscientiousness,
            self.extraversion,
            self.agreeableness,
            self.neuroticism,
        ]
        max_val = max(traits) if max(traits) > 0 else 1.0
        self.openness /= max_val
        self.conscientiousness /= max_val
        self.extraversion /= max_val
        self.agreeableness /= max_val
        self.neuroticism /= max_val

    def clamp(self) -> None:
        self.openness = max(0.0, min(1.0, self.openness))
        self.conscientiousness = max(0.0, min(1.0, self.conscientiousness))
        self.extraversion = max(0.0, min(1.0, self.extraversion))
        self.agreeableness = max(0.0, min(1.0, self.agreeableness))
        self.neuroticism = max(0.0, min(1.0, self.neuroticism))

    def distance_to(self, other: "TraitVector") -> float:
        return math.sqrt(
            (self.openness - other.openness) ** 2
            + (self.conscientiousness - other.conscientiousness) ** 2
            + (self.extraversion - other.extraversion) ** 2
            + (self.agreeableness - other.agreeableness) ** 2
            + (self.neuroticism - other.neuroticism) ** 2
        )

    def to_dict(self) -> dict:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


@dataclass
class EmotionalTendency:
    trait: BigFiveTrait
    preferred_emotions: list[str] = field(default_factory=list)
    avoidance_emotions: list[str] = field(default_factory=list)
    sensitivity: float = 0.5

    def emotion_affinity(self, emotion: str) -> float:
        base = 0.5
        if emotion in self.preferred_emotions:
            base += 0.3 * self.sensitivity
        if emotion in self.avoidance_emotions:
            base -= 0.3 * self.sensitivity
        return max(0.0, min(1.0, base))


@dataclass
class PersonalityProfile:
    traits: TraitVector = field(default_factory=TraitVector)
    tendencies: list[EmotionalTendency] = field(default_factory=list)
    emotional_range: float = 0.6
    recovery_time: float = 0.5
    expressiveness: float = 0.5

    def adjust_for_trait(self, emotion_valence: float, trait: BigFiveTrait) -> float:
        trait_val = getattr(self.traits, trait.value)
        if trait == BigFiveTrait.NEUROTICISM:
            return emotion_valence * (1.0 + trait_val * 0.5)
        elif trait == BigFiveTrait.EXTRAVERSION:
            return emotion_valence * (1.0 + trait_val * 0.2)
        elif trait == BigFiveTrait.AGREEABLENESS:
            return emotion_valence * (1.0 - trait_val * 0.15)
        return emotion_valence

    def random_variation(self, noise: float = 0.05) -> None:
        self.traits.openness = max(
            0.0, min(1.0, self.traits.openness + random.uniform(-noise, noise))
        )
        self.traits.conscientiousness = max(
            0.0, min(1.0, self.traits.conscientiousness + random.uniform(-noise, noise))
        )
        self.traits.extraversion = max(
            0.0, min(1.0, self.traits.extraversion + random.uniform(-noise, noise))
        )
        self.traits.agreeableness = max(
            0.0, min(1.0, self.traits.agreeableness + random.uniform(-noise, noise))
        )
        self.traits.neuroticism = max(
            0.0, min(1.0, self.traits.neuroticism + random.uniform(-noise, noise))
        )

    def to_dict(self) -> dict:
        return {
            "traits": self.traits.to_dict(),
            "tendencies": [
                {
                    "trait": t.trait.value,
                    "preferred": t.preferred_emotions,
                    "avoidance": t.avoidance_emotions,
                    "sensitivity": t.sensitivity,
                }
                for t in self.tendencies
            ],
            "emotional_range": self.emotional_range,
            "recovery_time": self.recovery_time,
            "expressiveness": self.expressiveness,
        }