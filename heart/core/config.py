from dataclasses import dataclass, field


@dataclass
class HeartConfig:
    default_valence_range: tuple[float, float] = (-1.0, 1.0)
    default_arousal_range: tuple[float, float] = (0.0, 1.0)
    decay_rate: float = 0.95
    memory_capacity: int = 1000
    context_window: int = 10
    min_confidence: float = 0.1
    primary_emotion_threshold: float = 0.5
    dimensional_weights: dict[str, float] = field(
        default_factory=lambda: {"valence": 0.5, "arousal": 0.3, "dominance": 0.2}
    )
    plutchik_intensity_levels: int = 3
    personality_trait_noise: float = 0.05
    emotional_blend_enabled: bool = True
    memory_retention_days: int = 30
    working_memory_size: int = 7
