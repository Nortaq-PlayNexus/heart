"""General utility functions for H.E.A.R.T.

Provides mathematical and transformation helpers used across modules.
"""


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to a specified range.

    Args:
        value: The value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        The clamped value.
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values.

    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor (0.0 to 1.0).

    Returns:
        Interpolated value.
    """
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def normalize_valence(valence: float) -> float:
    """Normalize a valence value from arbitrary range to [-1.0, 1.0].

    Args:
        valence: Raw valence value.

    Returns:
        Normalized valence in [-1.0, 1.0].
    """
    return clamp(valence, -1.0, 1.0)


def normalize_arousal(arousal: float) -> float:
    """Normalize an arousal value to [0.0, 1.0].

    Args:
        arousal: Raw arousal value.

    Returns:
        Normalized arousal in [0.0, 1.0].
    """
    return clamp(arousal, 0.0, 1.0)


def emotional_distance(
    v1: float, a1: float, v2: float, a2: float
) -> float:
    """Compute distance between two emotional points in valence-arousal space.

    Args:
        v1: Valence of first point.
        a1: Arousal of first point.
        v2: Valence of second point.
        a2: Arousal of second point.

    Returns:
        Euclidean distance in valence-arousal space.
    """
    import math
    return math.sqrt((v1 - v2) ** 2 + (a1 - a2) ** 2)


def intensity_label(intensity: float) -> str:
    """Convert a numeric intensity to a human-readable label.

    Args:
        intensity: Intensity value in [0.0, 1.0].

    Returns:
        Label string: 'minimal', 'low', 'moderate', 'high', or 'intense'.
    """
    if intensity < 0.1:
        return "minimal"
    elif intensity < 0.3:
        return "low"
    elif intensity < 0.6:
        return "moderate"
    elif intensity < 0.8:
        return "high"
    return "intense"
