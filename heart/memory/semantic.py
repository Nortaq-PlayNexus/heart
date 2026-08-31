"""Semantic memory for emotional knowledge and context associations.

Stores associations between contextual concepts and emotional states,
building a growing knowledge base of emotional semantics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from heart.core.state import EmotionalState, EmotionCategory


@dataclass
class SemanticAssociation:
    """An association between a context and emotional knowledge.

    Attributes:
        context: The contextual concept or phrase.
        emotion: The associated emotion category.
        intensity: Average intensity observed for this association.
        valence: Average valence observed.
        arousal: Average arousal observed.
        count: Number of times this association has been reinforced.
        last_seen: When this association was last updated.
    """
    context: str
    emotion: EmotionCategory
    intensity: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    count: int = 0
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticMemory:
    """Stores emotional knowledge as context-emotion associations.

    Maintains a dictionary of contextual concepts mapped to emotional
    profiles, allowing the system to learn emotional associations over time.
    """

    def __init__(self):
        self._associations: dict[str, list[SemanticAssociation]] = {}

    def add_context(self, context: str, state: EmotionalState) -> None:
        """Associate a context with an emotional state.

        If the context already has an association with this emotion,
        the existing association is reinforced with a running average.

        Args:
            context: The contextual concept.
            state: The emotional state to associate.
        """
        key = context.strip().lower()
        if key not in self._associations:
            self._associations[key] = []

        existing = None
        for assoc in self._associations[key]:
            if assoc.emotion == state.primary:
                existing = assoc
                break

        if existing:
            existing.count += 1
            n = existing.count
            existing.intensity = (existing.intensity * (n - 1) + state.intensity) / n
            existing.valence = (existing.valence * (n - 1) + state.valence) / n
            existing.arousal = (existing.arousal * (n - 1) + state.arousal) / n
            existing.last_seen = datetime.now(timezone.utc)
        else:
            self._associations[key].append(
                SemanticAssociation(
                    context=context,
                    emotion=state.primary,
                    intensity=state.intensity,
                    valence=state.valence,
                    arousal=state.arousal,
                    count=1,
                )
            )

    def get_context_emotions(self, context: str) -> list[SemanticAssociation]:
        """Retrieve emotional associations for a context.

        Args:
            context: The contextual concept to look up.

        Returns:
            List of SemanticAssociation objects, ordered by frequency.
        """
        key = context.strip().lower()
        assocs = self._associations.get(key, [])
        return sorted(assocs, key=lambda a: a.count, reverse=True)

    def get_dominant_emotion_for_context(self, context: str) -> EmotionCategory | None:
        """Get the most frequently associated emotion for a context.

        Args:
            context: The contextual concept to look up.

        Returns:
            The dominant EmotionCategory, or None if no associations exist.
        """
        assocs = self.get_context_emotions(context)
        if not assocs:
            return None
        return assocs[0].emotion

    def get_valence_for_context(self, context: str) -> float | None:
        """Get the average valence associated with a context.

        Args:
            context: The contextual concept to look up.

        Returns:
            Average valence, or None if no associations exist.
        """
        assocs = self.get_context_emotions(context)
        if not assocs:
            return None
        top = assocs[0]
        return top.valence

    def get_all_contexts(self) -> list[str]:
        """Retrieve all stored context keys.

        Returns:
            List of context strings.
        """
        return list(self._associations.keys())

    def clear(self) -> None:
        """Remove all semantic associations."""
        self._associations.clear()

    @property
    def size(self) -> int:
        """Total number of context entries."""
        return len(self._associations)
