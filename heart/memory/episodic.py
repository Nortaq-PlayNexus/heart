"""Episodic memory for storing emotional experiences.

Records emotional events with associated text, timestamps, and metadata,
enabling retrieval of past emotional experiences.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from heart.core.state import EmotionalState


@dataclass
class EpisodicEntry:
    """A single episodic memory entry.

    Attributes:
        state: The emotional state at the time of the event.
        text: The original text that triggered the emotional response.
        timestamp: When the event was recorded.
        context: Optional contextual tag for retrieval.
        recall_count: How many times this entry has been recalled.
    """
    state: EmotionalState
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: str = ""
    recall_count: int = 0


class EpisodicMemory:
    """Stores emotional experiences with timestamps and metadata.

    Maintains a bounded collection of episodic entries, ordered by
    recency. When capacity is exceeded, the oldest entries are evicted.

    Args:
        capacity: Maximum number of episodic entries to retain.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = max(1, capacity)
        self._entries: list[EpisodicEntry] = []

    def store(self, state: EmotionalState, text: str, context: str = "") -> EpisodicEntry:
        """Store a new emotional experience.

        Args:
            state: The emotional state experienced.
            text: The triggering text input.
            context: Optional contextual tag.

        Returns:
            The newly created EpisodicEntry.
        """
        entry = EpisodicEntry(state=state, text=text, context=context)
        self._entries.append(entry)

        if len(self._entries) > self._capacity:
            self._entries = self._entries[-self._capacity:]

        return entry

    def recall_recent(self, n: int = 1) -> list[EpisodicEntry]:
        """Recall the n most recent episodic entries.

        Args:
            n: Number of entries to recall.

        Returns:
            List of EpisodicEntry objects, most recent last.
        """
        n = min(n, len(self._entries))
        if n <= 0:
            return []
        entries = self._entries[-n:]
        for entry in entries:
            entry.recall_count += 1
        return entries

    def recall_by_emotion(self, emotion_value: str, limit: int = 10) -> list[EpisodicEntry]:
        """Recall entries matching a specific primary emotion.

        Args:
            emotion_value: The emotion string value to match.
            limit: Maximum entries to return.

        Returns:
            Matching entries, most recent first.
        """
        matches = [
            e for e in reversed(self._entries)
            if e.state.primary.value == emotion_value
        ]
        for entry in matches[:limit]:
            entry.recall_count += 1
        return matches[:limit]

    def recall_by_context(self, context: str, limit: int = 10) -> list[EpisodicEntry]:
        """Recall entries matching a context tag.

        Args:
            context: The context string to match.
            limit: Maximum entries to return.

        Returns:
            Matching entries, most recent first.
        """
        matches = [
            e for e in reversed(self._entries)
            if e.context == context
        ]
        for entry in matches[:limit]:
            entry.recall_count += 1
        return matches[:limit]

    def clear(self) -> None:
        """Remove all episodic entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Current number of stored entries."""
        return len(self._entries)

    @property
    def capacity(self) -> int:
        """Maximum capacity of episodic memory."""
        return self._capacity
