"""Working memory for maintaining immediate emotional context.

Implements a bounded buffer of recent emotional states, analogous to
human working memory with a limited capacity window.
"""

from collections import deque
from heart.core.state import EmotionalState


class WorkingMemory:
    """A bounded buffer storing the most recent emotional states.

    Implements a fixed-capacity FIFO queue. When capacity is exceeded,
    the oldest emotional state is evicted automatically.

    Args:
        capacity: Maximum number of emotional states to retain.
    """

    def __init__(self, capacity: int = 7):
        self._capacity = max(1, capacity)
        self._buffer: deque[EmotionalState] = deque(maxlen=self._capacity)

    def update(self, state: EmotionalState) -> None:
        """Add a new emotional state to working memory.

        If at capacity, the oldest state is automatically evicted.

        Args:
            state: The emotional state to store.
        """
        self._buffer.append(state)

    def get_recent(self, n: int = 1) -> list[EmotionalState]:
        """Retrieve the n most recent emotional states.

        Args:
            n: Number of recent states to retrieve (clamped to buffer size).

        Returns:
            List of EmotionalState objects, most recent last.
        """
        n = min(n, len(self._buffer))
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def get_all(self) -> list[EmotionalState]:
        """Retrieve all stored emotional states.

        Returns:
            List of all EmotionalState objects in chronological order.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all stored emotional states."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Current number of stored emotional states."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Maximum capacity of working memory."""
        return self._capacity
