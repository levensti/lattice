"""Store ABC — the contract every storage implementation must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import TraceSession


class Store(ABC):
    """Abstract base class for trace persistence.

    Subclass and implement both methods, then pass an instance to
    :func:`lattice.configure`::

        class MyStore(Store):
            def save_session(self, session: TraceSession) -> None:
                ...

            def load_sessions(
                self,
                *,
                workflow: str | None = None,
                last: int | None = None,
                trace_id: str | None = None,
            ) -> list[TraceSession]:
                ...

        lattice.configure(backend=MyStore())
    """

    @abstractmethod
    def save_session(self, session: TraceSession) -> None:
        """Persist a completed trace session."""

    @abstractmethod
    def load_sessions(
        self,
        *,
        workflow: str | None = None,
        last: int | None = None,
        trace_id: str | None = None,
    ) -> list[TraceSession]:
        """Return stored sessions, most recent first.

        Args:
            workflow: Filter by exact workflow name.
            last: Return only the N most recent sessions.
            trace_id: Fetch a single session by ID.
        """
