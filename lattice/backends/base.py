"""StorageBackend protocol — the contract every backend must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..context import TraceSession


@runtime_checkable
class Store(Protocol):
    """Minimal interface for persisting and retrieving trace sessions.

    Implement both methods and pass an instance to :func:`lattice.configure`::

        class MyBackend:
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

        lattice.configure(backend=MyBackend())
    """

    def save_session(self, session: TraceSession) -> None:
        """Persist a completed trace session."""
        ...

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
        ...
