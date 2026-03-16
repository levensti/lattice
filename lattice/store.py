"""Trace persistence — thin shim that delegates to the active Store.

The default backend is :class:`~lattice.backends.SQLiteStore` (zero config).
Swap it out via :func:`configure`::

    import lattice
    lattice.configure(backend=MyBackend())

Query traces with :func:`traces`::

    import lattice

    all_traces = lattice.traces()
    recent = lattice.traces(last=5)
    filtered = lattice.traces(workflow="my_workflow")
"""

from __future__ import annotations

from pathlib import Path

from .backends.base import Store
from .backends.sqlite import DEFAULT_DB_PATH, SQLiteStore
from .context import TraceSession

# Re-export so external code that does `from lattice.store import _DEFAULT_DB_PATH`
# keeps working.
_DEFAULT_DB_PATH = DEFAULT_DB_PATH

_backend: Store = SQLiteStore()


def configure(
    *,
    db_path: str | Path | None = None,
    backend: Store | None = None,
) -> None:
    """Override the default storage backend.

    Pass *either* ``db_path`` to keep using SQLite at a custom path, *or*
    ``backend`` to plug in a completely different implementation::

        import lattice

        # custom SQLite path
        lattice.configure(db_path="/tmp/my_traces.db")

        # custom backend
        from my_package import PostgresBackend
        lattice.configure(backend=PostgresBackend(os.environ["DATABASE_URL"]))
    """
    global _backend
    if backend is not None:
        _backend = backend
    elif db_path is not None:
        _backend = SQLiteStore(db_path)


def save_session(session: TraceSession) -> None:
    """Persist a completed trace session via the active backend."""
    _backend.save_session(session)


def traces(
    *,
    workflow: str | None = None,
    last: int | None = None,
    trace_id: str | None = None,
) -> list[TraceSession]:
    """Query stored traces via the active backend.

    Args:
        workflow: Filter by workflow name (exact match).
        last: Return only the N most recent traces.
        trace_id: Fetch a single trace by ID.

    Returns:
        A list of :class:`TraceSession` objects, most recent first.
    """
    return _backend.load_sessions(workflow=workflow, last=last, trace_id=trace_id)
