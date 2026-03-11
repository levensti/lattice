"""Local SQLite store for automatic trace persistence.

Traces are saved automatically when a ``trace_session`` context exits.
No setup required — the database and tables are created on first use.

Query traces later with :func:`traces`::

    import lattice

    all_traces = lattice.traces()
    recent = lattice.traces(last=5)
    filtered = lattice.traces(workflow="my_workflow")
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import (
    ActionRecord,
    GroupRecord,
    TraceSession,
    TransitionRecord,
)

_DEFAULT_DB_DIR = Path.home() / ".lattice"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "traces.db"

_CREATE_TRACES_TABLE = """\
CREATE TABLE IF NOT EXISTS traces (
    trace_id          TEXT PRIMARY KEY,
    workflow_name     TEXT NOT NULL DEFAULT '',
    goal              TEXT NOT NULL DEFAULT '',
    session_score     REAL,
    score_explanation TEXT,
    data              TEXT NOT NULL,
    created_at        TEXT NOT NULL
)
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_traces_workflow
ON traces (workflow_name)
"""

# Module-level singleton — one connection per thread via threading.local.
_local = threading.local()
_db_path: Path = _DEFAULT_DB_PATH


def configure(*, db_path: str | Path | None = None) -> None:
    """Override the default database path.

    Call this before any traces are recorded::

        import lattice
        lattice.configure(db_path="/tmp/my_traces.db")
    """
    global _db_path
    if db_path is not None:
        _db_path = Path(db_path)


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the DB if needed."""
    conn = getattr(_local, "conn", None)
    path = getattr(_local, "path", None)
    if conn is not None and path == _db_path:
        return conn

    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TRACES_TABLE)
    conn.execute(_CREATE_INDEX)
    conn.commit()
    _local.conn = conn
    _local.path = _db_path
    return conn


def save_session(session: TraceSession) -> None:
    """Persist a completed trace session to the local SQLite store."""
    conn = _get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO traces "
        "(trace_id, workflow_name, goal, session_score, score_explanation, data, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            session.trace_id,
            session.workflow_name,
            session.goal,
            session.session_score,
            session.session_score_explanation,
            json.dumps(session.to_dict()),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


def _row_to_session(row: tuple) -> TraceSession:
    """Reconstruct a TraceSession from a database row."""
    trace_id, workflow_name, goal, score, explanation, data_json, created_at = row
    data = json.loads(data_json)

    actions = [ActionRecord(**a) for a in data.get("actions", [])]
    groups = [GroupRecord(**g) for g in data.get("groups", [])]
    transitions = [TransitionRecord(**t) for t in data.get("transitions", [])]

    session = TraceSession(
        trace_id=trace_id,
        workflow_name=workflow_name,
        goal=goal,
        actions=actions,
        groups=groups,
        transitions=transitions,
        session_score=score,
        session_score_explanation=explanation,
    )
    session._created_at = created_at  # type: ignore[attr-defined]
    return session


def traces(
    *,
    workflow: str | None = None,
    last: int | None = None,
    trace_id: str | None = None,
) -> list[TraceSession]:
    """Query stored traces from the local SQLite database.

    Args:
        workflow: Filter by workflow name (exact match).
        last: Return only the N most recent traces.
        trace_id: Fetch a single trace by ID.

    Returns:
        A list of :class:`TraceSession` objects, most recent first.

    Example::

        import lattice

        # All traces
        all_traces = lattice.traces()

        # Most recent
        latest = lattice.traces(last=1)

        # By workflow
        summarize_runs = lattice.traces(workflow="summarize")
    """
    conn = _get_connection()
    clauses: list[str] = []
    params: list[Any] = []

    if workflow is not None:
        clauses.append("workflow_name = ?")
        params.append(workflow)
    if trace_id is not None:
        clauses.append("trace_id = ?")
        params.append(trace_id)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit = f"LIMIT {last}" if last is not None else ""

    query = f"SELECT * FROM traces {where} ORDER BY created_at DESC {limit}"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_session(row) for row in rows]
