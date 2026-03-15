"""SQLite storage backend — the default, zero-config backend."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..context import (
    ActionRecord,
    GroupRecord,
    TraceSession,
    TransitionRecord,
)

DEFAULT_DB_PATH = Path.home() / ".lattice" / "traces.db"

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


class SQLiteBackend:
    """Stores traces in a local SQLite database.

    Args:
        db_path: Path to the ``.db`` file.  The directory is created
            automatically on first use.  Defaults to
            ``~/.lattice/traces.db``.

    Example::

        import lattice
        lattice.configure(db_path="/tmp/my_traces.db")

        # or equivalently:
        from lattice.backends import SQLiteBackend
        lattice.configure(backend=SQLiteBackend("/tmp/my_traces.db"))
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._local = threading.local()

    # ── connection management ──────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating the DB if needed."""
        conn = getattr(self._local, "conn", None)
        path = getattr(self._local, "path", None)
        if conn is not None and path == self._db_path:
            return conn

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_TRACES_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()
        self._local.conn = conn
        self._local.path = self._db_path
        return conn

    # ── StorageBackend interface ───────────────────────────────────────

    def save_session(self, session: TraceSession) -> None:
        """Persist a completed trace session."""
        conn = self._get_connection()
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

    def load_sessions(
        self,
        *,
        workflow: str | None = None,
        last: int | None = None,
        trace_id: str | None = None,
    ) -> list[TraceSession]:
        """Return stored sessions, most recent first."""
        conn = self._get_connection()
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


# ── helpers ───────────────────────────────────────────────────────────


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
