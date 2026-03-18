"""SQLite storage backend — the default, zero-config backend."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..context import (
    ActionRecord,
    GroupRecord,
    JudgeResult,
    TraceSession,
    TransitionRecord,
    _judge_config_summary,
)
from .base import Store

DEFAULT_DB_PATH = Path.home() / ".lattice" / "traces.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS traces (
    trace_id                  TEXT PRIMARY KEY,
    workflow_name             TEXT NOT NULL DEFAULT '',
    goal                      TEXT NOT NULL DEFAULT '',
    input_data                TEXT,
    session_score             REAL,
    session_score_explanation TEXT,
    created_at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_workflow ON traces (workflow_name);
CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces (created_at);

CREATE TABLE IF NOT EXISTS actions (
    trace_id          TEXT NOT NULL REFERENCES traces ON DELETE CASCADE,
    span_id           TEXT NOT NULL,
    name              TEXT NOT NULL,
    actor_name        TEXT,
    actor_id          TEXT,
    goal              TEXT NOT NULL DEFAULT '',
    input_data        TEXT NOT NULL DEFAULT '',
    output_data       TEXT NOT NULL DEFAULT '',
    action_index      INTEGER NOT NULL,
    latency_ms        REAL NOT NULL,
    parent_span_id    TEXT,
    score             REAL,
    score_explanation TEXT,
    error             TEXT,
    group_id          TEXT,
    iteration         INTEGER,
    activation_reason TEXT,
    judge_config      TEXT,
    started_at        TEXT,
    ended_at          TEXT,
    judge_provider    TEXT,
    judge_model       TEXT,
    PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS idx_actions_trace ON actions (trace_id);
CREATE INDEX IF NOT EXISTS idx_actions_parent ON actions (trace_id, parent_span_id);
CREATE INDEX IF NOT EXISTS idx_actions_score ON actions (score);
CREATE INDEX IF NOT EXISTS idx_actions_judge_model ON actions (judge_model);
CREATE INDEX IF NOT EXISTS idx_actions_group ON actions (trace_id, group_id);
CREATE INDEX IF NOT EXISTS idx_actions_actor_id ON actions (actor_id);

CREATE TABLE IF NOT EXISTS judge_results (
    trace_id     TEXT NOT NULL,
    span_id      TEXT NOT NULL,
    result_index INTEGER NOT NULL,
    score        REAL NOT NULL,
    explanation  TEXT NOT NULL,
    name         TEXT NOT NULL,
    model        TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id, result_index),
    FOREIGN KEY (trace_id, span_id) REFERENCES actions ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_judge_results_model ON judge_results (model);

CREATE TABLE IF NOT EXISTS groups (
    trace_id       TEXT NOT NULL REFERENCES traces ON DELETE CASCADE,
    group_id       TEXT NOT NULL,
    group_type     TEXT NOT NULL CHECK (group_type IN ('loop', 'parallel')),
    name           TEXT NOT NULL,
    parent_span_id TEXT,
    PRIMARY KEY (trace_id, group_id)
);

CREATE TABLE IF NOT EXISTS transitions (
    trace_id         TEXT NOT NULL REFERENCES traces ON DELETE CASCADE,
    transition_index INTEGER NOT NULL,
    from_span_id     TEXT,
    to_name          TEXT NOT NULL,
    reason           TEXT NOT NULL DEFAULT '',
    auto             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (trace_id, transition_index)
);
"""


class SQLiteStore(Store):
    """Stores traces in a local SQLite database.

    Args:
        db_path: Path to the ``.db`` file.  The directory is created
            automatically on first use.  Defaults to
            ``~/.lattice/traces.db``.

    Example::

        import lattice
        lattice.configure(db_path="/tmp/my_traces.db")

        # or equivalently:
        from lattice.storage import SQLiteStore
        lattice.configure(backend=SQLiteStore("/tmp/my_traces.db"))
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

        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA foreign_keys=ON")
        self._local.conn = conn
        self._local.path = self._db_path
        return conn

    # ── Store interface ───────────────────────────────────────────────

    def save_session(self, session: TraceSession) -> None:
        """Persist a completed trace session."""
        conn = self._get_connection()
        if session.created_at is None:
            session.created_at = datetime.now(timezone.utc).isoformat()

        # Snapshot live judge configs before saving
        for action in session.actions:
            if action.judge is not None and action.judge_config is None:
                action.judge_config = _judge_config_summary(action.judge)

        # INSERT OR REPLACE cascades deletes via foreign keys
        conn.execute(
            "INSERT OR REPLACE INTO traces "
            "(trace_id, workflow_name, goal, input_data, session_score, "
            "session_score_explanation, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.trace_id,
                session.workflow_name,
                session.goal,
                session.input_data,
                session.session_score,
                session.session_score_explanation,
                session.created_at,
            ),
        )

        for a in session.actions:
            jc = a.judge_config
            jc_json = json.dumps(jc) if jc else None
            conn.execute(
                "INSERT INTO actions (trace_id, span_id, name, "
                "actor_name, actor_id, goal, input_data, output_data, action_index, latency_ms, "
                "parent_span_id, score, score_explanation, error, "
                "group_id, iteration, activation_reason, started_at, ended_at, "
                "judge_config, judge_provider, judge_model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.trace_id,
                    a.span_id,
                    a.name,
                    a.actor_name,
                    a.actor_id,
                    a.goal,
                    a.input_data,
                    a.output_data,
                    a.action_index,
                    a.latency_ms,
                    a.parent_span_id,
                    a.score,
                    a.score_explanation,
                    a.error,
                    a.group_id,
                    a.iteration,
                    a.activation_reason,
                    a.started_at,
                    a.ended_at,
                    jc_json,
                    jc.get("provider") if jc else None,
                    str(jc.get("model")) if jc and jc.get("model") is not None else None,
                ),
            )
            if a.judge_result:
                jr = a.judge_result
                conn.execute(
                    "INSERT INTO judge_results (trace_id, span_id, "
                    "result_index, score, explanation, name, model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session.trace_id, a.span_id, 0, jr.score, jr.explanation, jr.name, jr.model),
                )

        for g in session.groups:
            conn.execute(
                "INSERT INTO groups (trace_id, group_id, group_type, name, "
                "parent_span_id) VALUES (?, ?, ?, ?, ?)",
                (session.trace_id, g.group_id, g.group_type, g.name, g.parent_span_id),
            )

        for i, t in enumerate(session.transitions):
            conn.execute(
                "INSERT INTO transitions (trace_id, transition_index, "
                "from_span_id, to_name, reason, auto) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session.trace_id, i, t.from_span_id, t.to_name, t.reason, int(t.auto)),
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

        trace_rows = conn.execute(
            f"SELECT trace_id, workflow_name, goal, input_data, session_score, "
            f"session_score_explanation, created_at "
            f"FROM traces {where} ORDER BY created_at DESC {limit}",
            params,
        ).fetchall()

        if not trace_rows:
            return []

        trace_ids = [r[0] for r in trace_rows]
        ph = ",".join("?" * len(trace_ids))

        # Batch-fetch child data
        action_rows = conn.execute(
            f"SELECT trace_id, span_id, name, actor_name, actor_id, goal, "
            f"input_data, output_data, action_index, latency_ms, "
            f"parent_span_id, score, score_explanation, error, "
            f"group_id, iteration, activation_reason, started_at, ended_at, "
            f"judge_config "
            f"FROM actions WHERE trace_id IN ({ph}) ORDER BY action_index",
            trace_ids,
        ).fetchall()

        jr_rows = conn.execute(
            f"SELECT trace_id, span_id, score, explanation, name, model "
            f"FROM judge_results WHERE trace_id IN ({ph}) ORDER BY result_index",
            trace_ids,
        ).fetchall()

        group_rows = conn.execute(
            f"SELECT trace_id, group_id, group_type, name, parent_span_id "
            f"FROM groups WHERE trace_id IN ({ph})",
            trace_ids,
        ).fetchall()

        transition_rows = conn.execute(
            f"SELECT trace_id, from_span_id, to_name, reason, auto "
            f"FROM transitions WHERE trace_id IN ({ph}) ORDER BY transition_index",
            trace_ids,
        ).fetchall()

        jr_by_action: dict[tuple[str, str], JudgeResult] = {}
        for tid, sid, score, explanation, name, model in jr_rows:
            if (tid, sid) not in jr_by_action:
                jr_by_action[(tid, sid)] = JudgeResult(
                    score=score, explanation=explanation, name=name, model=model
                )

        actions_by_trace: dict[str, list[ActionRecord]] = defaultdict(list)
        for row in action_rows:
            tid, sid = row[0], row[1]
            jc_json = row[19]
            actions_by_trace[tid].append(
                ActionRecord(
                    span_id=sid,
                    name=row[2],
                    actor_name=row[3],
                    actor_id=row[4],
                    goal=row[5],
                    input_data=row[6],
                    output_data=row[7],
                    action_index=row[8],
                    latency_ms=row[9],
                    parent_span_id=row[10],
                    score=row[11],
                    score_explanation=row[12],
                    error=row[13],
                    group_id=row[14],
                    iteration=row[15],
                    activation_reason=row[16],
                    started_at=row[17],
                    ended_at=row[18],
                    judge_config=json.loads(jc_json) if jc_json else None,
                    judge_result=jr_by_action.get((tid, sid)),
                )
            )

        groups_by_trace: dict[str, list[GroupRecord]] = defaultdict(list)
        for tid, gid, gtype, name, psid in group_rows:
            groups_by_trace[tid].append(
                GroupRecord(group_id=gid, group_type=gtype, name=name, parent_span_id=psid)
            )

        transitions_by_trace: dict[str, list[TransitionRecord]] = defaultdict(list)
        for tid, from_sid, to_name, reason, auto_flag in transition_rows:
            transitions_by_trace[tid].append(
                TransitionRecord(
                    from_span_id=from_sid,
                    to_name=to_name,
                    reason=reason,
                    auto=bool(auto_flag),
                )
            )

        return [
            TraceSession(
                trace_id=tid,
                workflow_name=wf,
                goal=goal,
                input_data=input_data,
                actions=actions_by_trace.get(tid, []),
                groups=groups_by_trace.get(tid, []),
                transitions=transitions_by_trace.get(tid, []),
                session_score=score,
                session_score_explanation=expl,
                created_at=created_at,
            )
            for tid, wf, goal, input_data, score, expl, created_at in trace_rows
        ]
