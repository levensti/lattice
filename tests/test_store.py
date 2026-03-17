"""Tests for the local SQLite store and query API."""

import json
import sqlite3

import pytest

from lattice import action, trace_session, traces
from lattice.context import (
    ActionRecord,
    GroupRecord,
    JudgeResult,
    TraceSession,
    TransitionRecord,
)
from lattice.storage.store import configure, save_session, traces as store_traces


# ── save + query ──────────────────────────────────────────────────────


def test_save_and_query_session():
    session = TraceSession(trace_id="abc123", workflow_name="test_wf", goal="do stuff")
    save_session(session)

    results = store_traces()
    assert len(results) == 1
    assert results[0].trace_id == "abc123"
    assert results[0].workflow_name == "test_wf"
    assert results[0].goal == "do stuff"


def test_query_by_workflow():
    save_session(TraceSession(trace_id="a", workflow_name="alpha", goal="g1"))
    save_session(TraceSession(trace_id="b", workflow_name="beta", goal="g2"))
    save_session(TraceSession(trace_id="c", workflow_name="alpha", goal="g3"))

    results = store_traces(workflow="alpha")
    assert len(results) == 2
    assert all(r.workflow_name == "alpha" for r in results)


def test_query_by_trace_id():
    save_session(TraceSession(trace_id="x1", workflow_name="w", goal="g"))
    save_session(TraceSession(trace_id="x2", workflow_name="w", goal="g"))

    results = store_traces(trace_id="x1")
    assert len(results) == 1
    assert results[0].trace_id == "x1"


def test_query_last_n():
    for i in range(5):
        save_session(TraceSession(trace_id=f"t{i}", workflow_name="w", goal="g"))

    results = store_traces(last=2)
    assert len(results) == 2


def test_query_empty_db():
    results = store_traces()
    assert results == []


def test_session_with_actions_roundtrips():
    session = TraceSession(trace_id="rt1", workflow_name="roundtrip", goal="test")
    session.add_action(ActionRecord(
        span_id="s1", name="step1", description="desc", goal="goal",
        input_data="in", output_data="out", action_index=0, latency_ms=42.5,
        score=4.0, score_explanation="good",
    ))
    save_session(session)

    results = store_traces(trace_id="rt1")
    assert len(results) == 1
    loaded = results[0]
    assert len(loaded.actions) == 1
    assert loaded.actions[0].name == "step1"
    assert loaded.actions[0].latency_ms == 42.5
    assert loaded.actions[0].score == 4.0
    assert loaded.actions[0].score_explanation == "good"


def test_session_score_roundtrips():
    session = TraceSession(
        trace_id="scored", workflow_name="w", goal="g",
        session_score=3.5, session_score_explanation="decent",
    )
    save_session(session)

    loaded = store_traces(trace_id="scored")[0]
    assert loaded.session_score == 3.5
    assert loaded.session_score_explanation == "decent"


# ── auto-persist via trace_session ────────────────────────────────────


def test_trace_session_auto_persists():
    @action(goal="say hi")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session(goal="auto persist test", workflow_name="auto") as session:
        greet("World")

    # Should be in the store without any explicit save
    results = store_traces(trace_id=session.trace_id)
    assert len(results) == 1
    assert results[0].workflow_name == "auto"
    assert len(results[0].actions) == 1


def test_trace_session_persist_false():
    with trace_session(goal="no persist", persist=False) as session:
        pass

    results = store_traces(trace_id=session.trace_id)
    assert len(results) == 0


def test_upsert_on_duplicate_trace_id():
    session = TraceSession(trace_id="dup", workflow_name="v1", goal="g")
    save_session(session)

    session2 = TraceSession(trace_id="dup", workflow_name="v2", goal="g")
    save_session(session2)

    results = store_traces(trace_id="dup")
    assert len(results) == 1
    assert results[0].workflow_name == "v2"


# ── configure ─────────────────────────────────────────────────────────


def test_judge_result_roundtrip():
    session = TraceSession(trace_id="jr1", workflow_name="judge", goal="test judges")
    session.add_action(ActionRecord(
        span_id="s1", name="step1", description="desc", goal="goal",
        input_data="in", output_data="out", action_index=0, latency_ms=10.0,
        score=4.0, score_explanation="good",
        judge_result=JudgeResult(score=4.5, explanation="accurate", name="accuracy", model="gpt-4"),
    ))
    save_session(session)

    loaded = store_traces(trace_id="jr1")[0]
    jr = loaded.actions[0].judge_result
    assert isinstance(jr, JudgeResult)
    assert jr.score == 4.5
    assert jr.name == "accuracy"
    assert jr.model == "gpt-4"


def test_judge_config_roundtrip():
    session = TraceSession(trace_id="jc1", workflow_name="judge_cfg", goal="test config")
    session.add_action(ActionRecord(
        span_id="s1", name="step1", description="desc", goal="goal",
        input_data="in", output_data="out", action_index=0, latency_ms=10.0,
        judge_config={
            "system_prompt": "Rate 1-5",
            "provider": "OpenAIProvider",
            "model": "gpt-4o",
            "has_custom_prompt_builder": False,
            "temperature": 0.1,
            "top_p": 0.9,
            "timeout": 30.0,
            "api_type": "chat_completions",
        },
    ))
    save_session(session)

    loaded = store_traces(trace_id="jc1")[0]
    cfg = loaded.actions[0].judge_config
    assert cfg is not None
    assert cfg["system_prompt"] == "Rate 1-5"
    assert cfg["provider"] == "OpenAIProvider"
    assert cfg["model"] == "gpt-4o"
    assert cfg["has_custom_prompt_builder"] is False
    assert cfg["temperature"] == 0.1
    assert cfg["top_p"] == 0.9
    assert cfg["api_type"] == "chat_completions"


def test_created_at_roundtrip():
    session = TraceSession(trace_id="ca1", workflow_name="ts", goal="test created_at")
    save_session(session)

    loaded = store_traces(trace_id="ca1")[0]
    assert loaded.created_at is not None
    assert isinstance(loaded.created_at, str)
    assert "T" in loaded.created_at  # ISO format


def test_configure_changes_db_path(tmp_path):
    custom = tmp_path / "custom" / "traces.db"
    configure(db_path=custom)

    save_session(TraceSession(trace_id="custom1", workflow_name="w", goal="g"))
    assert custom.exists()

    results = store_traces()
    assert len(results) == 1


# ── normalized schema roundtrips ─────────────────────────────────────


def test_groups_roundtrip():
    session = TraceSession(trace_id="grp1", workflow_name="w", goal="g")
    session.add_group(GroupRecord(group_id="g1", group_type="loop", name="react_loop"))
    session.add_group(GroupRecord(group_id="g2", group_type="parallel", name="fanout", parent_span_id="s1"))
    save_session(session)

    loaded = store_traces(trace_id="grp1")[0]
    assert len(loaded.groups) == 2
    types = {g.group_type for g in loaded.groups}
    assert types == {"loop", "parallel"}
    par = next(g for g in loaded.groups if g.group_type == "parallel")
    assert par.parent_span_id == "s1"


def test_transitions_roundtrip():
    session = TraceSession(trace_id="tr1", workflow_name="w", goal="g")
    session.add_transition(TransitionRecord(from_span_id=None, to_name="step1", reason="start"))
    session.add_transition(TransitionRecord(from_span_id="s1", to_name="step2", reason="done", auto=True))
    save_session(session)

    loaded = store_traces(trace_id="tr1")[0]
    assert len(loaded.transitions) == 2
    assert loaded.transitions[0].to_name == "step1"
    assert loaded.transitions[0].from_span_id is None
    assert loaded.transitions[1].auto is True


def test_tags_roundtrip():
    session = TraceSession(trace_id="tag1", workflow_name="w", goal="g")
    session.add_action(ActionRecord(
        span_id="s1", name="step1", description="d", goal="g",
        input_data="in", output_data="out", action_index=0, latency_ms=10.0,
        tags=["important", "slow"],
    ))
    save_session(session)

    loaded = store_traces(trace_id="tag1")[0]
    assert set(loaded.actions[0].tags) == {"important", "slow"}


def test_parent_span_roundtrip():
    session = TraceSession(trace_id="tree1", workflow_name="w", goal="g")
    session.add_action(ActionRecord(
        span_id="root", name="parent_action", description="d", goal="g",
        input_data="", output_data="", action_index=0, latency_ms=10.0,
    ))
    session.add_action(ActionRecord(
        span_id="child1", name="child_action", description="d", goal="g",
        input_data="", output_data="", action_index=1, latency_ms=5.0,
        parent_span_id="root",
    ))
    save_session(session)

    loaded = store_traces(trace_id="tree1")[0]
    assert loaded.actions[0].parent_span_id is None
    assert loaded.actions[1].parent_span_id == "root"


def test_full_roundtrip():
    """Save a session with actions, groups, transitions, tags, and judge results — all fields should roundtrip."""
    session = TraceSession(
        trace_id="full1", workflow_name="full_test", goal="everything",
        session_score=4.2, session_score_explanation="solid run",
    )
    session.add_action(ActionRecord(
        span_id="s1", name="research", description="web search", goal="find info",
        input_data="query", output_data="results", action_index=0, latency_ms=150.0,
        score=4.0, score_explanation="thorough",
        tags=["web", "search"],
        role="researcher",
        group_id="g1",
        iteration=0,
        activation_reason="user request",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Rate 1-5"},
        judge_result=JudgeResult(score=4.0, explanation="good coverage", name="coverage", model="gpt-4o"),
    ))
    session.add_action(ActionRecord(
        span_id="s2", name="summarize", description="condense", goal="short summary",
        input_data="results", output_data="summary", action_index=1, latency_ms=50.0,
        parent_span_id="s1",
    ))
    session.add_group(GroupRecord(group_id="g1", group_type="loop", name="react"))
    session.add_transition(TransitionRecord(from_span_id="s1", to_name="summarize", reason="done searching"))
    save_session(session)

    loaded = store_traces(trace_id="full1")[0]
    assert loaded.session_score == 4.2
    assert loaded.session_score_explanation == "solid run"
    assert len(loaded.actions) == 2
    a0 = loaded.actions[0]
    assert a0.role == "researcher"
    assert a0.group_id == "g1"
    assert a0.iteration == 0
    assert a0.activation_reason == "user request"
    assert set(a0.tags) == {"web", "search"}
    assert a0.judge_result is not None
    assert a0.judge_result.score == 4.0
    assert a0.judge_config["model"] == "gpt-4o"
    a1 = loaded.actions[1]
    assert a1.parent_span_id == "s1"
    assert len(loaded.groups) == 1
    assert loaded.groups[0].group_type == "loop"
    assert len(loaded.transitions) == 1
    assert loaded.transitions[0].reason == "done searching"


# ── migration from v0 ────────────────────────────────────────────────


def test_migration_v0_to_v2(tmp_path):
    """Old-schema DB with JSON blob is auto-migrated to normalized tables."""
    db = tmp_path / "migrate.db"

    # Create old-schema DB manually
    conn = sqlite3.connect(str(db))
    conn.execute("""\
        CREATE TABLE traces (
            trace_id          TEXT PRIMARY KEY,
            workflow_name     TEXT NOT NULL DEFAULT '',
            goal              TEXT NOT NULL DEFAULT '',
            session_score     REAL,
            score_explanation TEXT,
            data              TEXT NOT NULL,
            created_at        TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_workflow ON traces (workflow_name)"
    )

    session_data = {
        "trace_id": "old1",
        "workflow_name": "legacy",
        "goal": "test migration",
        "actions": [
            {
                "span_id": "s1",
                "name": "step1",
                "description": "desc",
                "goal": "g",
                "input_data": "in",
                "output_data": "out",
                "action_index": 0,
                "latency_ms": 100.0,
                "parent_span_id": None,
                "score": 4.0,
                "score_explanation": "good",
                "error": None,
                "tags": ["important"],
                "role": "researcher",
                "group_id": None,
                "iteration": None,
                "activation_reason": None,
                "judge_config": {
                    "provider": "OpenAIProvider",
                    "model": "gpt-4o",
                    "system_prompt": "Rate 1-5",
                },
                "judge_results": [
                    {
                        "score": 4.5,
                        "explanation": "accurate",
                        "name": "accuracy",
                        "model": "gpt-4",
                    },
                ],
            },
            {
                "span_id": "s2",
                "name": "step2",
                "description": "child step",
                "goal": "sub-goal",
                "input_data": "in2",
                "output_data": "out2",
                "action_index": 1,
                "latency_ms": 50.0,
                "parent_span_id": "s1",
                "score": None,
                "score_explanation": None,
                "error": None,
                "tags": [],
                "role": None,
                "group_id": None,
                "iteration": None,
                "activation_reason": None,
                "judge_config": None,
                "judge_results": [],
            },
        ],
        "groups": [
            {
                "group_id": "g1",
                "group_type": "loop",
                "name": "react",
                "parent_span_id": None,
            }
        ],
        "transitions": [
            {
                "from_span_id": None,
                "to_name": "step1",
                "reason": "start",
                "auto": False,
            }
        ],
        "session_score": 4.0,
        "session_score_explanation": "good session",
        "created_at": "2025-01-01T00:00:00",
    }

    conn.execute(
        "INSERT INTO traces (trace_id, workflow_name, goal, session_score, "
        "score_explanation, data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "old1", "legacy", "test migration", 4.0,
            "good session", json.dumps(session_data), "2025-01-01T00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    # Open via SQLiteStore — should auto-migrate
    configure(db_path=db)

    results = store_traces(trace_id="old1")
    assert len(results) == 1
    s = results[0]
    assert s.workflow_name == "legacy"
    assert s.session_score == 4.0
    assert s.session_score_explanation == "good session"
    assert len(s.actions) == 2
    assert s.actions[0].name == "step1"
    assert s.actions[0].score == 4.0
    assert s.actions[0].tags == ["important"]
    assert s.actions[0].role == "researcher"
    assert s.actions[0].judge_result is not None
    assert s.actions[0].judge_result.score == 4.5
    assert s.actions[0].judge_config["provider"] == "OpenAIProvider"
    assert s.actions[1].parent_span_id == "s1"
    assert len(s.groups) == 1
    assert s.groups[0].group_type == "loop"
    assert len(s.transitions) == 1
    assert s.transitions[0].to_name == "step1"

    # Verify schema version is set
    conn = sqlite3.connect(str(db))
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == 2


def test_migration_preserves_data_across_restart(tmp_path):
    """After migration, data survives a new connection."""
    db = tmp_path / "migrate2.db"

    conn = sqlite3.connect(str(db))
    conn.execute("""\
        CREATE TABLE traces (
            trace_id TEXT PRIMARY KEY,
            workflow_name TEXT NOT NULL DEFAULT '',
            goal TEXT NOT NULL DEFAULT '',
            session_score REAL,
            score_explanation TEXT,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    data = {
        "actions": [],
        "groups": [],
        "transitions": [],
    }
    conn.execute(
        "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("t1", "wf", "goal", 3.0, "ok", json.dumps(data), "2025-06-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    # First open triggers migration
    configure(db_path=db)
    results = store_traces()
    assert len(results) == 1

    # Second open should use v2 schema directly (no migration)
    configure(db_path=db)
    results = store_traces()
    assert len(results) == 1
    assert results[0].trace_id == "t1"
