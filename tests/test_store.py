"""Tests for the local SQLite store and query API."""

import json

import pytest

from lattice import action, trace_session, traces
from lattice.context import TraceSession
from lattice.backends.sqlite import DEFAULT_DB_PATH
from lattice.store import configure, save_session, traces as store_traces


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Point the store at a temp directory for every test."""
    db = tmp_path / "test_traces.db"
    configure(db_path=db)
    yield
    configure(db_path=DEFAULT_DB_PATH)


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
    from lattice.context import ActionRecord

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


def test_configure_changes_db_path(tmp_path):
    custom = tmp_path / "custom" / "traces.db"
    configure(db_path=custom)

    save_session(TraceSession(trace_id="custom1", workflow_name="w", goal="g"))
    assert custom.exists()

    results = store_traces()
    assert len(results) == 1
