import pytest

from agent_trace.context import StepRecord, TraceSession
from agent_trace.bottleneck import (
    SessionComparison,
    StepComparison,
    compare_sessions,
    find_bottlenecks,
)


def _step(name, index, score=None, error=None, latency_ms=100.0):
    return StepRecord(
        span_id=f"span-{index}",
        name=name,
        step_type="agent",
        description="",
        criteria="",
        input_data="",
        output_data="",
        step_index=index,
        latency_ms=latency_ms,
        score=score,
        score_explanation=f"Explanation for {name}" if score else None,
        error=error,
    )


def test_bottlenecks_sorted_by_score():
    session = TraceSession()
    session.add_step(_step("good", 0, score=4.5))
    session.add_step(_step("bad", 1, score=1.5))
    session.add_step(_step("ok", 2, score=3.0))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].step_name == "bad"
    assert bottlenecks[-1].step_name == "good"


def test_errors_ranked_first():
    session = TraceSession()
    session.add_step(_step("good", 0, score=5.0))
    session.add_step(_step("broken", 1, error="crash"))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].step_name == "broken"
    assert bottlenecks[0].impact == "error"


def test_empty_session():
    session = TraceSession()
    assert find_bottlenecks(session) == []


def test_largest_drop_detected():
    session = TraceSession()
    session.add_step(_step("great", 0, score=5.0))
    session.add_step(_step("terrible", 1, score=1.0))
    session.add_step(_step("decent", 2, score=3.5))

    bottlenecks = find_bottlenecks(session)
    terrible = next(b for b in bottlenecks if b.step_name == "terrible")
    assert terrible.impact == "lowest_score"


def test_ties_broken_by_latency():
    session = TraceSession()
    session.add_step(_step("fast", 0, score=2.0, latency_ms=50.0))
    session.add_step(_step("slow", 1, score=2.0, latency_ms=500.0))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].step_name == "slow"
    assert bottlenecks[1].step_name == "fast"


# ---------------------------------------------------------------------------
# compare_sessions tests
# ---------------------------------------------------------------------------


def _make_session(steps, trace_id=None):
    """Build a TraceSession from a list of (name, index, score, latency_ms) tuples."""
    session = TraceSession(trace_id=trace_id or "test")
    for name, index, score, latency_ms in steps:
        session.add_step(_step(name, index, score=score, latency_ms=latency_ms))
    return session


def test_compare_requires_two_sessions():
    with pytest.raises(ValueError, match="at least two sessions"):
        compare_sessions([TraceSession()])


def test_compare_detects_score_regression():
    s1 = _make_session([
        ("planner", 0, 4.5, 100),
        ("writer", 1, 4.0, 200),
    ], trace_id="run-1")
    s2 = _make_session([
        ("planner", 0, 4.5, 100),
        ("writer", 1, 2.0, 200),
    ], trace_id="run-2")

    result = compare_sessions([s1, s2])

    assert result.session_ids == ["run-1", "run-2"]

    writer = next(c for c in result.step_comparisons if c.step_name == "writer")
    assert writer.score_trend == "regressed"
    assert writer.score_delta == pytest.approx(-2.0)
    assert writer in result.regressions

    planner = next(c for c in result.step_comparisons if c.step_name == "planner")
    assert planner.score_trend == "stable"
    assert planner not in result.regressions


def test_compare_detects_score_improvement():
    s1 = _make_session([("step_a", 0, 2.0, 100)], trace_id="r1")
    s2 = _make_session([("step_a", 0, 4.5, 100)], trace_id="r2")

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.score_trend == "improved"
    assert comp.score_delta == pytest.approx(2.5)
    assert result.regressions == []


def test_compare_detects_latency_regression():
    s1 = _make_session([("fetch", 0, 4.0, 100)], trace_id="r1")
    s2 = _make_session([("fetch", 0, 4.0, 500)], trace_id="r2")

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.latency_trend == "regressed"
    assert comp.latency_delta_ms == pytest.approx(400.0)


def test_compare_detects_latency_improvement():
    s1 = _make_session([("fetch", 0, 4.0, 500)], trace_id="r1")
    s2 = _make_session([("fetch", 0, 4.0, 100)], trace_id="r2")

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.latency_trend == "improved"
    assert comp.latency_delta_ms == pytest.approx(-400.0)


def test_compare_stable_latency():
    s1 = _make_session([("fetch", 0, 4.0, 100)], trace_id="r1")
    s2 = _make_session([("fetch", 0, 4.0, 105)], trace_id="r2")

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.latency_trend == "stable"


def test_compare_recurring_bottleneck():
    # "bad_step" is the worst step in every session -> recurring bottleneck
    s1 = _make_session([
        ("good_step", 0, 4.5, 100),
        ("bad_step", 1, 1.0, 100),
    ], trace_id="r1")
    s2 = _make_session([
        ("good_step", 0, 4.5, 100),
        ("bad_step", 1, 1.2, 100),
    ], trace_id="r2")
    s3 = _make_session([
        ("good_step", 0, 4.5, 100),
        ("bad_step", 1, 1.1, 100),
    ], trace_id="r3")

    result = compare_sessions([s1, s2, s3])
    bad = next(c for c in result.step_comparisons if c.step_name == "bad_step")
    assert bad.recurring_bottleneck is True
    assert bad in result.recurring_bottlenecks

    good = next(c for c in result.step_comparisons if c.step_name == "good_step")
    assert good.recurring_bottleneck is False


def test_compare_step_missing_from_session():
    s1 = _make_session([
        ("planner", 0, 4.0, 100),
        ("writer", 1, 3.0, 200),
    ], trace_id="r1")
    # Session 2 drops the "writer" step entirely
    s2 = _make_session([
        ("planner", 0, 4.5, 100),
    ], trace_id="r2")

    result = compare_sessions([s1, s2])
    writer = next(c for c in result.step_comparisons if c.step_name == "writer")
    # writer only has one real score, so trend is no_data
    assert writer.scores == [3.0, None]
    assert writer.score_trend == "no_data"


def test_compare_with_error_steps():
    s1 = _make_session([("step_a", 0, 4.0, 100)], trace_id="r1")
    s2 = TraceSession(trace_id="r2")
    s2.add_step(_step("step_a", 0, error="timeout"))

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.error_count == 1
    # First score is 4.0, second is None (error) -> only one valid score
    assert comp.scores == [4.0, None]
    assert comp.score_trend == "no_data"


def test_compare_unscored_steps():
    s1 = _make_session([("step_a", 0, None, 100)], trace_id="r1")
    s2 = _make_session([("step_a", 0, None, 200)], trace_id="r2")

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.scores == [None, None]
    assert comp.score_trend == "no_data"
    assert comp.latency_trend == "regressed"


def test_compare_three_sessions_trend():
    s1 = _make_session([("step_a", 0, 4.0, 100)], trace_id="r1")
    s2 = _make_session([("step_a", 0, 3.0, 200)], trace_id="r2")
    s3 = _make_session([("step_a", 0, 2.0, 300)], trace_id="r3")

    result = compare_sessions([s1, s2, s3])
    comp = result.step_comparisons[0]
    assert comp.score_trend == "regressed"
    assert comp.score_delta == pytest.approx(-2.0)
    assert comp.latency_trend == "regressed"
    assert comp.latency_delta_ms == pytest.approx(200.0)
    assert comp.scores == [4.0, 3.0, 2.0]
    assert comp.latencies_ms == [100.0, 200.0, 300.0]


def test_compare_new_step_in_later_session():
    s1 = _make_session([("step_a", 0, 4.0, 100)], trace_id="r1")
    s2 = _make_session([
        ("step_a", 0, 4.0, 100),
        ("step_b", 1, 3.0, 200),
    ], trace_id="r2")

    result = compare_sessions([s1, s2])
    step_b = next(c for c in result.step_comparisons if c.step_name == "step_b")
    # step_b only appears in session 2, so score list has None for session 1
    assert step_b.scores == [None, 3.0]
    assert step_b.score_trend == "no_data"


def test_compare_session_ids_preserved():
    s1 = TraceSession(trace_id="alpha")
    s1.add_step(_step("x", 0, score=3.0))
    s2 = TraceSession(trace_id="beta")
    s2.add_step(_step("x", 0, score=3.0))

    result = compare_sessions([s1, s2])
    assert result.session_ids == ["alpha", "beta"]


def test_compare_error_in_every_session():
    s1 = TraceSession(trace_id="r1")
    s1.add_step(_step("flaky", 0, error="crash"))
    s2 = TraceSession(trace_id="r2")
    s2.add_step(_step("flaky", 0, error="timeout"))

    result = compare_sessions([s1, s2])
    comp = result.step_comparisons[0]
    assert comp.error_count == 2
    assert comp.scores == [None, None]
    assert comp.score_trend == "no_data"
    assert comp.recurring_bottleneck is True
