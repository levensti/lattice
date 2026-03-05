from agent_trace.context import TraceSession
from agent_trace.bottleneck import find_bottlenecks

from .helpers import make_step as _step


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
