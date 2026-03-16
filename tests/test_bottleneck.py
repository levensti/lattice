from lattice.context import ActionRecord, GroupRecord, TraceSession
from lattice.bottleneck import find_bottlenecks


def _action(name, index, score=None, error=None, latency_ms=100.0,
            group_id=None, iteration=None, span_id=None, parent_span_id=None):
    return ActionRecord(
        span_id=span_id or f"span-{index}",
        name=name,
        description="",
        goal="",
        input_data="",
        output_data="",
        action_index=index,
        latency_ms=latency_ms,
        score=score,
        score_explanation=f"Explanation for {name}" if score else None,
        error=error,
        group_id=group_id,
        iteration=iteration,
        parent_span_id=parent_span_id,
    )


# ── errors ────────────────────────────────────────────────────────────


def test_errors_collected():
    session = TraceSession()
    session.add_action(_action("good", 0, score=5.0))
    session.add_action(_action("broken", 1, error="crash"))

    results = find_bottlenecks(session)
    assert len(results) == 1
    assert results[0].action_name == "broken"
    assert results[0].impact == "error"


def test_empty_session():
    session = TraceSession()
    assert find_bottlenecks(session) == []


def test_no_structural_issues_returns_empty():
    # Scored actions with no errors / loops / parallel groups → nothing to report
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.5))
    session.add_action(_action("b", 1, score=2.0))
    assert find_bottlenecks(session) == []


def test_result_carries_span_ids():
    session = TraceSession()
    session.add_action(_action("broken", 0, error="oops", span_id="s0", parent_span_id="p0"))

    results = find_bottlenecks(session)
    assert results[0].span_id == "s0"
    assert results[0].parent_span_id == "p0"


# ── loop convergence ──────────────────────────────────────────────────


def test_loop_no_convergence_flagged():
    session = TraceSession()
    gid = "loop-1"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="react"))
    session.add_action(_action("think", 0, score=3.0, group_id=gid, iteration=0))
    session.add_action(_action("think", 1, score=2.0, group_id=gid, iteration=1))
    session.add_action(_action("think", 2, score=2.0, group_id=gid, iteration=2))

    results = find_bottlenecks(session)
    assert len(results) == 1
    assert results[0].impact == "loop_no_convergence"
    assert "No improvement" in results[0].explanation
    assert "react" in results[0].action_name


def test_loop_convergence_not_flagged_when_improving():
    session = TraceSession()
    gid = "loop-2"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="optimize"))
    session.add_action(_action("gen", 0, score=2.0, group_id=gid, iteration=0))
    session.add_action(_action("gen", 1, score=3.0, group_id=gid, iteration=1))
    session.add_action(_action("gen", 2, score=4.5, group_id=gid, iteration=2))

    assert find_bottlenecks(session) == []


def test_loop_result_carries_span_ids():
    session = TraceSession()
    gid = "loop-3"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="react"))
    session.add_action(_action("think", 0, score=3.0, group_id=gid, iteration=0, span_id="s0"))
    session.add_action(_action("think", 1, score=2.0, group_id=gid, iteration=1, span_id="s1", parent_span_id="s0"))

    results = find_bottlenecks(session)
    assert results[0].span_id == "s1"
    assert results[0].parent_span_id == "s0"


# ── parallel branch imbalance ─────────────────────────────────────────


def test_parallel_weakest_branch_flagged():
    session = TraceSession()
    gid = "par-1"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="search"))
    session.add_action(_action("web", 0, score=4.0, group_id=gid))
    session.add_action(_action("db", 1, score=4.5, group_id=gid))
    session.add_action(_action("cache", 2, score=1.0, group_id=gid))

    results = find_bottlenecks(session)
    assert len(results) == 1
    assert results[0].impact == "weakest_branch"
    assert "cache" in results[0].action_name
    assert "search" in results[0].action_name


def test_parallel_balanced_branches_not_flagged():
    session = TraceSession()
    gid = "par-2"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="balanced"))
    session.add_action(_action("a", 0, score=4.0, group_id=gid))
    session.add_action(_action("b", 1, score=3.5, group_id=gid))
    session.add_action(_action("c", 2, score=4.0, group_id=gid))

    assert find_bottlenecks(session) == []


def test_parallel_result_carries_span_ids():
    session = TraceSession()
    gid = "par-3"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="search"))
    session.add_action(_action("web", 0, score=4.0, group_id=gid, span_id="s0"))
    session.add_action(_action("cache", 1, score=1.0, group_id=gid, span_id="s1", parent_span_id="s0"))

    results = find_bottlenecks(session)
    assert results[0].span_id == "s1"
    assert results[0].parent_span_id == "s0"
