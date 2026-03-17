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
    # All scores identical → no outlier to flag
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.0))
    session.add_action(_action("b", 1, score=4.0))
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
    convergence = [r for r in results if r.impact == "loop_no_convergence"]
    assert len(convergence) == 1
    assert "No improvement" in convergence[0].explanation
    assert "react" in convergence[0].action_name


def test_loop_convergence_not_flagged_when_improving():
    session = TraceSession()
    gid = "loop-2"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="optimize"))
    session.add_action(_action("gen", 0, score=2.0, group_id=gid, iteration=0))
    session.add_action(_action("gen", 1, score=3.0, group_id=gid, iteration=1))
    session.add_action(_action("gen", 2, score=4.5, group_id=gid, iteration=2))

    convergence = [r for r in find_bottlenecks(session) if r.impact == "loop_no_convergence"]
    assert convergence == []


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
    weakest = [r for r in results if r.impact == "weakest_branch"]
    assert len(weakest) == 1
    assert "cache" in weakest[0].action_name
    assert "search" in weakest[0].action_name


def test_parallel_balanced_branches_not_flagged():
    session = TraceSession()
    gid = "par-2"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="balanced"))
    session.add_action(_action("a", 0, score=4.0, group_id=gid))
    session.add_action(_action("b", 1, score=3.5, group_id=gid))
    session.add_action(_action("c", 2, score=4.0, group_id=gid))

    weakest = [r for r in find_bottlenecks(session) if r.impact == "weakest_branch"]
    assert weakest == []


def test_parallel_result_carries_span_ids():
    session = TraceSession()
    gid = "par-3"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="search"))
    session.add_action(_action("web", 0, score=4.0, group_id=gid, span_id="s0"))
    session.add_action(_action("cache", 1, score=1.0, group_id=gid, span_id="s1", parent_span_id="s0"))

    results = find_bottlenecks(session)
    weakest = [r for r in results if r.impact == "weakest_branch"]
    assert weakest[0].span_id == "s1"
    assert weakest[0].parent_span_id == "s0"


# ── lowest-scoring action ────────────────────────────────────────────


def test_lowest_score_flagged():
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.0))
    session.add_action(_action("b", 1, score=4.5))
    session.add_action(_action("c", 2, score=1.5))

    results = find_bottlenecks(session)
    lowest = [r for r in results if r.impact == "lowest_score"]
    assert len(lowest) == 1
    assert lowest[0].action_name == "c"
    assert "Lowest-scoring" in lowest[0].explanation


def test_lowest_score_not_flagged_when_all_equal():
    session = TraceSession()
    session.add_action(_action("a", 0, score=3.0))
    session.add_action(_action("b", 1, score=3.0))

    results = find_bottlenecks(session)
    lowest = [r for r in results if r.impact == "lowest_score"]
    assert len(lowest) == 0


def test_lowest_score_not_flagged_with_single_action():
    session = TraceSession()
    session.add_action(_action("a", 0, score=2.0))

    results = find_bottlenecks(session)
    lowest = [r for r in results if r.impact == "lowest_score"]
    assert len(lowest) == 0


# ── session.bottlenecks property ─────────────────────────────────────


# ── quality cascade ──────────────────────────────────────────────────


def test_cascade_flagged_when_parent_degrades_child():
    session = TraceSession()
    session.add_action(_action("step_a", 0, score=4.5, span_id="a"))
    session.add_action(_action("step_b", 1, score=1.0, span_id="b", parent_span_id="a"))
    session.add_action(_action("step_c", 2, score=1.0, span_id="c", parent_span_id="b"))

    results = find_bottlenecks(session)
    cascades = [r for r in results if r.impact == "quality_cascade"]
    assert len(cascades) >= 1
    assert cascades[0].action_name == "step_b"
    assert "cascaded" in cascades[0].explanation


def test_cascade_not_flagged_when_all_scores_high():
    session = TraceSession()
    session.add_action(_action("step_a", 0, score=4.0, span_id="a"))
    session.add_action(_action("step_b", 1, score=4.0, span_id="b", parent_span_id="a"))

    results = find_bottlenecks(session)
    cascades = [r for r in results if r.impact == "quality_cascade"]
    assert cascades == []


# ── session.bottlenecks property ─────────────────────────────────────


def test_session_bottlenecks_property():
    session = TraceSession()
    session.add_action(_action("ok", 0, score=5.0))
    session.add_action(_action("broken", 1, error="crash"))

    results = session.bottlenecks
    assert len(results) >= 1
    assert any(r.impact == "error" for r in results)
