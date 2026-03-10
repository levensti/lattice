from lattice.context import GroupRecord, StepRecord, TraceSession
from lattice.bottleneck import find_bottlenecks


def _step(name, index, score=None, error=None, latency_ms=100.0,
          group_id=None, iteration=None):
    return StepRecord(
        span_id=f"span-{index}",
        name=name,
        description="",
        goal="",
        input_data="",
        output_data="",
        step_index=index,
        latency_ms=latency_ms,
        score=score,
        score_explanation=f"Explanation for {name}" if score else None,
        error=error,
        group_id=group_id,
        iteration=iteration,
    )


# ── existing pipeline bottleneck tests ────────────────────────────────


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


# ── loop convergence tests ────────────────────────────────────────────


def test_loop_no_convergence_flagged():
    session = TraceSession()
    gid = "loop-group-1"
    session.add_group(GroupRecord(
        group_id=gid, group_type="loop", name="react",
    ))
    session.add_step(_step("think", 0, score=3.0, group_id=gid, iteration=0))
    session.add_step(_step("think", 1, score=2.0, group_id=gid, iteration=1))
    session.add_step(_step("think", 2, score=2.0, group_id=gid, iteration=2))

    bottlenecks = find_bottlenecks(session)
    convergence = [b for b in bottlenecks if b.impact == "loop_no_convergence"]
    assert len(convergence) == 1
    assert "No improvement" in convergence[0].explanation
    assert "react" in convergence[0].step_name


def test_loop_convergence_not_flagged_when_improving():
    session = TraceSession()
    gid = "loop-group-2"
    session.add_group(GroupRecord(
        group_id=gid, group_type="loop", name="optimize",
    ))
    session.add_step(_step("gen", 0, score=2.0, group_id=gid, iteration=0))
    session.add_step(_step("gen", 1, score=3.0, group_id=gid, iteration=1))
    session.add_step(_step("gen", 2, score=4.5, group_id=gid, iteration=2))

    bottlenecks = find_bottlenecks(session)
    convergence = [b for b in bottlenecks if b.impact == "loop_no_convergence"]
    assert len(convergence) == 0


# ── parallel branch tests ────────────────────────────────────────────


def test_parallel_weakest_branch_flagged():
    session = TraceSession()
    gid = "par-group-1"
    session.add_group(GroupRecord(
        group_id=gid, group_type="parallel", name="search",
    ))
    session.add_step(_step("web", 0, score=4.0, group_id=gid))
    session.add_step(_step("db", 1, score=4.5, group_id=gid))
    session.add_step(_step("cache", 2, score=1.0, group_id=gid))

    bottlenecks = find_bottlenecks(session)
    branch = [b for b in bottlenecks if b.impact == "weakest_branch"]
    assert len(branch) == 1
    assert "cache" in branch[0].step_name
    assert "search" in branch[0].step_name


def test_parallel_balanced_branches_not_flagged():
    session = TraceSession()
    gid = "par-group-2"
    session.add_group(GroupRecord(
        group_id=gid, group_type="parallel", name="balanced",
    ))
    session.add_step(_step("a", 0, score=4.0, group_id=gid))
    session.add_step(_step("b", 1, score=3.5, group_id=gid))
    session.add_step(_step("c", 2, score=4.0, group_id=gid))

    bottlenecks = find_bottlenecks(session)
    branch = [b for b in bottlenecks if b.impact == "weakest_branch"]
    assert len(branch) == 0
