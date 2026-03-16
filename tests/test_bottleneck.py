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


# ── basic sorting and errors ──────────────────────────────────────────


def test_bottlenecks_sorted_by_score():
    session = TraceSession()
    session.add_action(_action("good", 0, score=4.5))
    session.add_action(_action("bad", 1, score=1.5))
    session.add_action(_action("ok", 2, score=3.0))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].action_name == "bad"
    assert bottlenecks[-1].action_name == "good"


def test_errors_ranked_first():
    session = TraceSession()
    session.add_action(_action("good", 0, score=5.0))
    session.add_action(_action("broken", 1, error="crash"))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].action_name == "broken"
    assert bottlenecks[0].impact == "error"


def test_empty_session():
    session = TraceSession()
    assert find_bottlenecks(session) == []


def test_ties_broken_by_latency():
    session = TraceSession()
    session.add_action(_action("fast", 0, score=2.0, latency_ms=50.0))
    session.add_action(_action("slow", 1, score=2.0, latency_ms=500.0))

    bottlenecks = find_bottlenecks(session)
    assert bottlenecks[0].action_name == "slow"
    assert bottlenecks[1].action_name == "fast"


# ── span_id / parent_span_id on results ──────────────────────────────


def test_result_carries_span_ids():
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.0, span_id="s0"))
    session.add_action(_action("b", 1, score=2.0, span_id="s1", parent_span_id="s0"))

    bottlenecks = find_bottlenecks(session)
    b_result = next(r for r in bottlenecks if r.action_name == "b")
    assert b_result.span_id == "s1"
    assert b_result.parent_span_id == "s0"


# ── tree-aware impact classification ─────────────────────────────────


def test_root_cause_when_no_bad_ancestor():
    # a (good) -> b (bad): b has no bad ancestor, so it's a root_cause
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.5, span_id="s0"))
    session.add_action(_action("b", 1, score=1.0, span_id="s1", parent_span_id="s0"))

    bottlenecks = find_bottlenecks(session)
    b_result = next(r for r in bottlenecks if r.action_name == "b")
    assert b_result.impact == "root_cause"


def test_downstream_effect_when_parent_also_bad():
    # a (bad) -> b (bad): b is downstream of a's failure
    session = TraceSession()
    session.add_action(_action("a", 0, score=1.5, span_id="s0"))
    session.add_action(_action("b", 1, score=1.0, span_id="s1", parent_span_id="s0"))

    bottlenecks = find_bottlenecks(session)
    b_result = next(r for r in bottlenecks if r.action_name == "b")
    assert b_result.impact == "downstream_effect"


def test_downstream_effect_via_grandparent():
    # a (bad) -> b (ok) -> c (bad): c is downstream because grandparent a is below average
    session = TraceSession()
    session.add_action(_action("a", 0, score=1.0, span_id="s0"))
    session.add_action(_action("b", 1, score=3.5, span_id="s1", parent_span_id="s0"))
    session.add_action(_action("c", 2, score=1.5, span_id="s2", parent_span_id="s1"))

    bottlenecks = find_bottlenecks(session)
    c_result = next(r for r in bottlenecks if r.action_name == "c")
    assert c_result.impact == "downstream_effect"


def test_root_cause_with_no_parent():
    # top-level bad action with no parent is always root_cause
    session = TraceSession()
    session.add_action(_action("a", 0, score=4.5, span_id="s0"))
    session.add_action(_action("b", 1, score=1.0, span_id="s1"))  # no parent

    bottlenecks = find_bottlenecks(session)
    b_result = next(r for r in bottlenecks if r.action_name == "b")
    assert b_result.impact == "root_cause"


def test_deep_nested_agents():
    # orchestrator (bad) -> agent_a (bad) -> sub_a (bad)
    #                     -> agent_b (good) -> sub_b (good)
    session = TraceSession()
    session.add_action(_action("orchestrator", 0, score=1.5, span_id="s0"))
    session.add_action(_action("agent_a", 1, score=2.0, span_id="s1", parent_span_id="s0"))
    session.add_action(_action("sub_a", 2, score=1.0, span_id="s2", parent_span_id="s1"))
    session.add_action(_action("agent_b", 3, score=4.5, span_id="s3", parent_span_id="s0"))
    session.add_action(_action("sub_b", 4, score=4.0, span_id="s4", parent_span_id="s3"))

    bottlenecks = find_bottlenecks(session)
    by_name = {r.action_name: r for r in bottlenecks}

    assert by_name["orchestrator"].impact == "root_cause"
    assert by_name["agent_a"].impact == "downstream_effect"
    assert by_name["sub_a"].impact == "downstream_effect"
    assert by_name["agent_b"].impact == "root_cause"
    assert by_name["sub_b"].impact == "root_cause"


# ── loop convergence tests ────────────────────────────────────────────


def test_loop_no_convergence_flagged():
    session = TraceSession()
    gid = "loop-group-1"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="react"))
    session.add_action(_action("think", 0, score=3.0, group_id=gid, iteration=0))
    session.add_action(_action("think", 1, score=2.0, group_id=gid, iteration=1))
    session.add_action(_action("think", 2, score=2.0, group_id=gid, iteration=2))

    bottlenecks = find_bottlenecks(session)
    convergence = [b for b in bottlenecks if b.impact == "loop_no_convergence"]
    assert len(convergence) == 1
    assert "No improvement" in convergence[0].explanation
    assert "react" in convergence[0].action_name


def test_loop_convergence_not_flagged_when_improving():
    session = TraceSession()
    gid = "loop-group-2"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="optimize"))
    session.add_action(_action("gen", 0, score=2.0, group_id=gid, iteration=0))
    session.add_action(_action("gen", 1, score=3.0, group_id=gid, iteration=1))
    session.add_action(_action("gen", 2, score=4.5, group_id=gid, iteration=2))

    bottlenecks = find_bottlenecks(session)
    convergence = [b for b in bottlenecks if b.impact == "loop_no_convergence"]
    assert len(convergence) == 0


def test_loop_result_carries_span_ids():
    session = TraceSession()
    gid = "loop-group-3"
    session.add_group(GroupRecord(group_id=gid, group_type="loop", name="react"))
    session.add_action(_action("think", 0, score=3.0, group_id=gid, iteration=0, span_id="s0"))
    session.add_action(_action("think", 1, score=2.0, group_id=gid, iteration=1, span_id="s1", parent_span_id="s0"))

    bottlenecks = find_bottlenecks(session)
    convergence = next(b for b in bottlenecks if b.impact == "loop_no_convergence")
    assert convergence.span_id == "s1"
    assert convergence.parent_span_id == "s0"


# ── parallel branch tests ────────────────────────────────────────────


def test_parallel_weakest_branch_flagged():
    session = TraceSession()
    gid = "par-group-1"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="search"))
    session.add_action(_action("web", 0, score=4.0, group_id=gid))
    session.add_action(_action("db", 1, score=4.5, group_id=gid))
    session.add_action(_action("cache", 2, score=1.0, group_id=gid))

    bottlenecks = find_bottlenecks(session)
    branch = [b for b in bottlenecks if b.impact == "weakest_branch"]
    assert len(branch) == 1
    assert "cache" in branch[0].action_name
    assert "search" in branch[0].action_name


def test_parallel_balanced_branches_not_flagged():
    session = TraceSession()
    gid = "par-group-2"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="balanced"))
    session.add_action(_action("a", 0, score=4.0, group_id=gid))
    session.add_action(_action("b", 1, score=3.5, group_id=gid))
    session.add_action(_action("c", 2, score=4.0, group_id=gid))

    bottlenecks = find_bottlenecks(session)
    branch = [b for b in bottlenecks if b.impact == "weakest_branch"]
    assert len(branch) == 0


def test_parallel_result_carries_span_ids():
    session = TraceSession()
    gid = "par-group-3"
    session.add_group(GroupRecord(group_id=gid, group_type="parallel", name="search"))
    session.add_action(_action("web", 0, score=4.0, group_id=gid, span_id="s0"))
    session.add_action(_action("cache", 1, score=1.0, group_id=gid, span_id="s1", parent_span_id="s0"))

    bottlenecks = find_bottlenecks(session)
    branch = next(b for b in bottlenecks if b.impact == "weakest_branch")
    assert branch.span_id == "s1"
    assert branch.parent_span_id == "s0"
