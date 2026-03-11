"""Tests for topology primitives: trace_loop, trace_parallel, trace_transition, trace_activation, trace_iterations."""

import asyncio

import pytest

from lattice import action, trace_session
from lattice.context import (
    trace_activation,
    trace_iterations,
    trace_loop,
    trace_parallel,
    trace_transition,
)


# ── trace_loop ──────────────────────────────────────────────────────────


def test_loop_groups_steps_by_iteration():
    @action(goal="do work")
    def work(x: int) -> int:
        return x + 1

    with trace_session(goal="test") as session:
        with trace_loop("my_loop") as loop:
            for _ in range(3):
                with loop.iteration():
                    work(1)

    assert len(session.actions) == 3
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "loop"
    assert session.groups[0].name == "my_loop"
    assert loop.iteration_count == 3

    group_id = session.groups[0].group_id
    for i, s in enumerate(session.actions):
        assert s.group_id == group_id
        assert s.iteration == i


def test_loop_iteration_yields_number():
    @action(goal="no-op")
    def noop() -> None:
        pass

    with trace_session(goal="test") as session:
        with trace_loop("lp") as loop:
            collected = []
            for _ in range(2):
                with loop.iteration() as n:
                    collected.append(n)
                    noop()

    assert collected == [0, 1]


def test_loop_steps_outside_iteration_have_no_iteration():
    @action(goal="inside work")
    def inside() -> None:
        pass

    @action(goal="outside work")
    def outside() -> None:
        pass

    with trace_session(goal="test") as session:
        with trace_loop("lp") as loop:
            outside()
            with loop.iteration():
                inside()

    outside_action = next(s for s in session.actions if s.name == "outside")
    inside_action = next(s for s in session.actions if s.name == "inside")
    assert outside_action.iteration is None
    assert inside_action.iteration == 0
    assert outside_action.group_id is not None
    assert inside_action.group_id == outside_action.group_id


# ── trace_iterations ──────────────────────────────────────────────────


def test_trace_iterations_basic():
    @action(goal="work")
    def work(x: int) -> int:
        return x + 1

    with trace_session(goal="test") as session:
        for i in trace_iterations("my_loop", range(3)):
            work(i)

    assert len(session.actions) == 3
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "loop"
    assert session.groups[0].name == "my_loop"
    for i, s in enumerate(session.actions):
        assert s.iteration == i


def test_trace_iterations_early_break():
    @action(goal="do work")
    def work() -> None:
        pass

    with trace_session(goal="test") as session:
        iters = trace_iterations("breakable", range(10))
        for i in iters:
            work()
            if i == 2:
                break

    assert iters.iteration_count == 3
    assert len(session.actions) == 3


def test_trace_iterations_yields_original_values():
    collected = []

    with trace_session(goal="test"):
        for item in trace_iterations("custom", ["a", "b", "c"]):
            collected.append(item)

    assert collected == ["a", "b", "c"]


def test_trace_iterations_iteration_count():
    iters = trace_iterations("counted", range(5))
    with trace_session(goal="test"):
        for _ in iters:
            pass
    assert iters.iteration_count == 5


# ── trace_parallel ──────────────────────────────────────────────────────


def test_parallel_marks_concurrent_steps():
    @action(goal="branch a")
    async def branch_a() -> str:
        return "a"

    @action(goal="branch b")
    async def branch_b() -> str:
        return "b"

    async def _run():
        with trace_session(goal="test") as session:
            with trace_parallel("fanout"):
                await asyncio.gather(branch_a(), branch_b())
        return session

    session = asyncio.run(_run())
    assert len(session.actions) == 2
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "parallel"
    assert session.groups[0].name == "fanout"

    group_id = session.groups[0].group_id
    for s in session.actions:
        assert s.group_id == group_id


def test_parallel_sync_steps():
    @action(goal="task a")
    def task_a() -> str:
        return "a"

    @action(goal="task b")
    def task_b() -> str:
        return "b"

    with trace_session(goal="test") as session:
        with trace_parallel("batch"):
            task_a()
            task_b()

    assert len(session.groups) == 1
    group_id = session.groups[0].group_id
    assert all(s.group_id == group_id for s in session.actions)


# ── trace_transition ───────────────────────────────────────────────────


def test_transition_recorded():
    @action(goal="route request")
    def router(choice: str) -> str:
        trace_transition(to=choice, reason=f"chose {choice}")
        return choice

    with trace_session(goal="test") as session:
        router("agent_a")
        router("agent_b")

    manual = [t for t in session.transitions if not t.auto]
    assert len(manual) == 2
    assert manual[0].to_name == "agent_a"
    assert manual[0].reason == "chose agent_a"
    assert manual[1].to_name == "agent_b"


def test_transition_captures_from_span():
    @action(action_id="src-span", goal="emit transition")
    def source() -> str:
        trace_transition(to="target", reason="go")
        return "done"

    with trace_session(goal="test") as session:
        source()

    manual = [t for t in session.transitions if not t.auto]
    assert manual[0].from_span_id == "src-span"


# ── trace_activation ──────────────────────────────────────────────────


def test_activation_reason_recorded():
    @action(goal="respond to event")
    def listener() -> str:
        return "handled"

    with trace_session(goal="test") as session:
        with trace_activation(reason="state X changed"):
            listener()

    assert session.actions[0].activation_reason == "state X changed"


def test_activation_reason_clears_after_context():
    @action(goal="handle activation")
    def activated() -> str:
        return "yes"

    @action(goal="normal work")
    def normal() -> str:
        return "no"

    with trace_session(goal="test") as session:
        with trace_activation(reason="event fired"):
            activated()
        normal()

    assert session.actions[0].activation_reason == "event fired"
    assert session.actions[1].activation_reason is None


# ── role parameter ────────────────────────────────────────────────────


def test_role_recorded_on_step():
    @action(goal="generate", role="generator")
    def gen() -> str:
        return "output"

    @action(goal="evaluate", role="evaluator")
    def eval_fn(text: str) -> dict:
        return {"pass": True}

    with trace_session(goal="test") as session:
        text = gen()
        eval_fn(text)

    assert session.actions[0].role == "generator"
    assert session.actions[1].role == "evaluator"


def test_role_defaults_to_none():
    @action(goal="plain work")
    def plain() -> str:
        return "x"

    with trace_session(goal="test") as session:
        plain()

    assert session.actions[0].role is None


# ── nesting ───────────────────────────────────────────────────────────


def test_loop_inside_step_inherits_parent():
    @action(goal="inner work")
    def inner() -> str:
        return "done"

    @action(goal="outer work")
    def outer() -> str:
        with trace_loop("nested") as loop:
            with loop.iteration():
                inner()
        return "done"

    with trace_session(goal="test") as session:
        outer()

    inner_action = next(s for s in session.actions if s.name == "inner")
    outer_action = next(s for s in session.actions if s.name == "outer")
    assert inner_action.parent_span_id == outer_action.span_id
    assert inner_action.group_id is not None
    assert inner_action.iteration == 0
