"""Tests for topology primitives: trace_loop, trace_parallel, trace_transition, trace_activation, trace_iterations."""

import asyncio

import pytest

from lattice import step, trace_session
from lattice.context import (
    trace_activation,
    trace_iterations,
    trace_loop,
    trace_parallel,
    trace_transition,
)


# ── trace_loop ──────────────────────────────────────────────────────────


def test_loop_groups_steps_by_iteration():
    @step(goal="do work")
    def work(x: int) -> int:
        return x + 1

    with trace_session() as session:
        with trace_loop("my_loop") as loop:
            for _ in range(3):
                with loop.iteration():
                    work(1)

    assert len(session.steps) == 3
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "loop"
    assert session.groups[0].name == "my_loop"
    assert loop.iteration_count == 3

    group_id = session.groups[0].group_id
    for i, s in enumerate(session.steps):
        assert s.group_id == group_id
        assert s.iteration == i


def test_loop_iteration_yields_number():
    @step
    def noop() -> None:
        pass

    with trace_session() as session:
        with trace_loop("lp") as loop:
            collected = []
            for _ in range(2):
                with loop.iteration() as n:
                    collected.append(n)
                    noop()

    assert collected == [0, 1]


def test_loop_steps_outside_iteration_have_no_iteration():
    @step
    def inside() -> None:
        pass

    @step
    def outside() -> None:
        pass

    with trace_session() as session:
        with trace_loop("lp") as loop:
            outside()
            with loop.iteration():
                inside()

    outside_step = next(s for s in session.steps if s.name == "outside")
    inside_step = next(s for s in session.steps if s.name == "inside")
    assert outside_step.iteration is None
    assert inside_step.iteration == 0
    assert outside_step.group_id is not None
    assert inside_step.group_id == outside_step.group_id


# ── trace_iterations ──────────────────────────────────────────────────


def test_trace_iterations_basic():
    @step(goal="work")
    def work(x: int) -> int:
        return x + 1

    with trace_session() as session:
        for i in trace_iterations("my_loop", range(3)):
            work(i)

    assert len(session.steps) == 3
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "loop"
    assert session.groups[0].name == "my_loop"
    for i, s in enumerate(session.steps):
        assert s.iteration == i


def test_trace_iterations_early_break():
    @step
    def work() -> None:
        pass

    with trace_session() as session:
        iters = trace_iterations("breakable", range(10))
        for i in iters:
            work()
            if i == 2:
                break

    assert iters.iteration_count == 3
    assert len(session.steps) == 3


def test_trace_iterations_yields_original_values():
    collected = []

    with trace_session():
        for item in trace_iterations("custom", ["a", "b", "c"]):
            collected.append(item)

    assert collected == ["a", "b", "c"]


def test_trace_iterations_iteration_count():
    iters = trace_iterations("counted", range(5))
    with trace_session():
        for _ in iters:
            pass
    assert iters.iteration_count == 5


# ── trace_parallel ──────────────────────────────────────────────────────


def test_parallel_marks_concurrent_steps():
    @step
    async def branch_a() -> str:
        return "a"

    @step
    async def branch_b() -> str:
        return "b"

    async def _run():
        with trace_session() as session:
            with trace_parallel("fanout"):
                await asyncio.gather(branch_a(), branch_b())
        return session

    session = asyncio.run(_run())
    assert len(session.steps) == 2
    assert len(session.groups) == 1
    assert session.groups[0].group_type == "parallel"
    assert session.groups[0].name == "fanout"

    group_id = session.groups[0].group_id
    for s in session.steps:
        assert s.group_id == group_id


def test_parallel_sync_steps():
    @step
    def task_a() -> str:
        return "a"

    @step
    def task_b() -> str:
        return "b"

    with trace_session() as session:
        with trace_parallel("batch"):
            task_a()
            task_b()

    assert len(session.groups) == 1
    group_id = session.groups[0].group_id
    assert all(s.group_id == group_id for s in session.steps)


# ── trace_transition ───────────────────────────────────────────────────


def test_transition_recorded():
    @step
    def router(choice: str) -> str:
        trace_transition(to=choice, reason=f"chose {choice}")
        return choice

    with trace_session() as session:
        router("agent_a")
        router("agent_b")

    manual = [t for t in session.transitions if not t.auto]
    assert len(manual) == 2
    assert manual[0].to_name == "agent_a"
    assert manual[0].reason == "chose agent_a"
    assert manual[1].to_name == "agent_b"


def test_transition_captures_from_span():
    @step(step_id="src-span")
    def source() -> str:
        trace_transition(to="target", reason="go")
        return "done"

    with trace_session() as session:
        source()

    manual = [t for t in session.transitions if not t.auto]
    assert manual[0].from_span_id == "src-span"


# ── trace_activation ──────────────────────────────────────────────────


def test_activation_reason_recorded():
    @step(goal="respond to event")
    def listener() -> str:
        return "handled"

    with trace_session() as session:
        with trace_activation(reason="state X changed"):
            listener()

    assert session.steps[0].activation_reason == "state X changed"


def test_activation_reason_clears_after_context():
    @step
    def activated() -> str:
        return "yes"

    @step
    def normal() -> str:
        return "no"

    with trace_session() as session:
        with trace_activation(reason="event fired"):
            activated()
        normal()

    assert session.steps[0].activation_reason == "event fired"
    assert session.steps[1].activation_reason is None


# ── role parameter ────────────────────────────────────────────────────


def test_role_recorded_on_step():
    @step(goal="generate", role="generator")
    def gen() -> str:
        return "output"

    @step(goal="evaluate", role="evaluator")
    def eval_fn(text: str) -> dict:
        return {"pass": True}

    with trace_session() as session:
        text = gen()
        eval_fn(text)

    assert session.steps[0].role == "generator"
    assert session.steps[1].role == "evaluator"


def test_role_defaults_to_none():
    @step
    def plain() -> str:
        return "x"

    with trace_session() as session:
        plain()

    assert session.steps[0].role is None


# ── nesting ───────────────────────────────────────────────────────────


def test_loop_inside_step_inherits_parent():
    @step
    def inner() -> str:
        return "done"

    @step
    def outer() -> str:
        with trace_loop("nested") as loop:
            with loop.iteration():
                inner()
        return "done"

    with trace_session() as session:
        outer()

    inner_step = next(s for s in session.steps if s.name == "inner")
    outer_step = next(s for s in session.steps if s.name == "outer")
    assert inner_step.parent_span_id == outer_step.span_id
    assert inner_step.group_id is not None
    assert inner_step.iteration == 0
