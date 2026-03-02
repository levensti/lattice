import asyncio

import pytest

from agent_trace import trace_agent, trace_tool, trace_session


def test_sync_agent_recorded():
    @trace_agent(name="greeter", criteria="Must greet by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session() as session:
        result = greet("Alice")

    assert result == "Hello, Alice!"
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.name == "greeter"
    assert step.step_type == "agent"
    assert '"Alice"' in step.input_data
    assert "Hello, Alice!" in step.output_data
    assert step.latency_ms >= 0
    assert step.error is None


def test_sync_tool_recorded():
    @trace_tool(name="adder", criteria="Must return correct sum")
    def add(a: int, b: int) -> int:
        return a + b

    with trace_session() as session:
        result = add(2, 3)

    assert result == 5
    assert len(session.steps) == 1
    assert session.steps[0].step_type == "tool"


def test_nested_spans():
    @trace_tool(name="inner", criteria="inner criteria")
    def inner_tool(x: int) -> int:
        return x * 2

    @trace_agent(name="outer", criteria="outer criteria")
    def outer_agent(x: int) -> int:
        return inner_tool(x) + 1

    with trace_session() as session:
        result = outer_agent(5)

    assert result == 11
    assert len(session.steps) == 2
    inner_step = next(s for s in session.steps if s.name == "inner")
    outer_step = next(s for s in session.steps if s.name == "outer")
    assert inner_step.parent_span_id == outer_step.span_id


def test_error_captured():
    @trace_agent(name="failer", criteria="Should not fail")
    def failing_agent() -> str:
        raise ValueError("something went wrong")

    with trace_session() as session:
        with pytest.raises(ValueError, match="something went wrong"):
            failing_agent()

    assert len(session.steps) == 1
    assert session.steps[0].error == "something went wrong"


def test_no_session_passthrough():
    """Decorated functions work fine outside a trace_session."""

    @trace_agent(name="standalone", criteria="criteria")
    def standalone(x: int) -> int:
        return x + 1

    assert standalone(10) == 11


def test_step_ordering():
    @trace_agent(name="step_a", criteria="a")
    def step_a() -> str:
        return "a"

    @trace_agent(name="step_b", criteria="b")
    def step_b() -> str:
        return "b"

    @trace_agent(name="step_c", criteria="c")
    def step_c() -> str:
        return "c"

    with trace_session() as session:
        step_a()
        step_b()
        step_c()

    assert [s.name for s in session.steps] == ["step_a", "step_b", "step_c"]
    assert [s.step_index for s in session.steps] == [0, 1, 2]


def test_async_agent_recorded():
    @trace_agent(name="async_greeter", criteria="Must greet")
    async def greet(name: str) -> str:
        return f"Hi, {name}!"

    async def _run():
        with trace_session() as session:
            result = await greet("Bob")
        return result, session

    result, session = asyncio.run(_run())
    assert result == "Hi, Bob!"
    assert len(session.steps) == 1
    assert session.steps[0].name == "async_greeter"
