import asyncio

import pytest

from agent_trace import step, trace_agent, trace_tool, trace_session


# ── @step decorator tests ──────────────────────────────────────────────


def test_sync_step_recorded():
    @step(name="greeter", criteria="Must greet by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session() as session:
        result = greet("Alice")

    assert result == "Hello, Alice!"
    assert len(session.steps) == 1
    s = session.steps[0]
    assert s.name == "greeter"
    assert s.step_type == "step"
    assert '"Alice"' in s.input_data
    assert "Hello, Alice!" in s.output_data
    assert s.latency_ms >= 0
    assert s.error is None


def test_async_step_recorded():
    @step(name="async_greeter", criteria="Must greet")
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
    assert session.steps[0].step_type == "step"


def test_step_with_tags():
    @step(name="search", criteria="Return results", tags=["io", "external"])
    def search(query: str) -> list:
        return [{"title": "result"}]

    with trace_session() as session:
        search("python")

    assert session.steps[0].tags == ["io", "external"]


def test_step_nested_spans():
    @step(name="inner", criteria="inner criteria")
    def inner_fn(x: int) -> int:
        return x * 2

    @step(name="outer", criteria="outer criteria")
    def outer_fn(x: int) -> int:
        return inner_fn(x) + 1

    with trace_session() as session:
        result = outer_fn(5)

    assert result == 11
    assert len(session.steps) == 2
    inner_step = next(s for s in session.steps if s.name == "inner")
    outer_step = next(s for s in session.steps if s.name == "outer")
    assert inner_step.parent_span_id == outer_step.span_id


def test_step_error_captured():
    @step(name="failer", criteria="Should not fail")
    def failing() -> str:
        raise ValueError("something went wrong")

    with trace_session() as session:
        with pytest.raises(ValueError, match="something went wrong"):
            failing()

    assert len(session.steps) == 1
    assert session.steps[0].error == "something went wrong"


def test_step_no_session_passthrough():
    @step(name="standalone", criteria="criteria")
    def standalone(x: int) -> int:
        return x + 1

    assert standalone(10) == 11


def test_step_ordering():
    @step(name="step_a", criteria="a")
    def step_a() -> str:
        return "a"

    @step(name="step_b", criteria="b")
    def step_b() -> str:
        return "b"

    @step(name="step_c", criteria="c")
    def step_c() -> str:
        return "c"

    with trace_session() as session:
        step_a()
        step_b()
        step_c()

    assert [s.name for s in session.steps] == ["step_a", "step_b", "step_c"]
    assert [s.step_index for s in session.steps] == [0, 1, 2]


# ── Backwards-compatible @trace_agent / @trace_tool ────────────────────


def test_trace_agent_still_works():
    @trace_agent(name="greeter", criteria="Must greet by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session() as session:
        result = greet("Alice")

    assert result == "Hello, Alice!"
    assert len(session.steps) == 1
    assert session.steps[0].step_type == "agent"


def test_trace_tool_still_works():
    @trace_tool(name="adder", criteria="Must return correct sum")
    def add(a: int, b: int) -> int:
        return a + b

    with trace_session() as session:
        result = add(2, 3)

    assert result == 5
    assert len(session.steps) == 1
    assert session.steps[0].step_type == "tool"


def test_trace_agent_with_tags():
    @trace_agent(name="planner", criteria="criteria", tags=["llm"])
    def planner(goal: str) -> str:
        return "plan"

    with trace_session() as session:
        planner("goal")

    assert session.steps[0].tags == ["llm"]


def test_nested_legacy_decorators():
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
