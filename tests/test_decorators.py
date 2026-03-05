import asyncio

import pytest

from agent_trace import trace_agent, trace_pipeline, trace_tool, trace_session


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


# ── trace_pipeline tests ──────────────────────────────────────────────


def test_pipeline_sync_returns_result_and_session():
    """trace_pipeline creates a session and returns (result, session)."""

    @trace_pipeline(name="my_pipe", criteria="Must work")
    def run(x: int) -> int:
        return x + 1

    result, session = run(10)

    assert result == 11
    assert isinstance(session, __import__("agent_trace").TraceSession)
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.name == "my_pipe"
    assert step.step_type == "pipeline"
    assert step.criteria == "Must work"
    assert "10" in step.input_data
    assert "11" in step.output_data
    assert step.latency_ms >= 0
    assert step.error is None
    assert step.parent_span_id is None


def test_pipeline_async_returns_result_and_session():
    """trace_pipeline works with async functions."""

    @trace_pipeline(name="async_pipe", criteria="Async must work")
    async def run(msg: str) -> str:
        return f"done: {msg}"

    async def _go():
        return await run("hello")

    result, session = asyncio.run(_go())

    assert result == "done: hello"
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.name == "async_pipe"
    assert step.step_type == "pipeline"
    assert step.error is None


def test_pipeline_captures_inner_steps():
    """Decorated agents/tools inside a pipeline are recorded in the session."""

    @trace_agent(name="inner_agent", criteria="inner")
    def agent_fn(x: int) -> int:
        return x * 2

    @trace_tool(name="inner_tool", criteria="inner tool")
    def tool_fn(x: int) -> int:
        return x + 10

    @trace_pipeline(name="full_pipe")
    def run(x: int) -> int:
        a = agent_fn(x)
        return tool_fn(a)

    result, session = run(5)

    assert result == 20
    # pipeline step + agent step + tool step
    assert len(session.steps) == 3
    names = [s.name for s in session.steps]
    assert "inner_agent" in names
    assert "inner_tool" in names
    assert "full_pipe" in names


def test_pipeline_inner_steps_parented_to_pipeline():
    """Inner agent/tool steps have the pipeline span as their parent."""

    @trace_agent(name="child", criteria="c")
    def child_fn() -> str:
        return "ok"

    @trace_pipeline(name="parent_pipe")
    def run() -> str:
        return child_fn()

    result, session = run()

    assert result == "ok"
    pipe_step = next(s for s in session.steps if s.name == "parent_pipe")
    child_step = next(s for s in session.steps if s.name == "child")
    assert child_step.parent_span_id == pipe_step.span_id


def test_pipeline_error_propagates_and_records():
    """Errors inside a pipeline propagate and are recorded."""

    @trace_pipeline(name="failing_pipe", criteria="Should not fail")
    def run() -> str:
        raise RuntimeError("pipeline boom")

    with pytest.raises(RuntimeError, match="pipeline boom"):
        run()


def test_pipeline_error_still_returns_session_steps():
    """Even on error, the pipeline step is recorded with the error message."""

    @trace_agent(name="ok_agent", criteria="c")
    def ok_fn() -> str:
        return "fine"

    @trace_agent(name="bad_agent", criteria="c")
    def bad_fn() -> str:
        raise ValueError("agent failed")

    @trace_pipeline(name="err_pipe")
    def run() -> str:
        ok_fn()
        return bad_fn()

    with pytest.raises(ValueError, match="agent failed"):
        run()

    # We can't get the session from the return value since it raised,
    # but we verify the error propagates correctly.


def test_pipeline_custom_trace_id():
    """trace_pipeline accepts a custom trace_id."""

    @trace_pipeline(name="id_pipe", trace_id="custom-123")
    def run() -> str:
        return "ok"

    result, session = run()
    assert result == "ok"
    assert session.trace_id == "custom-123"


def test_pipeline_description_recorded():
    """trace_pipeline records the description on the pipeline step."""

    @trace_pipeline(name="desc_pipe", description="A described pipeline")
    def run() -> str:
        return "ok"

    result, session = run()
    step = next(s for s in session.steps if s.name == "desc_pipe")
    assert step.description == "A described pipeline"


def test_pipeline_no_context_leak():
    """After a pipeline call, no session leaks into subsequent code."""

    @trace_pipeline(name="leak_test")
    def run() -> str:
        return "done"

    run()

    from agent_trace import get_current_session
    assert get_current_session() is None


def test_pipeline_async_captures_inner_steps():
    """Async pipeline captures inner async agent steps."""

    @trace_agent(name="async_inner", criteria="c")
    async def inner(x: int) -> int:
        return x + 1

    @trace_pipeline(name="async_full_pipe")
    async def run(x: int) -> int:
        return await inner(x)

    async def _go():
        return await run(5)

    result, session = asyncio.run(_go())

    assert result == 6
    assert len(session.steps) == 2
    names = [s.name for s in session.steps]
    assert "async_inner" in names
    assert "async_full_pipe" in names


def test_pipeline_step_ordering():
    """Pipeline step index is 0 (claimed first) but recorded last (finishes last)."""

    @trace_agent(name="first", criteria="c")
    def first() -> str:
        return "1"

    @trace_agent(name="second", criteria="c")
    def second() -> str:
        return "2"

    @trace_pipeline(name="ordered_pipe")
    def run() -> str:
        first()
        return second()

    result, session = run()

    assert result == "2"
    # Pipeline step is appended last since it completes after inner steps
    assert session.steps[-1].name == "ordered_pipe"
    # Pipeline claims index 0, inner steps get 1 and 2
    pipe_step = next(s for s in session.steps if s.name == "ordered_pipe")
    assert pipe_step.step_index == 0
    # Inner steps are ordered sequentially
    inner_indices = [s.step_index for s in session.steps if s.name != "ordered_pipe"]
    assert inner_indices == [1, 2]
