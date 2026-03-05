"""Tests for TraceSession.to_dict / to_json and StepRecord.to_dict."""

import json

from agent_trace import trace_agent, trace_session, trace_tool
from agent_trace.context import StepRecord, TraceSession


def _step(name, index, score=None, error=None, latency_ms=100.0, parent_span_id=None):
    return StepRecord(
        span_id=f"span-{index}",
        name=name,
        step_type="agent",
        description=f"desc-{name}",
        criteria=f"criteria-{name}",
        input_data='{"x": 1}',
        output_data='"result"',
        step_index=index,
        latency_ms=latency_ms,
        parent_span_id=parent_span_id,
        score=score,
        score_explanation=f"Explanation for {name}" if score else None,
        error=error,
    )


# -- StepRecord.to_dict --


def test_step_to_dict_keys():
    step = _step("a", 0, score=4.0)
    d = step.to_dict()
    expected_keys = {
        "span_id", "name", "step_type", "description", "criteria",
        "input", "output", "step_index", "latency_ms",
        "parent_span_id", "score", "score_explanation", "error",
    }
    assert set(d.keys()) == expected_keys


def test_step_to_dict_values():
    step = _step("researcher", 0, score=4.5, latency_ms=123.456789)
    d = step.to_dict()
    assert d["span_id"] == "span-0"
    assert d["name"] == "researcher"
    assert d["step_type"] == "agent"
    assert d["description"] == "desc-researcher"
    assert d["criteria"] == "criteria-researcher"
    assert d["input"] == '{"x": 1}'
    assert d["output"] == '"result"'
    assert d["step_index"] == 0
    assert d["latency_ms"] == 123.457  # rounded to 3 decimal places
    assert d["parent_span_id"] is None
    assert d["score"] == 4.5
    assert d["score_explanation"] == "Explanation for researcher"
    assert d["error"] is None


def test_step_to_dict_with_error():
    step = _step("broken", 1, error="something failed")
    d = step.to_dict()
    assert d["error"] == "something failed"
    assert d["score"] is None
    assert d["score_explanation"] is None


def test_step_to_dict_uses_input_output_keys():
    """Verify the dict uses 'input'/'output' not 'input_data'/'output_data'."""
    step = _step("x", 0)
    d = step.to_dict()
    assert "input" in d
    assert "output" in d
    assert "input_data" not in d
    assert "output_data" not in d


# -- TraceSession.to_dict (flat) --


def test_session_to_dict_flat_empty():
    session = TraceSession(trace_id="test-trace-id")
    d = session.to_dict()
    assert d == {"trace_id": "test-trace-id", "steps": []}


def test_session_to_dict_flat_ordering():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("a", 0))
    session.add_step(_step("b", 1))
    session.add_step(_step("c", 2))
    d = session.to_dict()
    assert [s["name"] for s in d["steps"]] == ["a", "b", "c"]


def test_session_to_dict_flat_no_children_key():
    """Flat mode should not include a 'children' key on steps."""
    session = TraceSession(trace_id="t1")
    session.add_step(_step("a", 0))
    d = session.to_dict(tree=False)
    assert "children" not in d["steps"][0]


# -- TraceSession.to_dict (tree) --


def test_session_to_dict_tree_nesting():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("outer", 0))
    session.add_step(_step("inner", 1, parent_span_id="span-0"))
    d = session.to_dict(tree=True)

    # Only the root should be at the top level
    assert len(d["steps"]) == 1
    root = d["steps"][0]
    assert root["name"] == "outer"
    assert len(root["children"]) == 1
    assert root["children"][0]["name"] == "inner"


def test_session_to_dict_tree_deep_nesting():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("level0", 0))
    session.add_step(_step("level1", 1, parent_span_id="span-0"))
    session.add_step(_step("level2", 2, parent_span_id="span-1"))
    d = session.to_dict(tree=True)

    assert len(d["steps"]) == 1
    level0 = d["steps"][0]
    assert level0["name"] == "level0"
    level1 = level0["children"][0]
    assert level1["name"] == "level1"
    level2 = level1["children"][0]
    assert level2["name"] == "level2"
    assert level2["children"] == []


def test_session_to_dict_tree_multiple_roots():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("root1", 0))
    session.add_step(_step("root2", 1))
    session.add_step(_step("child_of_root1", 2, parent_span_id="span-0"))
    d = session.to_dict(tree=True)

    assert len(d["steps"]) == 2
    assert d["steps"][0]["name"] == "root1"
    assert d["steps"][1]["name"] == "root2"
    assert len(d["steps"][0]["children"]) == 1
    assert d["steps"][1]["children"] == []


def test_session_to_dict_tree_multiple_children():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("parent", 0))
    session.add_step(_step("child_a", 1, parent_span_id="span-0"))
    session.add_step(_step("child_b", 2, parent_span_id="span-0"))
    d = session.to_dict(tree=True)

    assert len(d["steps"]) == 1
    parent = d["steps"][0]
    assert len(parent["children"]) == 2
    assert parent["children"][0]["name"] == "child_a"
    assert parent["children"][1]["name"] == "child_b"


def test_session_to_dict_tree_orphan_becomes_root():
    """If parent_span_id references a span not in the session, treat as root."""
    session = TraceSession(trace_id="t1")
    session.add_step(_step("orphan", 0, parent_span_id="nonexistent-span"))
    d = session.to_dict(tree=True)

    assert len(d["steps"]) == 1
    assert d["steps"][0]["name"] == "orphan"


# -- TraceSession.to_json --


def test_session_to_json_is_valid_json():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("a", 0, score=3.5))
    raw = session.to_json()
    parsed = json.loads(raw)
    assert parsed["trace_id"] == "t1"
    assert len(parsed["steps"]) == 1


def test_session_to_json_compact():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("a", 0))
    raw = session.to_json(indent=None)
    assert "\n" not in raw
    json.loads(raw)  # still valid JSON


def test_session_to_json_tree_mode():
    session = TraceSession(trace_id="t1")
    session.add_step(_step("outer", 0))
    session.add_step(_step("inner", 1, parent_span_id="span-0"))
    raw = session.to_json(tree=True)
    parsed = json.loads(raw)
    assert len(parsed["steps"]) == 1
    assert parsed["steps"][0]["children"][0]["name"] == "inner"


# -- Integration with decorators --


def test_to_json_with_traced_decorators():
    @trace_tool(name="multiply", criteria="correct product")
    def multiply(a: int, b: int) -> int:
        return a * b

    @trace_agent(name="calculator", criteria="correct result")
    def calculator(x: int) -> int:
        return multiply(x, 2) + 1

    with trace_session(trace_id="integration-test") as session:
        result = calculator(5)

    assert result == 11

    # Flat mode
    d = session.to_dict()
    assert d["trace_id"] == "integration-test"
    assert len(d["steps"]) == 2

    # Tree mode
    d_tree = session.to_dict(tree=True)
    assert len(d_tree["steps"]) == 1  # calculator is the root
    root = d_tree["steps"][0]
    assert root["name"] == "calculator"
    assert len(root["children"]) == 1
    assert root["children"][0]["name"] == "multiply"

    # JSON round-trip
    raw = session.to_json(tree=True)
    parsed = json.loads(raw)
    assert parsed == d_tree


def test_to_json_with_error_step():
    @trace_agent(name="failer", criteria="should not fail")
    def failer() -> str:
        raise RuntimeError("boom")

    with trace_session(trace_id="error-test") as session:
        try:
            failer()
        except RuntimeError:
            pass

    d = session.to_dict()
    assert len(d["steps"]) == 1
    assert d["steps"][0]["error"] == "boom"

    raw = session.to_json()
    parsed = json.loads(raw)
    assert parsed["steps"][0]["error"] == "boom"
