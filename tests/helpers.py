"""Shared test helpers for agent-trace tests."""

from agent_trace.context import StepRecord


def make_step(
    name,
    index,
    score=None,
    error=None,
    latency_ms=100.0,
    parent_span_id=None,
    description="",
    criteria="",
    input_data="",
    output_data="",
):
    """Factory for creating StepRecord instances in tests."""
    return StepRecord(
        span_id=f"span-{index}",
        name=name,
        step_type="agent",
        description=description,
        criteria=criteria,
        input_data=input_data,
        output_data=output_data,
        step_index=index,
        latency_ms=latency_ms,
        parent_span_id=parent_span_id,
        score=score,
        score_explanation=f"Explanation for {name}" if score else None,
        error=error,
    )
