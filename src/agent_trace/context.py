from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("agent_trace")


@dataclass
class StepRecord:
    """A single traced step within a session."""

    span_id: str
    name: str
    description: str
    goal: str
    input_data: str
    output_data: str
    step_index: int
    latency_ms: float
    parent_span_id: str | None = None
    score: float | None = None
    score_explanation: str | None = None
    error: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TraceSession:
    """Groups multiple traced steps into a single debugging session."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workflow_name: str = ""
    goal: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    session_score: float | None = field(default=None)
    session_score_explanation: str | None = field(default=None)
    _step_counter: int = field(default=0, repr=False)

    def next_index(self) -> int:
        idx = self._step_counter
        self._step_counter += 1
        return idx

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)


_current_session: ContextVar[TraceSession | None] = ContextVar(
    "agent_trace_session", default=None
)
_current_span_id: ContextVar[str | None] = ContextVar(
    "agent_trace_span_id", default=None
)


@contextmanager
def trace_session(
    trace_id: str | None = None,
    *,
    workflow_name: str = "",
    goal: str = "",
):
    """Context manager that groups decorated calls into a single trace.

    Args:
        trace_id: Custom trace ID (auto-generated if omitted).
        workflow_name: Human-readable name for this workflow.
        goal: The desired outcome — used for auto-judging results.
    """
    session = TraceSession(
        trace_id=trace_id or uuid.uuid4().hex,
        workflow_name=workflow_name,
        goal=goal,
    )
    logger.info("Trace session started: %s (trace_id=%s)", workflow_name or "(unnamed)", session.trace_id)
    session_token = _current_session.set(session)
    span_token = _current_span_id.set(None)
    try:
        yield session
    finally:
        _current_session.reset(session_token)
        _current_span_id.reset(span_token)
        logger.info(
            "Trace session ended (trace_id=%s, steps=%d)",
            session.trace_id,
            len(session.steps),
        )


def get_current_session() -> TraceSession | None:
    return _current_session.get()
