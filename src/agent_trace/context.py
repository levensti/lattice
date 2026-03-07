from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    """A single traced step within a session."""

    span_id: str
    name: str
    step_type: str  # "agent" or "tool" (kept for backwards compat)
    description: str
    criteria: str
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
    steps: list[StepRecord] = field(default_factory=list)
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
def trace_session(trace_id: str | None = None):
    """Context manager that groups decorated calls into a single trace."""
    session = TraceSession(trace_id=trace_id or uuid.uuid4().hex)
    session_token = _current_session.set(session)
    span_token = _current_span_id.set(None)
    try:
        yield session
    finally:
        _current_session.reset(session_token)
        _current_span_id.reset(span_token)


def get_current_session() -> TraceSession | None:
    return _current_session.get()
