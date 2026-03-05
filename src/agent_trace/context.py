from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """A single traced step (agent call or tool call) within a session."""

    span_id: str
    name: str
    step_type: str  # "agent" or "tool"
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of this step."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "step_type": self.step_type,
            "description": self.description,
            "criteria": self.criteria,
            "input": self.input_data,
            "output": self.output_data,
            "step_index": self.step_index,
            "latency_ms": round(self.latency_ms, 3),
            "parent_span_id": self.parent_span_id,
            "score": self.score,
            "score_explanation": self.score_explanation,
            "error": self.error,
        }


def _build_tree(steps: list[StepRecord]) -> list[dict[str, Any]]:
    """Arrange flat steps into a nested tree based on parent_span_id."""
    nodes: dict[str, dict[str, Any]] = {}
    for step in steps:
        node = step.to_dict()
        node["children"] = []
        nodes[step.span_id] = node

    roots: list[dict[str, Any]] = []
    for step in steps:
        node = nodes[step.span_id]
        parent_id = step.parent_span_id
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)

    return roots


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

    def to_dict(self, *, tree: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the entire trace.

        Args:
            tree: If True, nest child steps under their parent's
                ``"children"`` key.  If False (default), return a flat
                ordered list.
        """
        if tree:
            steps_data = _build_tree(self.steps)
        else:
            steps_data = [step.to_dict() for step in self.steps]
        return {
            "trace_id": self.trace_id,
            "steps": steps_data,
        }

    def to_json(self, *, tree: bool = False, indent: int | None = 2) -> str:
        """Serialize the trace to a JSON string.

        Args:
            tree: If True, produce a nested tree structure.
            indent: JSON indentation level.  Pass ``None`` for compact
                output.
        """
        return json.dumps(self.to_dict(tree=tree), indent=indent, ensure_ascii=False)


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
