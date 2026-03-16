from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GroupType = Literal["loop", "parallel"]

logger = logging.getLogger("lattice")


@dataclass
class ActionRecord:
    """A single traced action within a session."""

    span_id: str
    name: str
    description: str
    goal: str
    input_data: str
    output_data: str
    action_index: int
    latency_ms: float
    parent_span_id: str | None = None
    score: float | None = None
    score_explanation: str | None = None
    error: str | None = None
    tags: list[str] = field(default_factory=list)
    role: str | None = None
    group_id: str | None = None
    iteration: int | None = None
    activation_reason: str | None = None


@dataclass
class GroupRecord:
    """Metadata about a structural group of actions (loop or parallel)."""

    group_id: str
    group_type: GroupType
    name: str
    parent_span_id: str | None = None


@dataclass
class TransitionRecord:
    """A recorded routing decision between actions."""

    from_span_id: str | None
    to_name: str
    reason: str
    auto: bool = False


@dataclass
class TraceSession:
    """Groups multiple traced actions into a single debugging session."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workflow_name: str = ""
    goal: str = ""
    actions: list[ActionRecord] = field(default_factory=list)
    groups: list[GroupRecord] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    session_score: float | None = field(default=None)
    session_score_explanation: str | None = field(default=None)
    _action_counter: int = field(default=0, repr=False)

    def next_index(self) -> int:
        idx = self._action_counter
        self._action_counter += 1
        return idx

    def add_action(self, action: ActionRecord) -> None:
        self.actions.append(action)

    def add_group(self, group: GroupRecord) -> None:
        self.groups.append(group)

    def add_transition(self, transition: TransitionRecord) -> None:
        self.transitions.append(transition)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the session to a plain dict (suitable for JSON)."""
        d = asdict(self)
        d.pop("_action_counter", None)
        return d


_current_session: ContextVar[TraceSession | None] = ContextVar(
    "lattice_session", default=None
)
_current_span_id: ContextVar[str | None] = ContextVar(
    "lattice_span_id", default=None
)
_current_group_id: ContextVar[str | None] = ContextVar(
    "lattice_group_id", default=None
)
_current_iteration: ContextVar[int | None] = ContextVar(
    "lattice_iteration", default=None
)
_current_activation_reason: ContextVar[str | None] = ContextVar(
    "lattice_activation_reason", default=None
)


class LoopContext:
    """Handle returned by :func:`trace_loop` for managing iterations."""

    def __init__(self, name: str, group_id: str):
        self.name = name
        self.group_id = group_id
        self._iteration_count = 0

    @contextmanager
    def iteration(self):
        """Mark the boundary of one loop iteration.

        All ``@action`` calls inside this context are tagged with the
        current iteration number.
        """
        iter_num = self._iteration_count
        self._iteration_count += 1
        token = _current_iteration.set(iter_num)
        try:
            yield iter_num
        finally:
            _current_iteration.reset(token)

    @property
    def iteration_count(self) -> int:
        return self._iteration_count


class trace_iterations:
    """Wrap an iterable with loop tracing, avoiding nested context managers.

    Equivalent to ``trace_loop`` + ``loop.iteration()`` in a single
    ``for`` statement::

        for i in trace_iterations("react", range(5)):
            thought = think(state)
            action = act(thought)
            state = observe(action)

    Access :attr:`iteration_count` after the loop finishes to see how
    many iterations actually ran (useful with early ``break``).
    """

    def __init__(self, name: str, iterable):
        self.name = name
        self._iterable = iterable
        self._iteration_count = 0
        self._group_id: str | None = None

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def __iter__(self):
        session = _current_session.get()
        group_id = uuid.uuid4().hex
        self._group_id = group_id
        parent_span = _current_span_id.get()

        if session:
            session.add_group(GroupRecord(
                group_id=group_id,
                group_type="loop",
                name=self.name,
                parent_span_id=parent_span,
            ))

        group_token = _current_group_id.set(group_id)
        try:
            for item in self._iterable:
                iter_token = _current_iteration.set(self._iteration_count)
                self._iteration_count += 1
                try:
                    yield item
                finally:
                    _current_iteration.reset(iter_token)
        finally:
            _current_group_id.reset(group_token)


@contextmanager
def trace_loop(name: str):
    """Group actions into a named loop with numbered iterations.

    Usage::

        with trace_loop("react") as loop:
            while not done:
                with loop.iteration():
                    thought = think(state)
                    action = act(thought)
                    state = observe(action)
    """
    session = _current_session.get()
    group_id = uuid.uuid4().hex
    parent_span = _current_span_id.get()

    if session:
        session.add_group(GroupRecord(
            group_id=group_id,
            group_type="loop",
            name=name,
            parent_span_id=parent_span,
        ))

    token = _current_group_id.set(group_id)
    loop_ctx = LoopContext(name, group_id)
    try:
        yield loop_ctx
    finally:
        _current_group_id.reset(token)


@contextmanager
def trace_parallel(name: str):
    """Mark actions as running concurrently within this context.

    Works with ``asyncio.gather`` (context is automatically copied to tasks)
    and with threads when combined with :func:`copy_trace_context`.

    Usage::

        with trace_parallel("search_fanout"):
            results = await asyncio.gather(
                search_web(q), search_db(q), search_cache(q)
            )
    """
    session = _current_session.get()
    group_id = uuid.uuid4().hex
    parent_span = _current_span_id.get()

    if session:
        session.add_group(GroupRecord(
            group_id=group_id,
            group_type="parallel",
            name=name,
            parent_span_id=parent_span,
        ))

    token = _current_group_id.set(group_id)
    try:
        yield group_id
    finally:
        _current_group_id.reset(token)


def trace_transition(*, to: str, reason: str = "") -> None:
    """Record a routing decision from the current action.

    Call inside a ``@action``-decorated function to capture why control
    is being transferred to a particular next action.

    Usage::

        @action(name="router", ...)
        def router(state):
            if state.needs_retry:
                trace_transition(to="retry", reason="validation failed")
                return retry(state)
            trace_transition(to="finalize", reason="checks passed")
            return finalize(state)
    """
    session = _current_session.get()
    if session:
        from_span = _current_span_id.get()
        session.add_transition(TransitionRecord(
            from_span_id=from_span,
            to_name=to,
            reason=reason,
        ))


@contextmanager
def trace_activation(*, reason: str):
    """Provide an activation reason for actions without a parent caller.

    Used in blackboard / event-driven architectures where agents
    self-activate based on shared state changes.

    Usage::

        with trace_activation(reason="knowledge_base updated by agent_a"):
            agent_b.process(blackboard)
    """
    token = _current_activation_reason.set(reason)
    try:
        yield
    finally:
        _current_activation_reason.reset(token)


@contextmanager
def trace_session(
    trace_id: str | None = None,
    *,
    workflow_name: str = "",
    goal: str,
    persist: bool = True,
):
    """Context manager that groups ``@action``-decorated calls into a single trace.

    Args:
        trace_id: Custom trace ID (auto-generated if omitted).
        workflow_name: Human-readable name for this workflow.
        goal: **Required.** The desired outcome of the workflow — used
            for session-level scoring via :func:`score_session`.
        persist: If ``True`` (default), automatically save the completed
            session to the local SQLite store when the context exits.
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
        if persist:
            try:
                from .storage.store import save_session
                save_session(session)
            except Exception:
                logger.warning(
                    "Failed to persist trace %s to local store",
                    session.trace_id,
                    exc_info=True,
                )
        logger.info(
            "Trace session ended (trace_id=%s, steps=%d)",
            session.trace_id,
            len(session.actions),
        )


def get_current_session() -> TraceSession | None:
    return _current_session.get()


def copy_trace_context() -> contextvars.Context:
    """Copy the current trace context for use in a new thread.

    ``ContextVar`` values are not automatically propagated to threads.
    Use this to capture the context and run traced actions within it::

        from concurrent.futures import ThreadPoolExecutor

        ctx = copy_trace_context()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(ctx.run, my_step, arg1, arg2)
    """
    return contextvars.copy_context()
