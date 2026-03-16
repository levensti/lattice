from __future__ import annotations

import functools
import inspect
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Sequence

from .context import (
    ActionRecord,
    JudgeConfig,
    TransitionRecord,
    _current_activation_reason,
    _current_group_id,
    _current_iteration,
    _current_session,
    _current_span_id,
)

logger = logging.getLogger("lattice")

_SELF_CLS = frozenset(("self", "cls"))


def _safe_serialize(obj: Any, max_length: int = 8192) -> str:
    """Best-effort JSON serialization with a size cap."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        s = repr(obj)
    if len(s) > max_length:
        return s[:max_length] + "...[truncated]"
    return s


def _capture_inputs(func: Callable, args: tuple, kwargs: dict) -> str:
    """Bind positional/keyword args to parameter names and serialize.

    Automatically excludes ``self`` and ``cls`` so class methods don't
    serialize the entire instance.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        filtered = {
            k: v for k, v in bound.arguments.items()
            if k not in _SELF_CLS
        }
        return _safe_serialize(filtered)
    except (TypeError, ValueError):
        return _safe_serialize({"args": args, "kwargs": kwargs})


def _record_action(
    session, *, span_id, name, description, goal,
    input_data, output_data, action_index, latency_ms, parent_span_id,
    tags, role=None, group_id=None, iteration=None,
    activation_reason=None, error=None, judge=None,
):
    session.add_action(ActionRecord(
        span_id=span_id,
        name=name,
        description=description,
        goal=goal,
        input_data=input_data,
        output_data=output_data,
        action_index=action_index,
        latency_ms=latency_ms,
        parent_span_id=parent_span_id,
        error=error,
        tags=list(tags),
        role=role,
        group_id=group_id,
        iteration=iteration,
        activation_reason=activation_reason,
        judge=judge,
    ))
    if parent_span_id is not None:
        session.add_transition(TransitionRecord(
            from_span_id=parent_span_id,
            to_name=name,
            reason="",
            auto=True,
        ))


def _read_topology_context() -> dict:
    """Read the current group, iteration, and activation context vars."""
    return {
        "group_id": _current_group_id.get(),
        "iteration": _current_iteration.get(),
        "activation_reason": _current_activation_reason.get(),
    }


def _begin_action(func, args, kwargs, *, action_id, name, tags, role):
    """Common setup for both sync and async action wrappers.

    Returns a dict of context needed by ``_complete_action`` / ``_fail_action``.
    """
    session = _current_session.get()
    input_str = _capture_inputs(func, args, kwargs)
    span_id = action_id or uuid.uuid4().hex
    parent_id = _current_span_id.get()
    action_index = session.next_index() if session else 0
    topo = _read_topology_context()
    parent_token = _current_span_id.set(span_id)

    logger.info("Action started: %s", name)
    logger.debug("Action %s input: %s", name, input_str[:500])

    return {
        "session": session, "input_str": input_str,
        "span_id": span_id, "parent_id": parent_id,
        "action_index": action_index, "topo": topo,
        "parent_token": parent_token,
        "start": time.perf_counter(),
    }


def _complete_action(ctx, *, name, description, goal, tags, role, result, judge=None):
    """Record a successful action and return the serialized output."""
    output_str = _safe_serialize(result)
    latency = (time.perf_counter() - ctx["start"]) * 1000
    if ctx["session"]:
        _record_action(
            ctx["session"],
            span_id=ctx["span_id"], name=name,
            description=description, goal=goal,
            input_data=ctx["input_str"], output_data=output_str,
            action_index=ctx["action_index"], latency_ms=latency,
            parent_span_id=ctx["parent_id"], tags=tags, role=role,
            judge=judge, **ctx["topo"],
        )
    logger.info("Action completed: %s (%.1fms)", name, latency)
    logger.debug("Action %s output: %s", name, output_str[:500])


def _fail_action(ctx, *, name, description, goal, tags, role, exc, judge=None):
    """Record a failed action."""
    latency = (time.perf_counter() - ctx["start"]) * 1000
    logger.error("Action failed: %s (%.1fms) — %s", name, latency, exc)
    if ctx["session"]:
        _record_action(
            ctx["session"],
            span_id=ctx["span_id"], name=name,
            description=description, goal=goal,
            input_data=ctx["input_str"], output_data="",
            action_index=ctx["action_index"], latency_ms=latency,
            parent_span_id=ctx["parent_id"], tags=tags, role=role,
            error=str(exc), judge=judge, **ctx["topo"],
        )


def _trace_decorator(
    name: str,
    description: str,
    goal: str,
    action_id: str | None,
    tags: Sequence[str],
    role: str | None,
    judge: JudgeConfig | None = None,
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _begin_action(func, args, kwargs, action_id=action_id, name=name, tags=tags, role=role)
            try:
                result = func(*args, **kwargs)
                _complete_action(ctx, name=name, description=description, goal=goal, tags=tags, role=role, result=result, judge=judge)
                return result
            except Exception as exc:
                _fail_action(ctx, name=name, description=description, goal=goal, tags=tags, role=role, exc=exc, judge=judge)
                raise
            finally:
                _current_span_id.reset(ctx["parent_token"])

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _begin_action(func, args, kwargs, action_id=action_id, name=name, tags=tags, role=role)
            try:
                result = await func(*args, **kwargs)
                _complete_action(ctx, name=name, description=description, goal=goal, tags=tags, role=role, result=result, judge=judge)
                return result
            except Exception as exc:
                _fail_action(ctx, name=name, description=description, goal=goal, tags=tags, role=role, exc=exc, judge=judge)
                raise
            finally:
                _current_span_id.reset(ctx["parent_token"])

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def action(
    _func=None,
    *,
    name: str | None = None,
    description: str = "",
    goal: str = "",
    action_id: str | None = None,
    tags: Sequence[str] = (),
    role: str | None = None,
    judge: JudgeConfig | None = None,
) -> Callable:
    """Decorator to trace an action in a workflow.

    A ``goal`` is required so the judge can evaluate quality::

        @action(goal="Must cite sources")
        def research(topic): ...

        @action(goal="Decide which tool to call", role="think")
        def think(state): ...

    Override the global judge for this action, add criteria, or inject a
    reference answer::

        @action(
            goal="Research and cite sources accurately",
            judge=JudgeConfig(
                model="claude-opus-4-6",
                criteria={
                    "factual_accuracy": "Are claims supported by evidence?",
                    "citation_quality": "Are sources cited properly?",
                },
            ),
        )
        def research(topic): ...

    Args:
        name: Action name (defaults to the function's ``__name__``).
        description: What this action does.
        goal: **Required.** What success looks like — the judge evaluates
            against this.
        action_id: Custom span ID (auto-generated if omitted).
        tags: Labels for grouping/filtering (e.g. ``["llm", "io"]``).
        role: Semantic role (e.g. ``"generator"``, ``"evaluator"``,
              ``"think"``). Used by topology-aware analysis.
        judge: Per-action :class:`~lattice.context.JudgeConfig`. When set,
            overrides the global judge for this action. Supports ``criteria``
            for multi-axis rubric evaluation and ``reference`` for calibrated
            scoring — both resolved in a single LLM call.

    Raises:
        TypeError: If *goal* is not provided.
    """
    if _func is not None and not callable(_func):
        if name is not None:
            raise TypeError("Cannot pass both positional and keyword 'name'")
        name = _func
        _func = None

    def decorator(func: Callable) -> Callable:
        actual_name = name if name is not None else func.__name__
        if not goal:
            raise TypeError(
                f"@action requires a goal — use @action(goal=\"...\") "
                f"instead of bare @action on '{actual_name}'."
            )
        return _trace_decorator(actual_name, description, goal, action_id, tags, role, judge)(func)

    if _func is not None:
        return decorator(_func)
    return decorator


class _TraceActionHandle:
    """Handle yielded by :func:`trace_action` for setting output."""

    __slots__ = ("_output",)

    def __init__(self):
        self._output = None

    def set_output(self, output: Any) -> None:
        """Record the output of the traced block."""
        self._output = output


@contextmanager
def trace_action(
    name: str,
    *,
    goal: str = "",
    description: str = "",
    role: str | None = None,
    tags: Sequence[str] = (),
    input_data: Any = None,
    judge: JudgeConfig | None = None,
):
    """Context manager for tracing code you don't own or can't decorate.

    Use this when the ``@action`` decorator isn't applicable — for example,
    when calling third-party functions or wrapping a block of code that
    isn't a single function call::

        with trace_action("external_search", goal="Return results") as ts:
            result = third_party_api.search(query)
            ts.set_output(result)

    Args:
        name: Action name.
        goal: **Required.** Quality goal for the judge.
        description: What this block does.
        role: Semantic role in the architecture.
        tags: Labels for grouping/filtering.
        input_data: Optional input to record on the action.
    """
    if not goal:
        raise TypeError(
            f"trace_action requires a goal — use "
            f"trace_action(\"{name}\", goal=\"...\")."
        )
    session = _current_session.get()
    span_id = uuid.uuid4().hex
    parent_id = _current_span_id.get()
    action_index = session.next_index() if session else 0
    topo = _read_topology_context()
    parent_token = _current_span_id.set(span_id)
    start = time.perf_counter()

    input_str = _safe_serialize(input_data) if input_data is not None else ""
    handle = _TraceActionHandle()

    logger.info("Action started: %s", name)
    try:
        yield handle
        output_str = _safe_serialize(handle._output) if handle._output is not None else ""
        latency = (time.perf_counter() - start) * 1000
        if session:
            _record_action(
                session,
                span_id=span_id, name=name,
                description=description, goal=goal,
                input_data=input_str, output_data=output_str,
                action_index=action_index, latency_ms=latency,
                parent_span_id=parent_id, tags=tags, role=role,
                judge=judge, **topo,
            )
        logger.info("Action completed: %s (%.1fms)", name, latency)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.error("Action failed: %s (%.1fms) — %s", name, latency, exc)
        if session:
            _record_action(
                session,
                span_id=span_id, name=name,
                description=description, goal=goal,
                input_data=input_str, output_data="",
                action_index=action_index, latency_ms=latency,
                parent_span_id=parent_id, tags=tags, role=role,
                error=str(exc), judge=judge, **topo,
            )
        raise
    finally:
        _current_span_id.reset(parent_token)


def instrument(
    func: Callable,
    *,
    name: str | None = None,
    goal: str = "",
    description: str = "",
    tags: Sequence[str] = (),
    role: str | None = None,
    judge: JudgeConfig | None = None,
) -> Callable:
    """Instrument an existing function for tracing without modifying its source.

    Returns a new callable that traces calls to *func*.
    The original function is not modified::

        from lattice import instrument

        # Instrument a third-party function
        traced_search = instrument(search_api, goal="Return results")
        results = traced_search(query)

        # Or instrument a method on an instance
        agent.search = instrument(agent.search, goal="Find documents")

    Args:
        func: The function to instrument.
        name: Action name (defaults to ``func.__name__``).
        goal: **Required.** Quality goal for the judge.
        description: What this function does.
        tags: Labels for grouping/filtering.
        role: Semantic role in the architecture.

    Raises:
        TypeError: If *goal* is not provided.
    """
    actual_name = name or getattr(func, "__name__", repr(func))
    if not goal:
        raise TypeError(
            f"instrument() requires a goal — use "
            f"instrument({actual_name}, goal=\"...\")."
        )
    return _trace_decorator(actual_name, description, goal, None, tags, role, judge)(func)
