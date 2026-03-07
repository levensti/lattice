from __future__ import annotations

import functools
import inspect
import json
import time
import uuid
from typing import Any, Callable, Sequence

from .context import (
    StepRecord,
    _current_session,
    _current_span_id,
)
from .otel import traced_span


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
    """Bind positional/keyword args to parameter names and serialize."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return _safe_serialize(dict(bound.arguments))
    except (TypeError, ValueError):
        return _safe_serialize({"args": args, "kwargs": kwargs})


def _record_step(
    session, *, span_id, name, description, criteria,
    input_data, output_data, step_index, latency_ms, parent_span_id,
    tags, error=None,
):
    session.add_step(StepRecord(
        span_id=span_id,
        name=name,
        description=description,
        criteria=criteria,
        input_data=input_data,
        output_data=output_data,
        step_index=step_index,
        latency_ms=latency_ms,
        parent_span_id=parent_span_id,
        error=error,
        tags=list(tags),
    ))


def _trace_decorator(
    name: str,
    description: str,
    criteria: str,
    step_id: str | None,
    tags: Sequence[str],
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            session = _current_session.get()
            input_str = _capture_inputs(func, args, kwargs)
            span_id = step_id or uuid.uuid4().hex
            parent_id = _current_span_id.get()
            step_index = session.next_index() if session else 0
            parent_token = _current_span_id.set(span_id)
            start = time.perf_counter()

            otel_attrs = {
                "agent_trace.name": name,
                "agent_trace.input": input_str,
            }
            if tags:
                otel_attrs["agent_trace.tags"] = ",".join(tags)
            try:
                with traced_span(f"step:{name}", otel_attrs) as span:
                    result = func(*args, **kwargs)
                    output_str = _safe_serialize(result)
                    latency = (time.perf_counter() - start) * 1000
                    if span is not None:
                        span.set_attribute("agent_trace.output", output_str[:4096])
                        span.set_attribute("agent_trace.latency_ms", latency)
                    if session:
                        _record_step(
                            session,
                            span_id=span_id, name=name,
                            description=description, criteria=criteria,
                            input_data=input_str, output_data=output_str,
                            step_index=step_index, latency_ms=latency,
                            parent_span_id=parent_id, tags=tags,
                        )
                    return result
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                if session:
                    _record_step(
                        session,
                        span_id=span_id, name=name,
                        description=description, criteria=criteria,
                        input_data=input_str, output_data="",
                        step_index=step_index, latency_ms=latency,
                        parent_span_id=parent_id, tags=tags, error=str(exc),
                    )
                raise
            finally:
                _current_span_id.reset(parent_token)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            session = _current_session.get()
            input_str = _capture_inputs(func, args, kwargs)
            span_id = step_id or uuid.uuid4().hex
            parent_id = _current_span_id.get()
            step_index = session.next_index() if session else 0
            parent_token = _current_span_id.set(span_id)
            start = time.perf_counter()

            otel_attrs = {
                "agent_trace.name": name,
                "agent_trace.input": input_str,
            }
            if tags:
                otel_attrs["agent_trace.tags"] = ",".join(tags)
            try:
                with traced_span(f"step:{name}", otel_attrs) as span:
                    result = await func(*args, **kwargs)
                    output_str = _safe_serialize(result)
                    latency = (time.perf_counter() - start) * 1000
                    if span is not None:
                        span.set_attribute("agent_trace.output", output_str[:4096])
                        span.set_attribute("agent_trace.latency_ms", latency)
                    if session:
                        _record_step(
                            session,
                            span_id=span_id, name=name,
                            description=description, criteria=criteria,
                            input_data=input_str, output_data=output_str,
                            step_index=step_index, latency_ms=latency,
                            parent_span_id=parent_id, tags=tags,
                        )
                    return result
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                if session:
                    _record_step(
                        session,
                        span_id=span_id, name=name,
                        description=description, criteria=criteria,
                        input_data=input_str, output_data="",
                        step_index=step_index, latency_ms=latency,
                        parent_span_id=parent_id, tags=tags, error=str(exc),
                    )
                raise
            finally:
                _current_span_id.reset(parent_token)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def step(
    name: str,
    *,
    description: str = "",
    criteria: str = "",
    step_id: str | None = None,
    tags: Sequence[str] = (),
) -> Callable:
    """Decorator to trace any step in a workflow with quality criteria.

    This is the primary decorator for agent-trace. Annotate each meaningful
    function in your pipeline and the framework handles tracing, scoring,
    and bottleneck detection automatically.

    Args:
        name: Human-readable step name.
        description: What this step does.
        criteria: Quality criteria the judge will evaluate against.
        step_id: Custom span ID (auto-generated if omitted).
        tags: Optional labels for grouping/filtering (e.g. ["llm", "io"]).
    """
    return _trace_decorator(name, description, criteria, step_id, tags)
