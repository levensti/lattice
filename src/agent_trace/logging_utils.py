"""Console logging helpers for agent-trace."""

from __future__ import annotations

import logging
import sys

from .context import TraceSession

logger = logging.getLogger("agent_trace")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Set up the ``agent_trace`` logger with a human-friendly console handler.

    Call this once at startup to see real-time step output in your terminal.
    If the logger already has handlers, this is a no-op so it won't duplicate
    output when called multiple times.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``"INFO"``).
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)


def print_trace_summary(session: TraceSession) -> None:
    """Print a human-readable summary of a completed trace session.

    Works independently of Python logging — always prints to stdout.
    """
    steps = session.steps
    if not steps:
        print("(no steps recorded)")
        return

    print(f"\n{'=' * 60}")
    print(f"  Trace Summary  (trace_id={session.trace_id})")
    print(f"{'=' * 60}")

    total_ms = sum(s.latency_ms for s in steps)

    for s in steps:
        indent = "  " if s.parent_span_id is None else "    -> "
        status = "ERROR" if s.error else "OK"
        score_str = f"{s.score:.1f}/5" if s.score is not None else "—"
        print(
            f"{indent}[{s.step_index}] {s.name}  "
            f"{status}  {s.latency_ms:.0f}ms  score={score_str}"
        )
        if s.error:
            print(f"{indent}     error: {s.error}")
        if s.score_explanation:
            print(f"{indent}     reason: {s.score_explanation}")

    print(f"{'-' * 60}")
    print(f"  Total steps: {len(steps)}  |  Total time: {total_ms:.0f}ms")
    print(f"{'=' * 60}\n")
