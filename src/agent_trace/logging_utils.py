"""Console logging helpers for agent-trace."""

from __future__ import annotations

import logging
import sys

from .context import TraceSession

logger = logging.getLogger("agent_trace")

# Auto-configure: attach a console handler so users get output for free.
# Uses NullHandler-fallback pattern — if the user has already configured the
# root logger or this logger, the duplicate-handler guard keeps things clean.
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


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
