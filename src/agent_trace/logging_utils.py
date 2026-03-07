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


def _build_tree(steps):
    """Build a parent_span_id → children mapping and return root steps."""
    children: dict[str | None, list] = {}
    for s in steps:
        children.setdefault(s.parent_span_id, []).append(s)
    roots = children.get(None, [])
    return roots, children


def _format_step_line(s) -> str:
    status = "ERROR" if s.error else "OK"
    score_str = f"{s.score:.1f}/5" if s.score is not None else "-"
    return f"{s.name}  {status}  {s.latency_ms:.0f}ms  score={score_str}"


def _print_tree(node, children, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{_format_step_line(node)}")
    if node.error:
        detail_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{detail_prefix}error: {node.error}")
    if node.score_explanation:
        detail_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{detail_prefix}reason: {node.score_explanation}")
    child_nodes = children.get(node.span_id, [])
    for i, child in enumerate(child_nodes):
        child_is_last = i == len(child_nodes) - 1
        _print_tree(
            child, children,
            prefix=prefix + ("    " if is_last else "│   "),
            is_last=child_is_last,
        )


def print_trace_summary(session: TraceSession) -> None:
    """Print a human-readable DAG summary of a completed trace session.

    Works independently of Python logging — always prints to stdout.
    """
    steps = session.steps
    if not steps:
        print("(no steps recorded)")
        return

    print(f"\n{'=' * 60}")
    title = session.workflow_name or "Trace Summary"
    print(f"  {title}  (trace_id={session.trace_id})")
    if session.goal:
        print(f"  Goal: {session.goal}")
    print(f"{'=' * 60}")

    roots, children = _build_tree(steps)

    for i, root in enumerate(roots):
        _print_tree(root, children, prefix="  ", is_last=(i == len(roots) - 1))

    total_ms = sum(s.latency_ms for s in steps)
    print(f"{'-' * 60}")
    print(f"  Total steps: {len(steps)}  |  Total time: {total_ms:.0f}ms")
    print(f"{'=' * 60}\n")
