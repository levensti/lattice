"""Console logging helpers for agent-trace."""

from __future__ import annotations

import logging
import sys

from .context import TraceSession

logger = logging.getLogger("agent_trace")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s \u2014 %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _build_tree(steps):
    """Build a parent_span_id -> children mapping and return root steps."""
    children: dict[str | None, list] = {}
    for s in steps:
        children.setdefault(s.parent_span_id, []).append(s)
    roots = children.get(None, [])
    return roots, children


def _format_step_line(s) -> str:
    status = "ERROR" if s.error else "OK"
    score_str = f"{s.score:.1f}/5" if s.score is not None else "-"
    parts = [f"{s.name}  {status}  {s.latency_ms:.0f}ms  score={score_str}"]
    if s.role:
        parts.append(f"role={s.role}")
    if s.iteration is not None:
        parts.append(f"iter={s.iteration}")
    if s.activation_reason:
        parts.append(f"activated: {s.activation_reason}")
    return "  ".join(parts)


def _print_tree(node, children, prefix="", is_last=True):
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    print(f"{prefix}{connector}{_format_step_line(node)}")
    if node.error:
        detail_prefix = prefix + ("    " if is_last else "\u2502   ")
        print(f"{detail_prefix}error: {node.error}")
    if node.score_explanation:
        detail_prefix = prefix + ("    " if is_last else "\u2502   ")
        print(f"{detail_prefix}reason: {node.score_explanation}")
    child_nodes = children.get(node.span_id, [])
    for i, child in enumerate(child_nodes):
        child_is_last = i == len(child_nodes) - 1
        _print_tree(
            child, children,
            prefix=prefix + ("    " if is_last else "\u2502   "),
            is_last=child_is_last,
        )


def print_trace_summary(session: TraceSession) -> None:
    """Print a human-readable summary of a completed trace session.

    Shows the step tree, group info, and recorded transitions.
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

    if session.groups:
        print(f"  Groups:")
        for g in session.groups:
            symbol = "\u27f3" if g.group_type == "loop" else "\u2225"
            group_steps = [s for s in steps if s.group_id == g.group_id]
            step_count = len(group_steps)
            if g.group_type == "loop":
                iters = {
                    s.iteration for s in group_steps
                    if s.iteration is not None
                }
                iter_info = f", {len(iters)} iterations" if iters else ""
                print(f"    {symbol} {g.name} ({step_count} steps{iter_info})")
            else:
                print(f"    {symbol} {g.name} ({step_count} branches)")

    if session.transitions:
        merged: dict[tuple, object] = {}
        for t in session.transitions:
            key = (t.from_span_id, t.to_name)
            existing = merged.get(key)
            if existing is None or (existing.auto and not t.auto):
                merged[key] = t
        display = [t for t in merged.values() if not t.auto or t.reason]
        if display:
            print(f"  Transitions:")
            for t in display:
                from_name = "?"
                if t.from_span_id:
                    from_step = next(
                        (s for s in steps if s.span_id == t.from_span_id), None
                    )
                    from_name = from_step.name if from_step else t.from_span_id
                reason = f': "{t.reason}"' if t.reason else ""
                print(f"    {from_name} \u2192 {t.to_name}{reason}")

    if session.session_score is not None:
        print(f"  Session score: {session.session_score:.1f}/5")
        if session.session_score_explanation:
            print(f"  Verdict: {session.session_score_explanation}")
    print(f"{'=' * 60}\n")
