"""Console logging helpers for Lattice."""

from __future__ import annotations

import logging
import sys

from .context import ActionRecord, TraceSession, TransitionRecord

logger = logging.getLogger("lattice")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s \u2014 %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _build_tree(actions: list[ActionRecord]) -> tuple[list[ActionRecord], dict[str | None, list[ActionRecord]]]:
    """Build a parent_span_id -> children mapping and return root actions."""
    children: dict[str | None, list[ActionRecord]] = {}
    for s in actions:
        children.setdefault(s.parent_span_id, []).append(s)
    roots = children.get(None, [])
    return roots, children


def _format_action_line(s: ActionRecord) -> str:
    status = "ERROR" if s.error else "OK"
    score_str = f"{s.score:.1f}" if s.score is not None else "-"
    parts = [f"{s.name}  {status}  {s.latency_ms:.0f}ms  score={score_str}"]
    if s.role:
        parts.append(f"role={s.role}")
    if s.iteration is not None:
        parts.append(f"iter={s.iteration}")
    if s.activation_reason:
        parts.append(f"activated: {s.activation_reason}")
    return "  ".join(parts)


def _print_tree(node: ActionRecord, children: dict[str | None, list[ActionRecord]], prefix: str = "", is_last: bool = True) -> None:
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    print(f"{prefix}{connector}{_format_action_line(node)}")
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

    Shows the action tree, group info, and recorded transitions.
    """
    actions = session.actions
    if not actions:
        print("(no actions recorded)")
        return

    print(f"\n{'=' * 60}")
    title = session.workflow_name or "Trace Summary"
    print(f"  {title}  (trace_id={session.trace_id})")
    if session.goal:
        print(f"  Goal: {session.goal}")
    print(f"{'=' * 60}")

    roots, children = _build_tree(actions)

    for i, root in enumerate(roots):
        _print_tree(root, children, prefix="  ", is_last=(i == len(roots) - 1))

    total_ms = sum(s.latency_ms for s in actions)
    print(f"{'-' * 60}")
    print(f"  Total actions: {len(actions)}  |  Total time: {total_ms:.0f}ms")

    if session.groups:
        print(f"  Groups:")
        for g in session.groups:
            symbol = "\u27f3" if g.group_type == "loop" else "\u2225"
            group_steps = [s for s in actions if s.group_id == g.group_id]
            action_count = len(group_steps)
            if g.group_type == "loop":
                iters = {
                    s.iteration for s in group_steps
                    if s.iteration is not None
                }
                iter_info = f", {len(iters)} iterations" if iters else ""
                print(f"    {symbol} {g.name} ({action_count} steps{iter_info})")
            else:
                print(f"    {symbol} {g.name} ({action_count} branches)")

    if session.transitions:
        merged: dict[tuple[str | None, str], TransitionRecord] = {}
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
                        (s for s in actions if s.span_id == t.from_span_id), None
                    )
                    from_name = from_step.name if from_step else t.from_span_id
                reason = f': "{t.reason}"' if t.reason else ""
                print(f"    {from_name} \u2192 {t.to_name}{reason}")

    if session.session_score is not None:
        print(f"  Session score: {session.session_score:.1f}")
        if session.session_score_explanation:
            print(f"  Verdict: {session.session_score_explanation}")
    print(f"{'=' * 60}\n")
