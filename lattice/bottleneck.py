from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .context import ActionRecord, TraceSession

ImpactType = Literal[
    "error",
    "root_cause",
    "downstream_effect",
    "loop_no_convergence",
    "weakest_branch",
]

logger = logging.getLogger("lattice")


@dataclass
class BottleneckResult:
    """A single bottleneck finding from the analysis."""

    action_name: str
    action_index: int
    score: float
    explanation: str
    impact: ImpactType
    latency_ms: float
    span_id: str
    parent_span_id: str | None


def find_bottlenecks(session: TraceSession) -> list[BottleneckResult]:
    """Rank actions by quality issues, worst first.

    Performs three layers of analysis:

    1. **Tree-aware individual actions** — errors surface first, then scored
       actions classified as ``"root_cause"`` (no low-scoring ancestor) or
       ``"downstream_effect"`` (a parent also scored below average, so the
       problem likely originated upstream).
    2. **Loop convergence** — flags repeated actions whose scores failed to
       improve across iterations (impact ``"loop_no_convergence"``).
    3. **Parallel branch imbalance** — flags the weakest branch when it
       scores significantly below the group average
       (impact ``"weakest_branch"``).

    ``span_id`` and ``parent_span_id`` on each result let callers reconstruct
    the call tree and trace a chain of downstream effects back to their
    root cause.
    """
    span_map: dict[str, ActionRecord] = {
        a.span_id: a for a in session.actions if a.span_id
    }

    results: list[BottleneckResult] = []

    for a in session.actions:
        if a.error is not None:
            results.append(BottleneckResult(
                action_name=a.name,
                action_index=a.action_index,
                score=0.0,
                explanation=f"Action raised an error: {a.error}",
                impact="error",
                latency_ms=a.latency_ms,
                span_id=a.span_id,
                parent_span_id=a.parent_span_id,
            ))
            continue

        if a.score is None:
            continue

        impact = _classify_impact(a, span_map)
        results.append(BottleneckResult(
            action_name=a.name,
            action_index=a.action_index,
            score=a.score,
            explanation=a.score_explanation or "",
            impact=impact,
            latency_ms=a.latency_ms,
            span_id=a.span_id,
            parent_span_id=a.parent_span_id,
        ))

    _analyze_loops(session, results)
    _analyze_parallel(session, results)

    results.sort(key=lambda r: (r.score, -r.latency_ms))
    for r in results:
        logger.info(
            "Bottleneck: %s (score=%.1f, impact=%s, %.1fms)",
            r.action_name, r.score, r.impact, r.latency_ms,
        )
    return results


_BAD_SCORE_THRESHOLD = 3.0  # midpoint of the 1–5 rating scale


def _classify_impact(
    action: ActionRecord,
    span_map: dict[str, ActionRecord],
) -> ImpactType:
    """Walk up the parent chain.

    An action is a ``downstream_effect`` only when it is itself below the
    bad-score threshold *and* at least one ancestor is also below that
    threshold — meaning the problem likely originated upstream.  Actions
    that score well are always ``root_cause`` regardless of their ancestry.
    """
    if action.score >= _BAD_SCORE_THRESHOLD:
        return "root_cause"

    parent_id = action.parent_span_id
    while parent_id is not None:
        parent = span_map.get(parent_id)
        if parent is None:
            break
        if parent.score is not None and parent.score < _BAD_SCORE_THRESHOLD:
            return "downstream_effect"
        parent_id = parent.parent_span_id
    return "root_cause"


def _analyze_loops(
    session: TraceSession,
    results: list[BottleneckResult],
) -> None:
    """Detect loop convergence issues.

    For each action name that repeats across iterations within a loop group,
    check whether scores improved from first to last iteration.
    """
    loop_groups = [g for g in session.groups if g.group_type == "loop"]
    for group in loop_groups:
        loop_steps = [
            s for s in session.actions
            if s.group_id == group.group_id
            and s.score is not None
            and s.error is None
        ]
        if not loop_steps:
            continue

        by_name: dict[str, list] = {}
        for s in loop_steps:
            by_name.setdefault(s.name, []).append(s)

        for name, named_steps in by_name.items():
            if len(named_steps) < 2:
                continue
            named_steps.sort(key=lambda s: (s.iteration if s.iteration is not None else 0))
            scores = [s.score for s in named_steps]
            if scores[-1] <= scores[0]:
                last = named_steps[-1]
                results.append(BottleneckResult(
                    action_name=f"{name} (loop '{group.name}')",
                    action_index=last.action_index,
                    score=scores[-1],
                    explanation=(
                        f"No improvement across {len(scores)} iterations: "
                        + " \u2192 ".join(f"{s:.1f}" for s in scores)
                    ),
                    impact="loop_no_convergence",
                    latency_ms=sum(s.latency_ms for s in named_steps),
                    span_id=last.span_id,
                    parent_span_id=last.parent_span_id,
                ))


def _analyze_parallel(
    session: TraceSession,
    results: list[BottleneckResult],
) -> None:
    """Detect parallel branch imbalance.

    Flags the weakest branch when its score is more than 1.0 below the
    group average.
    """
    parallel_groups = [g for g in session.groups if g.group_type == "parallel"]
    for group in parallel_groups:
        scored_steps = [
            s for s in session.actions
            if s.group_id == group.group_id
            and s.score is not None
            and s.error is None
        ]
        if len(scored_steps) < 2:
            continue

        avg_score = sum(s.score for s in scored_steps) / len(scored_steps)
        worst = min(scored_steps, key=lambda s: s.score)
        if worst.score < avg_score - 1.0:
            results.append(BottleneckResult(
                action_name=f"{worst.name} (parallel '{group.name}')",
                action_index=worst.action_index,
                score=worst.score,
                explanation=(
                    f"Weakest branch: {worst.score:.1f}/5 vs "
                    f"group average {avg_score:.1f}/5"
                ),
                impact="weakest_branch",
                latency_ms=worst.latency_ms,
                span_id=worst.span_id,
                parent_span_id=worst.parent_span_id,
            ))
