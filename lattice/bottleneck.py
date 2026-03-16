from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .context import TraceSession

ImpactType = Literal[
    "error",
    "loop_no_convergence",
    "weakest_branch",
]

logger = logging.getLogger("lattice")


@dataclass
class BottleneckResult:
    """A structural issue found in the trace that scores alone don't surface."""

    action_name: str
    action_index: int
    score: float
    explanation: str
    impact: ImpactType
    latency_ms: float
    span_id: str
    parent_span_id: str | None


def find_bottlenecks(session: TraceSession) -> list[BottleneckResult]:
    """Find structural issues that scores alone don't surface.

    Individual action scores are already on each :class:`ActionRecord` —
    use those directly to see which steps performed poorly.  This function
    adds the non-obvious patterns:

    1. **Errors** — actions that raised exceptions.
    2. **Loop convergence** — repeated actions whose scores failed to improve
       across iterations (impact ``"loop_no_convergence"``).
    3. **Parallel branch imbalance** — the weakest branch when it scores
       significantly below the group average (impact ``"weakest_branch"``).
    """
    results: list[BottleneckResult] = []
    _collect_errors(session, results)
    _analyze_loops(session, results)
    _analyze_parallel(session, results)
    results.sort(key=lambda r: (r.score, -r.latency_ms))
    for r in results:
        logger.info(
            "Bottleneck: %s (score=%.1f, impact=%s, %.1fms)",
            r.action_name, r.score, r.impact, r.latency_ms,
        )
    return results


def _collect_errors(session: TraceSession, results: list[BottleneckResult]) -> None:
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


def _analyze_loops(session: TraceSession, results: list[BottleneckResult]) -> None:
    """Flag repeated actions whose scores failed to improve across iterations."""
    for group in (g for g in session.groups if g.group_type == "loop"):
        loop_steps = [
            s for s in session.actions
            if s.group_id == group.group_id and s.score is not None and s.error is None
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


def _analyze_parallel(session: TraceSession, results: list[BottleneckResult]) -> None:
    """Flag the weakest parallel branch when it falls significantly behind the group."""
    for group in (g for g in session.groups if g.group_type == "parallel"):
        scored_steps = [
            s for s in session.actions
            if s.group_id == group.group_id and s.score is not None and s.error is None
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
