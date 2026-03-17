from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .context import ActionRecord, TraceSession

ImpactType = Literal[
    "error",
    "lowest_score",
    "quality_cascade",
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


def find_bottlenecks(
    session: TraceSession,
    *,
    weakest_branch_threshold: float = 1.0,
    convergence_strict: bool = True,
) -> list[BottleneckResult]:
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
    _find_lowest_scored(session, results)
    _analyze_cascades(session, results)
    _analyze_loops(session, results, convergence_strict=convergence_strict)
    _analyze_parallel(session, results, weakest_branch_threshold=weakest_branch_threshold)
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


def _find_lowest_scored(session: TraceSession, results: list[BottleneckResult]) -> None:
    """Flag the lowest-scoring action when it falls notably below the session average.

    Uses a relative threshold: only flags when the worst score is more than one
    standard deviation below the mean, avoiding false positives from small
    natural variation.
    """
    scored = [a for a in session.actions if a.score is not None and a.error is None]
    if len(scored) < 2:
        return
    scores = [a.score for a in scored]
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    stddev = variance ** 0.5
    if stddev == 0:
        return  # all scores identical
    worst = min(scored, key=lambda a: a.score)
    if worst.score < avg - stddev:
        results.append(BottleneckResult(
            action_name=worst.name,
            action_index=worst.action_index,
            score=worst.score,
            explanation=(
                f"Lowest-scoring action: {worst.score:.1f} vs "
                f"session average {avg:.1f}"
            ),
            impact="lowest_score",
            latency_ms=worst.latency_ms,
            span_id=worst.span_id,
            parent_span_id=worst.parent_span_id,
        ))


def _analyze_cascades(session: TraceSession, results: list[BottleneckResult]) -> None:
    """Flag parent actions whose low score cascaded to degrade child quality.

    A cascade is detected when a parent scores below the session average and
    at least one of its children also scores below average — the parent is
    flagged as the root cause since its output becomes the child's input.
    """
    scored = {
        a.span_id: a for a in session.actions
        if a.score is not None and a.error is None
    }
    if len(scored) < 2:
        return

    all_scores = [a.score for a in scored.values()]
    avg = sum(all_scores) / len(all_scores)
    flagged: set[str] = set()

    for a in session.actions:
        if a.parent_span_id is None or a.span_id not in scored:
            continue
        parent = scored.get(a.parent_span_id)
        if parent is None:
            continue
        # Parent scored below average AND child also below average → cascade
        if parent.score < avg and a.score < avg and parent.span_id not in flagged:
            flagged.add(parent.span_id)
            results.append(BottleneckResult(
                action_name=parent.name,
                action_index=parent.action_index,
                score=parent.score,
                explanation=(
                    f"Low score ({parent.score:.1f}) cascaded to child "
                    f"'{a.name}' ({a.score:.1f})"
                ),
                impact="quality_cascade",
                latency_ms=parent.latency_ms,
                span_id=parent.span_id,
                parent_span_id=parent.parent_span_id,
            ))


def _analyze_loops(session: TraceSession, results: list[BottleneckResult], *, convergence_strict: bool = True) -> None:
    """Flag repeated actions whose scores failed to improve across iterations."""
    for group in (g for g in session.groups if g.group_type == "loop"):
        loop_steps = [
            s for s in session.actions
            if s.group_id == group.group_id and s.score is not None and s.error is None
        ]
        if not loop_steps:
            continue

        by_name: dict[str, list[ActionRecord]] = {}
        for s in loop_steps:
            by_name.setdefault(s.name, []).append(s)

        for name, named_steps in by_name.items():
            if len(named_steps) < 2:
                continue
            named_steps.sort(key=lambda s: (s.iteration if s.iteration is not None else 0))
            scores = [s.score for s in named_steps]
            if scores[-1] <= scores[0] if convergence_strict else scores[-1] < scores[0]:
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


def _analyze_parallel(session: TraceSession, results: list[BottleneckResult], *, weakest_branch_threshold: float = 1.0) -> None:
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
        if worst.score < avg_score - weakest_branch_threshold:
            results.append(BottleneckResult(
                action_name=f"{worst.name} (parallel '{group.name}')",
                action_index=worst.action_index,
                score=worst.score,
                explanation=(
                    f"Weakest branch: {worst.score:.1f} vs "
                    f"group average {avg_score:.1f}"
                ),
                impact="weakest_branch",
                latency_ms=worst.latency_ms,
                span_id=worst.span_id,
                parent_span_id=worst.parent_span_id,
            ))
