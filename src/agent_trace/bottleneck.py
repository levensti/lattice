from __future__ import annotations

import logging
from dataclasses import dataclass

from .context import TraceSession

logger = logging.getLogger("agent_trace")


@dataclass
class BottleneckResult:
    """A single bottleneck finding from the analysis."""

    step_name: str
    step_index: int
    score: float
    explanation: str
    impact: str
    latency_ms: float


def find_bottlenecks(session: TraceSession) -> list[BottleneckResult]:
    """Rank steps by quality issues, worst first.

    Performs three layers of analysis:

    1. **Individual steps** — errors surface first, then scored steps ordered
       by ascending score with ties broken by latency.
    2. **Loop convergence** — flags repeated steps whose scores failed to
       improve across iterations (impact ``"loop_no_convergence"``).
    3. **Parallel branch imbalance** — flags the weakest branch when it
       scores significantly below the group average
       (impact ``"weakest_branch"``).
    """
    results: list[BottleneckResult] = []

    for step in session.steps:
        if step.error is not None:
            results.append(BottleneckResult(
                step_name=step.name,
                step_index=step.step_index,
                score=0.0,
                explanation=f"Step raised an error: {step.error}",
                impact="error",
                latency_ms=step.latency_ms,
            ))
            continue

        if step.score is None:
            continue

        impact = _classify_impact(step.score, step.step_index, session)
        results.append(BottleneckResult(
            step_name=step.name,
            step_index=step.step_index,
            score=step.score,
            explanation=step.score_explanation or "",
            impact=impact,
            latency_ms=step.latency_ms,
        ))

    _analyze_loops(session, results)
    _analyze_parallel(session, results)

    results.sort(key=lambda r: (r.score, -r.latency_ms))
    for r in results:
        logger.info(
            "Bottleneck: %s (score=%.1f, impact=%s, %.1fms)",
            r.step_name, r.score, r.impact, r.latency_ms,
        )
    return results


def _classify_impact(
    score: float, step_index: int, session: TraceSession
) -> str:
    scored = sorted(
        [s for s in session.steps if s.score is not None],
        key=lambda s: s.step_index,
    )
    if not scored:
        return "lowest_score"

    min_score = min(s.score for s in scored)
    if score <= min_score:
        return "lowest_score"

    prev_steps = [s for s in scored if s.step_index < step_index]
    if prev_steps:
        prev_score = prev_steps[-1].score
        drop = prev_score - score
        all_drops = [
            scored[i - 1].score - scored[i].score
            for i in range(1, len(scored))
        ]
        if all_drops and drop >= max(all_drops) and drop > 0:
            return "largest_drop"

    return "below_average"


def _analyze_loops(
    session: TraceSession, results: list[BottleneckResult]
) -> None:
    """Detect loop convergence issues.

    For each step name that repeats across iterations within a loop group,
    check whether scores improved from first to last iteration.
    """
    loop_groups = [g for g in session.groups if g.group_type == "loop"]
    for group in loop_groups:
        loop_steps = [
            s for s in session.steps
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
                results.append(BottleneckResult(
                    step_name=f"{name} (loop '{group.name}')",
                    step_index=named_steps[-1].step_index,
                    score=scores[-1],
                    explanation=(
                        f"No improvement across {len(scores)} iterations: "
                        + " \u2192 ".join(f"{s:.1f}" for s in scores)
                    ),
                    impact="loop_no_convergence",
                    latency_ms=sum(s.latency_ms for s in named_steps),
                ))


def _analyze_parallel(
    session: TraceSession, results: list[BottleneckResult]
) -> None:
    """Detect parallel branch imbalance.

    Flags the weakest branch when its score is more than 1.0 below the
    group average.
    """
    parallel_groups = [g for g in session.groups if g.group_type == "parallel"]
    for group in parallel_groups:
        scored_steps = [
            s for s in session.steps
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
                step_name=f"{worst.name} (parallel '{group.name}')",
                step_index=worst.step_index,
                score=worst.score,
                explanation=(
                    f"Weakest branch: {worst.score:.1f}/5 vs "
                    f"group average {avg_score:.1f}/5"
                ),
                impact="weakest_branch",
                latency_ms=worst.latency_ms,
            ))
