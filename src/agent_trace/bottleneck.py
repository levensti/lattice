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
    impact: str  # "error", "lowest_score", "largest_drop", "below_average"
    latency_ms: float


def find_bottlenecks(session: TraceSession) -> list[BottleneckResult]:
    """Rank steps by quality issues, worst first.

    Steps with errors come first, then scored steps ordered by ascending score.
    Ties are broken by latency (slowest first).
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

    # Check if this step had the largest quality drop from the preceding step
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
