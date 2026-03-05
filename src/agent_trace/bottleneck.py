from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from .context import TraceSession


@dataclass
class BottleneckResult:
    """A single bottleneck finding from the analysis."""

    step_name: str
    step_index: int
    score: float
    explanation: str
    impact: str  # "error", "lowest_score", "largest_drop", "below_average"
    latency_ms: float


@dataclass
class StepComparison:
    """Comparison of a single pipeline step across multiple session runs."""

    step_name: str
    scores: list[float | None]
    latencies_ms: list[float]
    error_count: int
    score_trend: str  # "improved", "regressed", "stable", "no_data"
    latency_trend: str  # "improved", "regressed", "stable", "no_data"
    score_delta: float  # last - first (negative means regression)
    latency_delta_ms: float  # last - first (positive means slower)
    recurring_bottleneck: bool


@dataclass
class SessionComparison:
    """Result of comparing the same pipeline across multiple session runs."""

    session_ids: list[str]
    step_comparisons: list[StepComparison]
    regressions: list[StepComparison] = field(default_factory=list)
    recurring_bottlenecks: list[StepComparison] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Cross-session comparison
# ---------------------------------------------------------------------------

_SCORE_THRESHOLD = 0.1  # minimum absolute change to count as non-stable
_LATENCY_THRESHOLD_RATIO = 0.1  # 10 % relative change to count as non-stable


def compare_sessions(sessions: list[TraceSession]) -> SessionComparison:
    """Compare the same pipeline across multiple runs.

    Steps are matched by *name* across sessions.  For each step the function
    reports score regressions, latency trends, and whether the step is a
    recurring bottleneck (appeared in ``find_bottlenecks`` results in more
    than half of the sessions it was present in).

    Parameters
    ----------
    sessions:
        Two or more :class:`TraceSession` objects, ordered chronologically
        (earliest run first).

    Returns
    -------
    SessionComparison
        A summary containing per-step comparisons plus convenience lists of
        regressions and recurring bottlenecks.

    Raises
    ------
    ValueError
        If fewer than two sessions are provided.
    """
    if len(sessions) < 2:
        raise ValueError("compare_sessions requires at least two sessions")

    # Collect per-step data across sessions, preserving session order.
    step_scores: dict[str, list[float | None]] = {}
    step_latencies: dict[str, list[float]] = {}
    step_errors: dict[str, int] = {}

    # Track which steps are bottlenecks in each session.
    # A step is a "bottleneck" if it appears in the *worst half* of
    # find_bottlenecks results (or has an error).
    bottleneck_counts: dict[str, int] = {}
    step_session_counts: dict[str, int] = {}

    # Preserve insertion order so output is deterministic: steps appear in
    # the order they are first encountered across sessions.
    seen_steps: list[str] = []

    for session_idx, session in enumerate(sessions):
        # Build a set of step names present in this session so we can pad
        # missing steps with None scores later.
        session_step_names: set[str] = set()

        for step in session.steps:
            name = step.name
            session_step_names.add(name)

            if name not in step_scores:
                # Initialize with None/0.0 for all prior sessions where this
                # step did not exist.
                step_scores[name] = [None] * session_idx
                step_latencies[name] = [0.0] * session_idx
                step_errors[name] = 0
                bottleneck_counts[name] = 0
                step_session_counts[name] = 0
                seen_steps.append(name)

            step_session_counts[name] += 1

            if step.error is not None:
                step_scores[name].append(None)
                step_errors[name] += 1
            else:
                step_scores[name].append(step.score)

            step_latencies[name].append(step.latency_ms)

        # Determine bottleneck steps for this session.
        bnecks = find_bottlenecks(session)
        if bnecks:
            cutoff = max(len(bnecks) // 2, 1)
            worst_names = {b.step_name for b in bnecks[:cutoff]}
            for name in worst_names:
                bottleneck_counts[name] = bottleneck_counts.get(name, 0) + 1

        # For steps that exist in other sessions but not this one, we still
        # need to record a gap so the scores list stays aligned with sessions.
        for name in seen_steps:
            if name not in session_step_names:
                step_scores[name].append(None)
                step_latencies[name].append(0.0)

    # Build StepComparison objects.
    comparisons: list[StepComparison] = []
    regressions: list[StepComparison] = []
    recurring: list[StepComparison] = []

    for name in seen_steps:
        scores = step_scores[name]
        latencies = step_latencies[name]
        error_count = step_errors[name]

        score_trend, score_delta = _compute_score_trend(scores)
        latency_trend, latency_delta = _compute_latency_trend(latencies)

        present_count = step_session_counts[name]
        is_recurring = (
            present_count >= 2
            and bottleneck_counts.get(name, 0) > present_count / 2
        )

        comp = StepComparison(
            step_name=name,
            scores=scores,
            latencies_ms=latencies,
            error_count=error_count,
            score_trend=score_trend,
            latency_trend=latency_trend,
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            recurring_bottleneck=is_recurring,
        )
        comparisons.append(comp)

        if score_trend == "regressed":
            regressions.append(comp)
        if is_recurring:
            recurring.append(comp)

    return SessionComparison(
        session_ids=[s.trace_id for s in sessions],
        step_comparisons=comparisons,
        regressions=regressions,
        recurring_bottlenecks=recurring,
    )


def _compute_score_trend(
    scores: list[float | None],
) -> tuple[str, float]:
    """Return (trend, delta) for a list of per-session scores.

    ``None`` entries (errors / unscored) are skipped.  The trend is based on
    the first and last *available* scores.
    """
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if len(valid) < 2:
        return ("no_data", 0.0)

    first = valid[0][1]
    last = valid[-1][1]
    delta = last - first

    if delta < -_SCORE_THRESHOLD:
        return ("regressed", delta)
    if delta > _SCORE_THRESHOLD:
        return ("improved", delta)
    return ("stable", delta)


def _compute_latency_trend(
    latencies: list[float],
) -> tuple[str, float]:
    """Return (trend, delta_ms) for a list of per-session latencies.

    Zero-valued entries (step absent from a session) are ignored.
    """
    valid = [lat for lat in latencies if lat > 0]
    if len(valid) < 2:
        return ("no_data", 0.0)

    first = valid[0]
    last = valid[-1]
    delta = last - first

    # Use relative threshold so small absolute jitter on fast steps
    # doesn't register as a trend.
    avg = mean(valid)
    if avg == 0:
        return ("stable", 0.0)

    if delta / avg > _LATENCY_THRESHOLD_RATIO:
        return ("regressed", delta)
    if delta / avg < -_LATENCY_THRESHOLD_RATIO:
        return ("improved", delta)
    return ("stable", delta)
