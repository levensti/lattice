from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .context import ActionRecord, GroupRecord, TraceSession

InsightType = Literal[
    "error_streak",
    "low_score_streak",
    "quality_cascade_chain",
    "loop_no_convergence",
    "parallel_imbalance",
    "routing_thrashing",
    "latency_spike",
]


@dataclass
class SessionInsight:
    """Session-level issue with evidence pointing into the trace."""

    insight_type: InsightType
    severity: int  # 1..5
    title: str
    summary: str
    evidence_span_ids: list[str]
    evidence_action_indices: list[int]


def analyze_session(
    session: TraceSession,
    *,
    low_score_threshold: float | None = None,
    min_streak: int = 3,
    cascade_max_depth: int = 6,
    parallel_imbalance_threshold: float = 1.0,
) -> list[SessionInsight]:
    """Derive session-level insights from a trace session.

    This focuses on patterns that are hard to see from per-action scores:
    cascades, streaks, routing thrash, and structural group pathologies.
    """
    insights: list[SessionInsight] = []

    actions = list(session.actions)
    if not actions:
        return insights

    by_span = {a.span_id: a for a in actions}

    scored_actions = [a for a in actions if a.score is not None and a.error is None]
    if low_score_threshold is None:
        if scored_actions:
            scores = sorted(a.score for a in scored_actions)
            # Approx p25 as a "concerning" cutoff; keeps this robust across rubrics.
            low_score_threshold = scores[max(0, int(len(scores) * 0.25) - 1)]
        else:
            low_score_threshold = 0.0

    insights.extend(_error_streak(actions, min_streak=min_streak))
    insights.extend(
        _low_score_streak(
            actions,
            threshold=low_score_threshold,
            min_streak=min_streak,
        )
    )
    insights.extend(_cascade_chains(session, by_span, low_score_threshold, max_depth=cascade_max_depth))
    insights.extend(_loop_non_convergence(session, min_iters=2))
    insights.extend(_parallel_imbalance(session, threshold=parallel_imbalance_threshold))
    insights.extend(_routing_thrashing(session, min_repeat=3))
    insights.extend(_latency_spikes(actions, by_span, min_ratio=2.5))

    insights.sort(key=lambda i: (-i.severity, i.title))
    return insights


def _error_streak(actions: list[ActionRecord], *, min_streak: int) -> list[SessionInsight]:
    streaks: list[tuple[int, int]] = []
    start = None
    for i, a in enumerate(actions):
        if a.error:
            if start is None:
                start = i
        else:
            if start is not None:
                streaks.append((start, i - 1))
                start = None
    if start is not None:
        streaks.append((start, len(actions) - 1))

    insights: list[SessionInsight] = []
    for s, e in streaks:
        length = e - s + 1
        if length < min_streak:
            continue
        ev = actions[s : e + 1]
        insights.append(
            SessionInsight(
                insight_type="error_streak",
                severity=min(5, 2 + length // 2),
                title=f"Error streak ({length} actions)",
                summary="Consecutive errors suggest a shared upstream assumption or missing precondition.",
                evidence_span_ids=[a.span_id for a in ev],
                evidence_action_indices=[a.action_index for a in ev],
            )
        )
    return insights


def _low_score_streak(
    actions: list[ActionRecord],
    *,
    threshold: float,
    min_streak: int,
) -> list[SessionInsight]:
    streaks: list[tuple[int, int]] = []
    start = None
    for i, a in enumerate(actions):
        is_low = a.score is not None and a.error is None and a.score <= threshold
        if is_low:
            if start is None:
                start = i
        else:
            if start is not None:
                streaks.append((start, i - 1))
                start = None
    if start is not None:
        streaks.append((start, len(actions) - 1))

    insights: list[SessionInsight] = []
    for s, e in streaks:
        length = e - s + 1
        if length < min_streak:
            continue
        ev = actions[s : e + 1]
        avg = sum(a.score for a in ev if a.score is not None) / length
        insights.append(
            SessionInsight(
                insight_type="low_score_streak",
                severity=min(5, 2 + length // 2),
                title=f"Low-score streak ({length} actions)",
                summary=f"Scores stayed low across multiple steps (avg {avg:.1f}). This often indicates a broken plan or wrong intermediate representation.",
                evidence_span_ids=[a.span_id for a in ev],
                evidence_action_indices=[a.action_index for a in ev],
            )
        )
    return insights


def _cascade_chains(
    session: TraceSession,
    by_span: dict[str, ActionRecord],
    low_threshold: float,
    *,
    max_depth: int,
) -> list[SessionInsight]:
    # Build child map
    children: dict[str, list[ActionRecord]] = {}
    for a in session.actions:
        if a.parent_span_id:
            children.setdefault(a.parent_span_id, []).append(a)

    def score(a: ActionRecord) -> float | None:
        if a.error is not None:
            return None
        return a.score

    insights: list[SessionInsight] = []
    seen_roots: set[str] = set()

    for a in session.actions:
        s = score(a)
        if s is None or s > low_threshold:
            continue
        if a.span_id in seen_roots:
            continue

        chain: list[ActionRecord] = [a]
        cur = a
        for _ in range(max_depth - 1):
            kids = [k for k in children.get(cur.span_id, []) if score(k) is not None]
            if not kids:
                break
            # Pick the lowest-scoring child as the "worst cascade path"
            nxt = min(kids, key=lambda k: score(k) if score(k) is not None else 1e9)
            if score(nxt) is None:
                break
            # Stop if the child is not low; cascade seems to have recovered.
            if score(nxt) > low_threshold:
                break
            chain.append(nxt)
            cur = nxt

        if len(chain) >= 3:
            seen_roots.add(a.span_id)
            first, last = chain[0], chain[-1]
            drop = (first.score - last.score) if (first.score is not None and last.score is not None) else 0.0
            insights.append(
                SessionInsight(
                    insight_type="quality_cascade_chain",
                    severity=min(5, 3 + (len(chain) - 3)),
                    title=f"Quality cascade chain ({len(chain)} steps)",
                    summary=(
                        f"Downstream steps stayed low after '{first.name}'. "
                        f"Focus on improving upstream output contracts. "
                        f"Score drift along worst path: {drop:.1f}."
                    ),
                    evidence_span_ids=[x.span_id for x in chain],
                    evidence_action_indices=[x.action_index for x in chain],
                )
            )

    return insights


def _loop_non_convergence(session: TraceSession, *, min_iters: int) -> list[SessionInsight]:
    group_map = {g.group_id: g for g in session.groups}
    insights: list[SessionInsight] = []

    for g in session.groups:
        if g.group_type != "loop":
            continue
        steps = [
            a for a in session.actions
            if a.group_id == g.group_id and a.score is not None and a.error is None and a.iteration is not None
        ]
        if not steps:
            continue
        iters = sorted({a.iteration for a in steps if a.iteration is not None})
        if len(iters) < min_iters:
            continue

        by_name: dict[str, list[ActionRecord]] = {}
        for a in steps:
            by_name.setdefault(a.name, []).append(a)

        for name, named in by_name.items():
            if len(named) < min_iters:
                continue
            named.sort(key=lambda a: a.iteration or 0)
            scores = [a.score for a in named if a.score is not None]
            if len(scores) < min_iters:
                continue
            if scores[-1] <= scores[0]:
                insights.append(
                    SessionInsight(
                        insight_type="loop_no_convergence",
                        severity=4,
                        title=f"Loop not converging: {name}",
                        summary=(
                            f"In loop '{group_map[g.group_id].name}', scores did not improve: "
                            + " → ".join(f"{s:.1f}" for s in scores)
                        ),
                        evidence_span_ids=[a.span_id for a in named],
                        evidence_action_indices=[a.action_index for a in named],
                    )
                )

    return insights


def _parallel_imbalance(session: TraceSession, *, threshold: float) -> list[SessionInsight]:
    group_map: dict[str, GroupRecord] = {g.group_id: g for g in session.groups}
    insights: list[SessionInsight] = []

    for g in session.groups:
        if g.group_type != "parallel":
            continue
        steps = [
            a for a in session.actions
            if a.group_id == g.group_id and a.score is not None and a.error is None
        ]
        if len(steps) < 2:
            continue
        avg = sum(a.score for a in steps if a.score is not None) / len(steps)
        worst = min(steps, key=lambda a: a.score if a.score is not None else 1e9)
        if worst.score is not None and worst.score < avg - threshold:
            insights.append(
                SessionInsight(
                    insight_type="parallel_imbalance",
                    severity=3,
                    title=f"Parallel branch imbalance ({group_map[g.group_id].name})",
                    summary=f"'{worst.name}' underperformed: {worst.score:.1f} vs group avg {avg:.1f}.",
                    evidence_span_ids=[worst.span_id],
                    evidence_action_indices=[worst.action_index],
                )
            )

    return insights


def _routing_thrashing(session: TraceSession, *, min_repeat: int) -> list[SessionInsight]:
    # Detect repeated transitions to the same next action name from multiple points
    # without clear progress signals.
    if not session.transitions:
        return []
    counts: dict[str, list[str | None]] = {}
    for t in session.transitions:
        counts.setdefault(t.to_name, []).append(t.from_span_id)

    insights: list[SessionInsight] = []
    for to_name, froms in counts.items():
        if len(froms) < min_repeat:
            continue
        # Keep evidence as the unique "from" spans we can link to.
        ev_spans = [f for f in froms if f is not None]
        if not ev_spans:
            continue
        insights.append(
            SessionInsight(
                insight_type="routing_thrashing",
                severity=3,
                title=f"Routing thrash to '{to_name}'",
                summary=f"'{to_name}' was selected {len(froms)} times. Consider adding a stronger router guard / termination criterion.",
                evidence_span_ids=list(dict.fromkeys(ev_spans))[:8],
                evidence_action_indices=[],
            )
        )
    return insights


def _latency_spikes(
    actions: list[ActionRecord],
    by_span: dict[str, ActionRecord],
    *,
    min_ratio: float,
) -> list[SessionInsight]:
    latencies = [a.latency_ms for a in actions if a.latency_ms is not None]
    if len(latencies) < 4:
        return []
    s = sorted(latencies)
    p50 = s[len(s) // 2]
    if p50 <= 0:
        return []

    spikes = [a for a in actions if a.latency_ms >= p50 * min_ratio]
    if not spikes:
        return []
    spikes.sort(key=lambda a: -a.latency_ms)
    top = spikes[:5]
    return [
        SessionInsight(
            insight_type="latency_spike",
            severity=2,
            title="Latency spikes",
            summary=f"Some actions were much slower than typical (median {_fmt_ms(p50)}).",
            evidence_span_ids=[a.span_id for a in top],
            evidence_action_indices=[a.action_index for a in top],
        )
    ]


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"

