"""Export utilities for TraceSession: JSON and self-contained HTML reports."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import StepRecord, TraceSession


def _step_to_dict(step: StepRecord) -> dict[str, Any]:
    """Convert a StepRecord to a JSON-serialisable dictionary."""
    return asdict(step)


def _session_to_dict(session: TraceSession) -> dict[str, Any]:
    """Build the full JSON-serialisable representation of a session."""
    from .bottleneck import find_bottlenecks

    steps = [_step_to_dict(s) for s in session.steps]
    bottlenecks = [asdict(b) for b in find_bottlenecks(session)]

    scored = [s for s in session.steps if s.score is not None]
    total_latency = sum(s.latency_ms for s in session.steps)
    avg_score = (
        sum(s.score for s in scored) / len(scored) if scored else None
    )

    return {
        "trace_id": session.trace_id,
        "summary": {
            "total_steps": len(session.steps),
            "total_latency_ms": round(total_latency, 2),
            "average_score": round(avg_score, 2) if avg_score is not None else None,
            "error_count": sum(1 for s in session.steps if s.error is not None),
            "scored_steps": len(scored),
        },
        "steps": steps,
        "bottlenecks": bottlenecks,
    }


def export_json(session: TraceSession, indent: int = 2) -> str:
    """Return the session as a formatted JSON string."""
    return json.dumps(_session_to_dict(session), indent=indent)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:#0f1117;color:#e1e4e8;padding:24px;line-height:1.5}
h1{font-size:1.6rem;margin-bottom:4px;color:#f0f6fc}
.subtitle{color:#8b949e;font-size:.85rem;margin-bottom:24px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px;margin-bottom:28px}
.summary-card{background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:16px;text-align:center}
.summary-card .value{font-size:1.5rem;font-weight:700;color:#58a6ff}
.summary-card .label{font-size:.75rem;color:#8b949e;text-transform:uppercase;
  letter-spacing:.05em;margin-top:4px}
h2{font-size:1.15rem;margin:24px 0 12px;color:#f0f6fc;
  border-bottom:1px solid #30363d;padding-bottom:6px}
.timeline{position:relative;padding-left:28px;margin-bottom:28px}
.timeline::before{content:'';position:absolute;left:10px;top:0;bottom:0;
  width:2px;background:#30363d}
.step{position:relative;margin-bottom:16px;background:#161b22;
  border:1px solid #30363d;border-radius:8px;padding:14px 16px}
.step .dot{position:absolute;left:-23px;top:18px;width:12px;height:12px;
  border-radius:50%;border:2px solid #30363d;background:#0f1117}
.step.error .dot{background:#f85149;border-color:#f85149}
.step.bottleneck{border-color:#d29922}
.step-header{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.step-name{font-weight:600;font-size:.95rem}
.badge{font-size:.7rem;padding:2px 7px;border-radius:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.03em}
.badge-agent{background:#1f3a5f;color:#58a6ff}
.badge-tool{background:#1a3a2a;color:#3fb950}
.badge-error{background:#4a1e1e;color:#f85149}
.badge-bottleneck{background:#3d2e00;color:#d29922}
.step-meta{display:flex;gap:16px;margin-top:6px;font-size:.78rem;color:#8b949e}
.score-bar-container{margin-top:8px}
.score-bar-track{height:8px;background:#21262d;border-radius:4px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:4px;transition:width .3s}
.score-label{font-size:.72rem;color:#8b949e;margin-top:2px}
.explanation{margin-top:6px;font-size:.82rem;color:#b1bac4;
  background:#0d1117;border-radius:6px;padding:8px 10px;
  border-left:3px solid #30363d}
.bottleneck-callout{margin-top:6px;font-size:.82rem;color:#d29922;
  background:#1c1500;border-radius:6px;padding:8px 10px;
  border-left:3px solid #d29922}
.heatmap{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:28px}
.heatmap-cell{width:48px;height:48px;border-radius:6px;display:flex;
  flex-direction:column;align-items:center;justify-content:center;
  font-size:.65rem;color:#e1e4e8;border:1px solid #30363d;cursor:default}
.heatmap-cell .cell-name{overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;max-width:44px;font-size:.55rem}
.heatmap-cell .cell-score{font-weight:700;font-size:.8rem}
.collapsible{cursor:pointer;user-select:none}
.collapsible-content{display:none;margin-top:8px}
.collapsible-content.open{display:block}
.io-block{font-size:.75rem;color:#8b949e;background:#0d1117;
  border-radius:6px;padding:8px 10px;margin-top:6px;
  max-height:120px;overflow:auto;white-space:pre-wrap;word-break:break-all;
  border:1px solid #21262d}
.footer{margin-top:32px;text-align:center;font-size:.72rem;color:#484f58}
"""

_JS = """\
document.addEventListener('DOMContentLoaded',function(){
  document.querySelectorAll('.collapsible').forEach(function(el){
    el.addEventListener('click',function(){
      var content=el.nextElementSibling;
      if(content&&content.classList.contains('collapsible-content')){
        content.classList.toggle('open');
      }
    });
  });
});
"""


def _score_color(score: float | None) -> str:
    """Return a CSS colour for a score on the 1-5 scale."""
    if score is None:
        return "#484f58"
    if score >= 4.5:
        return "#3fb950"
    if score >= 3.5:
        return "#58a6ff"
    if score >= 2.5:
        return "#d29922"
    if score >= 1.5:
        return "#f0883e"
    return "#f85149"


def _score_percent(score: float | None) -> float:
    """Convert a 1-5 score to a 0-100 percentage."""
    if score is None:
        return 0.0
    return max(0.0, min(100.0, (score - 1.0) / 4.0 * 100.0))


def _esc(text: str | None) -> str:
    """HTML-escape a string."""
    if text is None:
        return ""
    return html.escape(str(text))


def _fmt_latency(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def _build_summary_html(data: dict[str, Any]) -> str:
    s = data["summary"]
    cards = [
        ("Total Steps", str(s["total_steps"])),
        ("Total Latency", _fmt_latency(s["total_latency_ms"])),
        (
            "Avg Score",
            f"{s['average_score']:.1f} / 5" if s["average_score"] is not None else "N/A",
        ),
        ("Errors", str(s["error_count"])),
        ("Scored Steps", str(s["scored_steps"])),
    ]
    parts = ['<div class="summary-grid">']
    for label, value in cards:
        parts.append(
            f'<div class="summary-card">'
            f'<div class="value">{_esc(value)}</div>'
            f'<div class="label">{_esc(label)}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_heatmap_html(steps: list[dict[str, Any]]) -> str:
    """Build a colour-coded score heatmap showing each step."""
    parts = ['<h2>Score Heatmap</h2>', '<div class="heatmap">']
    for step in steps:
        score = step.get("score")
        color = _score_color(score)
        name = step.get("name", "?")
        display_score = f"{score:.1f}" if score is not None else ("ERR" if step.get("error") else "-")
        bg = color if score is not None or step.get("error") else "#21262d"
        parts.append(
            f'<div class="heatmap-cell" style="background:{bg}33;border-color:{bg}" '
            f'title="{_esc(name)}: {display_score}">'
            f'<span class="cell-score">{_esc(display_score)}</span>'
            f'<span class="cell-name">{_esc(name)}</span>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_timeline_html(
    steps: list[dict[str, Any]],
    bottleneck_indices: set[int],
) -> str:
    """Build the step-by-step timeline."""
    parts = ['<h2>Step-by-Step Timeline</h2>', '<div class="timeline">']

    for step in steps:
        idx = step["step_index"]
        is_error = step.get("error") is not None
        is_bn = idx in bottleneck_indices
        cls = "step"
        if is_error:
            cls += " error"
        if is_bn:
            cls += " bottleneck"

        score = step.get("score")
        latency = step.get("latency_ms", 0)
        parent = step.get("parent_span_id")

        parts.append(f'<div class="{cls}">')
        parts.append('<div class="dot"></div>')

        # Header row
        parts.append('<div class="step-header">')
        parts.append(f'<span class="step-name">{_esc(step["name"])}</span>')
        badge_cls = "badge-agent" if step["step_type"] == "agent" else "badge-tool"
        parts.append(
            f'<span class="badge {badge_cls}">{_esc(step["step_type"])}</span>'
        )
        if is_error:
            parts.append('<span class="badge badge-error">error</span>')
        if is_bn:
            parts.append('<span class="badge badge-bottleneck">bottleneck</span>')
        parts.append("</div>")

        # Meta row
        parts.append('<div class="step-meta">')
        parts.append(f"<span>Step #{idx}</span>")
        parts.append(f"<span>{_fmt_latency(latency)}</span>")
        if score is not None:
            parts.append(f"<span>Score: {score:.1f}/5</span>")
        if parent:
            parts.append(f"<span>Parent: {_esc(parent[:8])}...</span>")
        parts.append("</div>")

        # Score bar
        if score is not None:
            pct = _score_percent(score)
            color = _score_color(score)
            parts.append('<div class="score-bar-container">')
            parts.append('<div class="score-bar-track">')
            parts.append(
                f'<div class="score-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
            )
            parts.append("</div>")
            parts.append(
                f'<div class="score-label">{score:.1f} / 5.0</div>'
            )
            parts.append("</div>")

        # Score explanation
        if step.get("score_explanation"):
            parts.append(
                f'<div class="explanation">{_esc(step["score_explanation"])}</div>'
            )

        # Error message
        if is_error:
            parts.append(
                f'<div class="bottleneck-callout">Error: {_esc(step["error"])}</div>'
            )

        # Collapsible I/O
        parts.append('<div class="collapsible">&#9654; Show Input / Output</div>')
        parts.append('<div class="collapsible-content">')
        if step.get("input_data"):
            parts.append(f'<div class="io-block"><strong>Input:</strong>\n{_esc(step["input_data"])}</div>')
        if step.get("output_data"):
            parts.append(f'<div class="io-block"><strong>Output:</strong>\n{_esc(step["output_data"])}</div>')
        parts.append("</div>")

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _build_bottleneck_html(bottlenecks: list[dict[str, Any]]) -> str:
    """Build the bottleneck callouts section."""
    if not bottlenecks:
        return '<h2>Bottleneck Analysis</h2><p style="color:#8b949e">No bottlenecks detected.</p>'

    parts = ['<h2>Bottleneck Analysis</h2>']
    for bn in bottlenecks:
        color = _score_color(bn["score"])
        impact = bn["impact"]
        parts.append(
            f'<div class="bottleneck-callout" style="margin-bottom:10px">'
            f'<strong>{_esc(bn["step_name"])}</strong> '
            f'<span class="badge badge-bottleneck">{_esc(impact)}</span> '
            f'&mdash; Score: {bn["score"]:.1f}/5, '
            f'Latency: {_fmt_latency(bn["latency_ms"])}<br>'
            f'{_esc(bn["explanation"])}'
            f"</div>"
        )
    return "\n".join(parts)


def export_html(session: TraceSession) -> str:
    """Return a self-contained HTML report for the trace session."""
    data = _session_to_dict(session)
    bottleneck_indices = {b["step_index"] for b in data["bottlenecks"]}

    summary = _build_summary_html(data)
    heatmap = _build_heatmap_html(data["steps"])
    timeline = _build_timeline_html(data["steps"], bottleneck_indices)
    bottlenecks = _build_bottleneck_html(data["bottlenecks"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace Report &mdash; {_esc(data["trace_id"])}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Trace Report</h1>
<div class="subtitle">Trace ID: {_esc(data["trace_id"])}</div>
{summary}
{heatmap}
{bottlenecks}
{timeline}
<div class="footer">Generated by agent-trace</div>
<script>{_JS}</script>
</body>
</html>"""
