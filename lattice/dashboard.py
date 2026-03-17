"""Local web dashboard for browsing Lattice traces.

Launch with::

    python -m lattice dashboard
    python -m lattice dashboard --port 8080

No dependencies beyond the standard library. Reads traces from the
local SQLite store and serves a single-page HTML viewer.
"""

from __future__ import annotations

import html
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .bottleneck import find_bottlenecks
from .storage.store import traces


# ── score helpers ──────────────────────────────────────────────────────


def _score_bar(score: float | None) -> str:
    if score is None:
        return '<span class="score-none">&mdash;</span>'
    if score >= 4:
        color = "#16a34a"
    elif score >= 3:
        color = "#ca8a04"
    else:
        color = "#dc2626"
    return f'<span class="score" style="color:{color}">{score:.1f}<span class="score-max">/5</span></span>'


def _score_pill(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 4:
        cls = "pill-green"
    elif score >= 3:
        cls = "pill-yellow"
    else:
        cls = "pill-red"
    return f'<span class="score-pill {cls}">{score:.1f}</span>'


# ── expandable content ─────────────────────────────────────────────────


def _expandable(content: str, max_preview: int = 100) -> str:
    if not content:
        return '<span class="muted">&mdash;</span>'
    escaped = html.escape(content)
    preview = html.escape(content[:max_preview])
    if len(content) > max_preview:
        preview += "..."
    return (
        f'<details><summary>{preview}</summary>'
        f'<pre class="expandable-pre">{escaped}</pre></details>'
    )


# ── evaluation section (judge config + results) ──────────────────────


def _judge_model_badge(judge_config) -> str:
    """Render a compact badge showing the judge model, e.g. 'gpt-4o'."""
    if not judge_config:
        return ""
    model = str(judge_config.get("model", "") or "")
    if not model:
        model = judge_config.get("provider", "judge")
    return f'<span class="judge-badge">{html.escape(model)}</span>'


def _rubric_section(judge_config) -> str:
    """Render expandable rubric if judge config has a system prompt."""
    if not judge_config:
        return ""
    prompt = judge_config.get("system_prompt", "")
    if not prompt:
        return ""
    return (
        f'<div class="rubric-section">'
        f'<details class="eval-rubric">'
        f'<summary>View rubric</summary>'
        f'<pre class="expandable-pre">{html.escape(prompt)}</pre>'
        f'</details>'
        f'</div>'
    )


# ── horizontal tree ──────────────────────────────────────────────────


def _role_badge(role: str | None) -> str:
    if not role:
        return ""
    return f'<span class="role-badge">{html.escape(role)}</span>'


def _status_dot(error: str | None) -> str:
    if error:
        return '<span class="dot dot-err" title="Error"></span>'
    return '<span class="dot dot-ok" title="OK"></span>'


def _latency_fmt(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def _clean_name(name: str) -> str:
    """Strip [parallel] and [loop] prefixes from action names."""
    if name.startswith("[parallel] "):
        return name[11:]
    if name.startswith("[loop] "):
        return name[7:]
    return name


def _render_node_card(action) -> str:
    """Render a compact card for a single action node."""
    name_esc = html.escape(_clean_name(action.name))
    role = _role_badge(action.role)
    dot = _status_dot(action.error)
    latency = _latency_fmt(action.latency_ms)
    score = _score_pill(action.score)
    judge_config = getattr(action, "judge_config", None)
    judge_badge = _judge_model_badge(judge_config)

    error_html = ""
    if action.error:
        error_html = f'<div class="hnode-error">{html.escape(action.error[:200])}</div>'

    tags_html = ""
    if getattr(action, "tags", []):
        tags_html = '<div class="hnode-tags">' + " ".join(
            f'<span class="tag">{html.escape(t)}</span>' for t in action.tags
        ) + "</div>"

    rubric_html = _rubric_section(judge_config)

    io_html = ""
    if action.input_data or action.output_data:
        io_parts = []
        if action.input_data:
            io_parts.append(
                f'<div class="io-block"><span class="io-label">In</span>'
                f'{_expandable(action.input_data, 60)}</div>'
            )
        if action.output_data:
            io_parts.append(
                f'<div class="io-block"><span class="io-label">Out</span>'
                f'{_expandable(action.output_data, 60)}</div>'
            )
        io_html = f'<div class="hnode-io">{"".join(io_parts)}</div>'

    err_cls = " hnode-err" if action.error else ""
    return (
        f'<div class="hnode{err_cls}">'
        f'<div class="hnode-head">'
        f'<div class="hnode-title">{dot}<span class="hnode-name">{name_esc}</span>{role}</div>'
        f'</div>'
        f'<div class="hnode-meta-row">{score}{judge_badge}<span class="hnode-latency">{latency}</span></div>'
        f'{error_html}'
        f'{tags_html}'
        f'{rubric_html}'
        f'{io_html}'
        f'</div>'
    )


def _build_h_tree(actions, groups=None) -> str:
    """Build horizontal left-to-right process tree from actions."""
    if not actions:
        return '<div class="muted">No actions</div>'
    groups = groups or []
    group_map = {g.group_id: g for g in groups}
    span_ids = {a.span_id for a in actions}
    by_parent: dict[str, list] = {}
    roots = []
    for a in actions:
        if a.parent_span_id is None or a.parent_span_id not in span_ids:
            roots.append(a)
        else:
            by_parent.setdefault(a.parent_span_id, []).append(a)

    def render_subtree(action) -> str:
        card = _render_node_card(action)
        children = by_parent.get(action.span_id, [])
        if not children:
            return card

        child_gids = {c.group_id for c in children if c.group_id}
        is_parallel = False
        group_label = ""
        for gid in child_gids:
            g = group_map.get(gid)
            if g:
                is_parallel = g.group_type == "parallel"
                icon = "\u2225" if is_parallel else "\u21bb"
                label_text = html.escape(g.name)
                label_cls = "parallel" if is_parallel else "loop"
                group_label = (
                    f'<div class="htree-group-label htree-group-{label_cls}">'
                    f'<span class="htree-group-icon">{icon}</span>'
                    f'{label_text}'
                    f'</div>'
                )
                break

        branches = []
        n = len(children)
        for i, child in enumerate(children):
            if n == 1:
                pos = "only"
            elif i == 0:
                pos = "first"
            elif i == n - 1:
                pos = "last"
            else:
                pos = "mid"
            branches.append(
                f'<div class="htree-branch htree-branch-{pos}">'
                f'<div class="htree-arm"></div>'
                f'{render_subtree(child)}'
                f'</div>'
            )

        children_cls = " htree-children-parallel" if is_parallel else ""
        return (
            f'<div class="htree-node-group">'
            f'{card}'
            f'<div class="htree-connector"></div>'
            f'<div class="htree-children{children_cls}">{group_label}{"".join(branches)}</div>'
            f'</div>'
        )

    parts = [render_subtree(r) for r in roots]
    return f'<div class="htree">{"".join(parts)}</div>'


# ── trace detail ───────────────────────────────────────────────────────


def _build_trace_detail(session) -> str:
    tree_html = _build_h_tree(session.actions, session.groups)
    total_ms = sum(s.latency_ms for s in session.actions)
    action_count = len(session.actions)
    score = _score_bar(session.session_score)

    explanation = ""
    if session.session_score_explanation:
        explanation = f'<p class="session-explanation">{html.escape(session.session_score_explanation)}</p>'

    # Stats bar
    error_count = sum(1 for a in session.actions if a.error)
    avg_score_actions = [a for a in session.actions if a.score is not None]
    avg_score = sum(a.score for a in avg_score_actions) / len(avg_score_actions) if avg_score_actions else None

    stats = f"""
    <div class="stats-bar">
        <div class="stat"><div class="stat-value">{action_count}</div><div class="stat-label">Actions</div></div>
        <div class="stat"><div class="stat-value">{_latency_fmt(total_ms)}</div><div class="stat-label">Total time</div></div>
        <div class="stat"><div class="stat-value">{score}</div><div class="stat-label">Session score</div></div>
        <div class="stat"><div class="stat-value">{_score_bar(avg_score)}</div><div class="stat-label">Avg action score</div></div>
        {'<div class="stat"><div class="stat-value stat-err">' + str(error_count) + '</div><div class="stat-label">Errors</div></div>' if error_count else ''}
    </div>
    """

    return f"""
    <div class="detail">
        <div class="detail-header">
            <h2>{html.escape(session.workflow_name or 'Unnamed Workflow')}</h2>
            <code class="trace-id">{session.trace_id}</code>
        </div>
        <p class="goal-text">{html.escape(session.goal)}</p>
        {explanation}
        {stats}

        <div class="section">
            <h3>Action Tree</h3>
            <div class="tree-container">{tree_html}</div>
        </div>
        {_build_bottleneck_section(session)}
    </div>
    """


# ── sidebar ────────────────────────────────────────────────────────────


def _build_trace_list_item(session, is_active: bool = False) -> str:
    name = html.escape(session.workflow_name or "Unnamed")
    action_count = len(session.actions)
    score = _score_pill(session.session_score)
    created = session.created_at or ""
    if created:
        created = created[:19].replace("T", " ")
    error_count = sum(1 for a in session.actions if a.error)
    error_badge = f'<span class="sidebar-err">{error_count} err</span>' if error_count else ""
    active_cls = " active" if is_active else ""
    return f"""
    <a class="trace-item{active_cls}" href="/?trace_id={session.trace_id}">
        <div class="trace-item-top">
            <span class="trace-name">{name}</span>
            {score}
        </div>
        <div class="trace-meta">{action_count} actions {error_badge}</div>
        <div class="trace-date">{created}</div>
    </a>
    """


# ── bottlenecks ────────────────────────────────────────────────────────


_IMPACT_COLORS = {
    "error": "#dc2626",
    "loop_no_convergence": "#ca8a04",
    "weakest_branch": "#ea580c",
}


def _build_bottleneck_section(session) -> str:
    bottlenecks = find_bottlenecks(session)
    if not bottlenecks:
        return ""
    items = []
    for b in bottlenecks:
        color = _IMPACT_COLORS.get(b.impact, "#a1a1aa")
        items.append(
            f'<div class="bottleneck-item">'
            f'<span class="bottleneck-name">{html.escape(b.action_name)}</span>'
            f'<span class="bottleneck-impact" style="color:{color}">{html.escape(b.impact)}</span>'
            f'{_score_pill(b.score)}'
            f'<span class="muted">{b.latency_ms:.0f}ms</span>'
            f'<span class="bottleneck-expl">{html.escape(b.explanation)}</span>'
            f'</div>'
        )
    return f"""
    <div class="section">
        <h3>Bottlenecks</h3>
        <div class="bottleneck-list">{"".join(items)}</div>
    </div>
    """


# ── CSS ────────────────────────────────────────────────────────────────


_CSS = """
:root {
    --bg: #ffffff;
    --surface: #fafafa;
    --surface-2: #f4f4f5;
    --surface-3: #e4e4e7;
    --border: #e4e4e7;
    --border-light: #d4d4d8;
    --text: #09090b;
    --text-secondary: #52525b;
    --text-muted: #a1a1aa;
    --accent: #000000;
    --green: #16a34a;
    --yellow: #ca8a04;
    --red: #dc2626;
    --parallel-accent: #6366f1;
    --parallel-bg: rgba(99,102,241,0.04);
    --parallel-border: rgba(99,102,241,0.18);
    --radius: 8px;
    --radius-sm: 4px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    height: 100vh;
    -webkit-font-smoothing: antialiased;
    font-size: 14px;
    line-height: 1.5;
}

/* ── Sidebar ───────────────────────────────────────────────── */

.sidebar {
    width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
}

.sidebar-header {
    padding: 20px 16px 16px;
    border-bottom: 1px solid var(--border);
}

.sidebar-header h1 {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-secondary);
}

.trace-item {
    display: block;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    text-decoration: none;
    color: inherit;
    transition: background 0.12s ease;
}
.trace-item:hover { background: var(--surface-2); }
.trace-item.active { background: var(--surface-3); border-left: 2px solid var(--text); }

.trace-item-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.trace-name { font-weight: 500; font-size: 13px; }
.trace-meta { font-size: 11px; color: var(--text-muted); margin-top: 4px; display: flex; align-items: center; gap: 6px; }
.trace-date { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

.sidebar-err {
    background: rgba(220,38,38,0.08);
    color: var(--red);
    font-size: 10px;
    padding: 1px 5px;
    border-radius: var(--radius-sm);
    font-weight: 500;
}

/* ── Main ──────────────────────────────────────────────────── */

.main { flex: 1; overflow-y: auto; padding: 32px 40px; }

.detail-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 4px;
}
.detail h2 {
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.trace-id {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--surface-2);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    font-family: "SF Mono", "Fira Code", monospace;
}
.goal-text {
    color: var(--text-secondary);
    font-size: 14px;
    margin: 4px 0 0;
    line-height: 1.6;
}
.session-explanation {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 4px;
}

/* ── Stats bar ─────────────────────────────────────────────── */

.stats-bar {
    display: flex;
    gap: 1px;
    background: var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin: 20px 0;
}
.stat {
    flex: 1;
    background: var(--surface);
    padding: 14px 16px;
    text-align: center;
}
.stat-value { font-size: 18px; font-weight: 600; letter-spacing: -0.02em; }
.stat-label { font-size: 11px; color: var(--text-muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-err { color: var(--red); }

/* ── Sections ──────────────────────────────────────────────── */

.section { margin-top: 28px; }
.section h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Horizontal Tree ──────────────────────────────────────── */

.tree-container {
    overflow-x: auto;
    padding: 8px 0 16px;
}

.htree {
    display: inline-flex;
    align-items: flex-start;
    padding: 8px 0;
    min-width: max-content;
}

.htree-node-group {
    display: flex;
    align-items: flex-start;
}

.htree-connector {
    width: 28px;
    min-width: 28px;
    flex-shrink: 0;
    position: relative;
    align-self: stretch;
}
.htree-connector::after {
    content: '';
    position: absolute;
    top: 22px;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--border-light);
}

.htree-children {
    display: flex;
    flex-direction: column;
    position: relative;
    gap: 0;
}

/* Parallel group visual container */
.htree-children-parallel {
    background: var(--parallel-bg);
    border: 1px dashed var(--parallel-border);
    border-radius: var(--radius);
    padding: 6px 8px 6px 4px;
}

.htree-group-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    padding: 0 0 4px 24px;
    display: flex;
    align-items: center;
    gap: 4px;
    font-weight: 600;
    white-space: nowrap;
}

.htree-group-parallel {
    color: var(--parallel-accent);
}
.htree-group-parallel .htree-group-icon {
    color: var(--parallel-accent);
}

.htree-group-icon {
    font-size: 13px;
    color: var(--text-secondary);
}

.htree-branch {
    display: flex;
    align-items: center;
    position: relative;
    padding: 5px 0;
}

/* Vertical trunk line */
.htree-branch::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border-light);
}
.htree-branch-first::before { top: 50%; }
.htree-branch-last::before { bottom: 50%; }
.htree-branch-only::before { display: none; }

/* Parallel branches get accent-colored connectors */
.htree-children-parallel .htree-branch::before {
    background: var(--parallel-border);
}
.htree-children-parallel .htree-arm {
    background: var(--parallel-border);
}

/* Horizontal arm from trunk to child */
.htree-arm {
    width: 20px;
    min-width: 20px;
    height: 2px;
    background: var(--border-light);
    flex-shrink: 0;
}

/* ── Node card ─────────────────────────────────────────────── */

.hnode {
    min-width: 200px;
    max-width: 280px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 12px;
    flex-shrink: 0;
    transition: border-color 0.12s ease;
}
.hnode:hover { border-color: var(--border-light); }
.hnode-err { border-color: rgba(220,38,38,0.25); }
.hnode-err:hover { border-color: rgba(220,38,38,0.4); }

.hnode-head {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
}
.hnode-title {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
}
.hnode-name {
    font-weight: 600;
    font-size: 13px;
    white-space: nowrap;
}
.hnode-meta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 3px;
}
.hnode-latency {
    font-size: 11px;
    color: var(--text-muted);
    font-family: "SF Mono", "Fira Code", monospace;
}

.hnode-error {
    color: var(--red);
    font-size: 11px;
    margin-top: 6px;
    padding: 4px 6px;
    background: rgba(220,38,38,0.04);
    border-radius: var(--radius-sm);
    border-left: 2px solid var(--red);
    word-break: break-word;
}

.hnode-tags { display: flex; gap: 3px; flex-wrap: wrap; margin-top: 5px; }
.hnode-io { margin-top: 6px; }

/* ── Judge ──────────────────────────────────────────────────── */

.judge-badge {
    display: inline-block;
    font-size: 10px;
    padding: 1px 5px;
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: "SF Mono", "Fira Code", monospace;
    font-weight: 500;
    white-space: nowrap;
}

.rubric-section {
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
}
.eval-rubric summary {
    font-size: 10px;
    color: var(--text-muted);
    cursor: pointer;
    letter-spacing: 0.03em;
}

/* ── Tags & pills ──────────────────────────────────────────── */

.tag {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--surface-3);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    font-weight: 500;
}

.role-badge {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: rgba(0,0,0,0.04);
    color: var(--text-secondary);
    border: 1px solid var(--border-light);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.score-pill {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    font-family: "SF Mono", "Fira Code", monospace;
}
.pill-green { background: rgba(22,163,74,0.08); color: var(--green); }
.pill-yellow { background: rgba(202,138,4,0.08); color: var(--yellow); }
.pill-red { background: rgba(220,38,38,0.08); color: var(--red); }

/* ── Score ──────────────────────────────────────────────────── */

.score { font-weight: 600; font-family: "SF Mono", "Fira Code", monospace; }
.score-max { font-size: 0.75em; color: var(--text-muted); font-weight: 400; }
.score-none { color: var(--text-muted); }

/* ── Dot ───────────────────────────────────────────────────── */

.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.dot-ok { background: var(--green); }
.dot-err { background: var(--red); box-shadow: 0 0 6px rgba(220,38,38,0.3); }

/* ── IO blocks ─────────────────────────────────────────────── */

.io-block { min-width: 0; }
.io-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    display: block;
    margin-bottom: 2px;
}

/* ── Expandable ────────────────────────────────────────────── */

details { cursor: pointer; }
details summary {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
}
.expandable-pre {
    margin-top: 6px;
    padding: 10px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-all;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    font-family: "SF Mono", "Fira Code", monospace;
}

/* ── Bottlenecks ───────────────────────────────────────────── */

.bottleneck-list { display: flex; flex-direction: column; gap: 4px; }
.bottleneck-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
}
.bottleneck-name { font-weight: 500; min-width: 120px; }
.bottleneck-impact { font-weight: 600; font-size: 11px; }
.bottleneck-expl { color: var(--text-muted); font-size: 11px; margin-left: auto; }

.muted { color: var(--text-muted); }

/* ── Empty state ───────────────────────────────────────────── */

.empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    font-size: 14px;
}

/* ── Scrollbar ─────────────────────────────────────────────── */

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* ── Table (for bottlenecks fallback) ──────────────────────── */

table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin-top: 4px;
}
th {
    text-align: left; padding: 8px 12px; background: var(--surface);
    color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
}
td {
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    vertical-align: top;
}
tr:hover td { background: var(--surface-2); }
"""


# ── page assembly ──────────────────────────────────────────────────────


def _render_page(trace_id: str | None = None) -> str:
    all_traces = traces()

    active_id = trace_id
    if not active_id and all_traces:
        active_id = all_traces[0].trace_id

    sidebar_items = "\n".join(
        _build_trace_list_item(t, is_active=(t.trace_id == active_id))
        for t in all_traces
    )
    if not sidebar_items:
        sidebar_items = '<div class="trace-item"><div class="trace-meta">No traces yet</div></div>'

    if active_id:
        matching = [t for t in all_traces if t.trace_id == active_id]
        if matching:
            detail = _build_trace_detail(matching[0])
        else:
            detail = '<div class="empty">Trace not found</div>'
    else:
        detail = '<div class="empty">Run a traced workflow to see results here</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Lattice Dashboard</title>
    <style>{_CSS}</style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>Lattice</h1>
        </div>
        {sidebar_items}
    </div>
    <div class="main">
        {detail}
    </div>
</body>
</html>"""


# ── HTTP server ────────────────────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/traces":
            self._serve_json()
            return

        params = parse_qs(parsed.query)
        trace_id = params.get("trace_id", [None])[0]
        body = _render_page(trace_id).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self):
        all_traces = traces()
        data = [t.to_dict() for t in all_traces]
        body = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def serve(*, host: str = "localhost", port: int = 8080) -> None:
    """Start the local Lattice dashboard server.

    Args:
        host: Bind address (default ``localhost``).
        port: Port number (default ``8080``).
    """
    server = HTTPServer((host, port), _DashboardHandler)
    print(f"Lattice dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
