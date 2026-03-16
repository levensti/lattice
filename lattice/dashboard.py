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

from .storage.store import traces


def _score_bar(score: float | None) -> str:
    """Render a colored score as inline HTML."""
    if score is None:
        return '<span style="color:#888">—</span>'
    if score >= 4:
        color = "#22c55e"
    elif score >= 3:
        color = "#eab308"
    else:
        color = "#ef4444"
    return f'<span style="color:{color};font-weight:600">{score:.1f}/5</span>'


def _build_action_rows(actions) -> str:
    """Build HTML table rows for actions."""
    rows = []
    for s in actions:
        status = "ERROR" if s.error else "OK"
        status_color = "#ef4444" if s.error else "#22c55e"
        role = html.escape(s.role or "—")
        name = html.escape(s.name)
        error_detail = ""
        if s.error:
            error_detail = (
                f'<div style="color:#ef4444;font-size:12px;margin-top:2px">'
                f'{html.escape(s.error[:200])}</div>'
            )
        score_explanation = ""
        if s.score_explanation:
            score_explanation = (
                f'<div style="color:#888;font-size:12px;margin-top:2px">'
                f'{html.escape(s.score_explanation[:200])}</div>'
            )
        rows.append(f"""
            <tr>
                <td>{s.action_index}</td>
                <td>{name}{error_detail}</td>
                <td>{role}</td>
                <td><span style="color:{status_color}">{status}</span></td>
                <td>{s.latency_ms:.0f}ms</td>
                <td>{_score_bar(s.score)}{score_explanation}</td>
            </tr>
        """)
    return "\n".join(rows)


def _build_trace_detail(session) -> str:
    """Build the HTML detail view for a single trace."""
    action_rows = _build_action_rows(session.actions)
    total_ms = sum(s.latency_ms for s in session.actions)
    session_score = _score_bar(session.session_score)
    explanation = ""
    if session.session_score_explanation:
        explanation = (
            f'<p style="color:#888;margin-top:4px">'
            f'{html.escape(session.session_score_explanation)}</p>'
        )

    transitions_html = ""
    if session.transitions:
        t_rows = []
        for t in session.transitions:
            if t.auto and not t.reason:
                continue
            from_name = t.from_span_id or "start"
            from_action = next(
                (s for s in session.actions if s.span_id == t.from_span_id), None
            )
            if from_action:
                from_name = from_action.name
            reason = html.escape(t.reason) if t.reason else "—"
            t_rows.append(
                f"<tr><td>{html.escape(from_name)}</td>"
                f"<td>{html.escape(t.to_name)}</td>"
                f"<td>{reason}</td></tr>"
            )
        if t_rows:
            transitions_html = f"""
            <h3>Transitions</h3>
            <table>
                <tr><th>From</th><th>To</th><th>Reason</th></tr>
                {"".join(t_rows)}
            </table>
            """

    groups_html = ""
    if session.groups:
        g_rows = []
        for g in session.groups:
            symbol = "&#10227;" if g.group_type == "loop" else "&#8741;"
            group_actions = [s for s in session.actions if s.group_id == g.group_id]
            g_rows.append(
                f"<tr><td>{symbol} {html.escape(g.name)}</td>"
                f"<td>{g.group_type}</td>"
                f"<td>{len(group_actions)}</td></tr>"
            )
        groups_html = f"""
        <h3>Groups</h3>
        <table>
            <tr><th>Name</th><th>Type</th><th>Actions</th></tr>
            {"".join(g_rows)}
        </table>
        """

    return f"""
    <div class="detail">
        <h2>{html.escape(session.workflow_name or 'Unnamed Workflow')}</h2>
        <p><strong>Trace ID:</strong> <code>{session.trace_id}</code></p>
        <p><strong>Goal:</strong> {html.escape(session.goal)}</p>
        <p><strong>Actions:</strong> {len(session.actions)} &middot;
           <strong>Total time:</strong> {total_ms:.0f}ms &middot;
           <strong>Session score:</strong> {session_score}</p>
        {explanation}

        <h3>Actions</h3>
        <table>
            <tr>
                <th>#</th><th>Name</th><th>Role</th>
                <th>Status</th><th>Latency</th><th>Score</th>
            </tr>
            {action_rows}
        </table>
        {groups_html}
        {transitions_html}
    </div>
    """


def _build_trace_list_item(session) -> str:
    """Build a sidebar list item for a trace."""
    name = html.escape(session.workflow_name or "Unnamed")
    action_count = len(session.actions)
    score = _score_bar(session.session_score)
    created = getattr(session, "_created_at", "")
    if created:
        # Show just date and time
        created = created[:19].replace("T", " ")
    return f"""
    <a class="trace-item" href="/?trace_id={session.trace_id}">
        <div class="trace-name">{name}</div>
        <div class="trace-meta">{action_count} actions &middot; {score}</div>
        <div class="trace-date">{created}</div>
    </a>
    """


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; display: flex; height: 100vh;
}
.sidebar {
    width: 300px; background: #1e293b; border-right: 1px solid #334155;
    overflow-y: auto; flex-shrink: 0;
}
.sidebar h1 {
    padding: 16px; font-size: 18px; border-bottom: 1px solid #334155;
    color: #38bdf8;
}
.trace-item {
    display: block; padding: 12px 16px; border-bottom: 1px solid #334155;
    text-decoration: none; color: inherit; transition: background 0.15s;
}
.trace-item:hover { background: #334155; }
.trace-name { font-weight: 600; font-size: 14px; }
.trace-meta { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.trace-date { font-size: 11px; color: #64748b; margin-top: 2px; }
.main { flex: 1; overflow-y: auto; padding: 24px; }
.detail h2 { color: #38bdf8; margin-bottom: 12px; }
.detail h3 { color: #94a3b8; margin: 20px 0 8px; font-size: 14px; text-transform: uppercase; }
.detail p { margin: 4px 0; font-size: 14px; }
code {
    background: #334155; padding: 2px 6px; border-radius: 4px;
    font-size: 13px; color: #38bdf8;
}
table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin-top: 4px;
}
th {
    text-align: left; padding: 8px 12px; background: #1e293b;
    color: #94a3b8; font-weight: 500; border-bottom: 1px solid #334155;
}
td {
    padding: 8px 12px; border-bottom: 1px solid #1e293b;
    vertical-align: top;
}
tr:hover td { background: #1e293b; }
.empty {
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: #64748b; font-size: 16px;
}
"""


def _render_page(trace_id: str | None = None) -> str:
    """Render the full dashboard HTML page."""
    all_traces = traces()

    sidebar_items = "\n".join(_build_trace_list_item(t) for t in all_traces)
    if not sidebar_items:
        sidebar_items = '<div class="trace-item"><div class="trace-meta">No traces yet</div></div>'

    # Detail pane
    if trace_id:
        matching = [t for t in all_traces if t.trace_id == trace_id]
        if matching:
            detail = _build_trace_detail(matching[0])
        else:
            detail = '<div class="empty">Trace not found</div>'
    elif all_traces:
        detail = _build_trace_detail(all_traces[0])
    else:
        detail = '<div class="empty">Run a traced workflow to see results here</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Lattice Dashboard</title>
    <style>{_CSS}</style>
</head>
<body>
    <div class="sidebar">
        <h1>Lattice Traces</h1>
        {sidebar_items}
    </div>
    <div class="main">
        {detail}
    </div>
</body>
</html>"""


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/traces":
            self._serve_json()
            return

        # HTML dashboard
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
        # Quieter logs — only show in debug
        pass


def serve(*, host: str = "127.0.0.1", port: int = 8787) -> None:
    """Start the local Lattice dashboard server.

    Args:
        host: Bind address (default ``127.0.0.1``).
        port: Port number (default ``8787``).
    """
    server = HTTPServer((host, port), _DashboardHandler)
    print(f"Lattice dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
