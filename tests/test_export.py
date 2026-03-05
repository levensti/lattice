"""Tests for export_json() and export_html() on TraceSession."""

from __future__ import annotations

import json

from agent_trace.bottleneck import find_bottlenecks
from agent_trace.context import StepRecord, TraceSession
from agent_trace.export import export_html, export_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(
    *,
    span_id: str = "aaa",
    name: str = "step",
    step_type: str = "agent",
    description: str = "",
    criteria: str = "",
    input_data: str = '{"q": "hello"}',
    output_data: str = '"world"',
    step_index: int = 0,
    latency_ms: float = 100.0,
    parent_span_id: str | None = None,
    score: float | None = None,
    score_explanation: str | None = None,
    error: str | None = None,
) -> StepRecord:
    return StepRecord(
        span_id=span_id,
        name=name,
        step_type=step_type,
        description=description,
        criteria=criteria,
        input_data=input_data,
        output_data=output_data,
        step_index=step_index,
        latency_ms=latency_ms,
        parent_span_id=parent_span_id,
        score=score,
        score_explanation=score_explanation,
        error=error,
    )


def _populated_session() -> TraceSession:
    """Build a session with a realistic mix of steps."""
    session = TraceSession(trace_id="test-trace-001")
    session.add_step(_step(
        span_id="s1", name="researcher", step_type="agent",
        input_data='{"topic": "AI"}', output_data='"research results"',
        step_index=0, latency_ms=250.0, score=4.5,
        score_explanation="Thorough research with good sources.",
    ))
    session.add_step(_step(
        span_id="s2", name="web_search", step_type="tool",
        input_data='{"query": "AI trends"}', output_data='"search results"',
        step_index=1, latency_ms=800.0, parent_span_id="s1",
        score=3.0, score_explanation="Acceptable but limited coverage.",
    ))
    session.add_step(_step(
        span_id="s3", name="writer", step_type="agent",
        input_data='{"data": "research results"}', output_data='"draft article"',
        step_index=2, latency_ms=500.0, score=1.5,
        score_explanation="Major quality issues in the draft.",
    ))
    session.add_step(_step(
        span_id="s4", name="editor", step_type="agent",
        input_data='{"draft": "draft article"}', output_data='"final article"',
        step_index=3, latency_ms=300.0, score=4.0,
        score_explanation="Good editing with minor issues.",
    ))
    return session


# ---------------------------------------------------------------------------
# export_json tests
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_returns_valid_json(self):
        session = _populated_session()
        result = export_json(session)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_trace_id(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        assert data["trace_id"] == "test-trace-001"

    def test_summary_fields(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        summary = data["summary"]
        assert summary["total_steps"] == 4
        assert summary["error_count"] == 0
        assert summary["scored_steps"] == 4
        assert summary["total_latency_ms"] > 0
        assert summary["average_score"] is not None

    def test_summary_average_score(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        expected_avg = round((4.5 + 3.0 + 1.5 + 4.0) / 4, 2)
        assert data["summary"]["average_score"] == expected_avg

    def test_steps_list(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        assert len(data["steps"]) == 4
        assert data["steps"][0]["name"] == "researcher"
        assert data["steps"][2]["score"] == 1.5

    def test_bottlenecks_present(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        assert "bottlenecks" in data
        assert isinstance(data["bottlenecks"], list)
        # writer (score 1.5) should be the worst bottleneck
        assert len(data["bottlenecks"]) > 0
        assert data["bottlenecks"][0]["step_name"] == "writer"

    def test_empty_session(self):
        session = TraceSession(trace_id="empty")
        data = json.loads(export_json(session))
        assert data["summary"]["total_steps"] == 0
        assert data["summary"]["average_score"] is None
        assert data["steps"] == []
        assert data["bottlenecks"] == []

    def test_session_with_error_step(self):
        session = TraceSession(trace_id="err-session")
        session.add_step(_step(
            span_id="e1", name="failing_tool", step_type="tool",
            step_index=0, latency_ms=50.0, error="Connection timeout",
        ))
        data = json.loads(export_json(session))
        assert data["summary"]["error_count"] == 1
        assert data["steps"][0]["error"] == "Connection timeout"
        assert len(data["bottlenecks"]) == 1
        assert data["bottlenecks"][0]["impact"] == "error"

    def test_unscored_steps(self):
        session = TraceSession(trace_id="unscored")
        session.add_step(_step(span_id="u1", name="step_a", step_index=0))
        session.add_step(_step(span_id="u2", name="step_b", step_index=1))
        data = json.loads(export_json(session))
        assert data["summary"]["scored_steps"] == 0
        assert data["summary"]["average_score"] is None

    def test_indent_parameter(self):
        session = _populated_session()
        compact = export_json(session, indent=None)
        pretty = export_json(session, indent=4)
        # compact has no leading spaces; pretty does
        assert "\n    " in pretty
        # Both parse to the same data
        assert json.loads(compact) == json.loads(pretty)

    def test_parent_span_id_preserved(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        web_search = data["steps"][1]
        assert web_search["parent_span_id"] == "s1"

    def test_session_method_matches_function(self):
        session = _populated_session()
        assert session.export_json() == export_json(session)

    def test_step_fields_complete(self):
        session = _populated_session()
        data = json.loads(export_json(session))
        step = data["steps"][0]
        expected_keys = {
            "span_id", "name", "step_type", "description", "criteria",
            "input_data", "output_data", "step_index", "latency_ms",
            "parent_span_id", "score", "score_explanation", "error",
        }
        assert set(step.keys()) == expected_keys


# ---------------------------------------------------------------------------
# export_html tests
# ---------------------------------------------------------------------------

class TestExportHtml:
    def test_returns_string(self):
        session = _populated_session()
        result = export_html(session)
        assert isinstance(result, str)

    def test_is_valid_html_document(self):
        session = _populated_session()
        result = export_html(session)
        assert result.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in result

    def test_contains_trace_id(self):
        session = _populated_session()
        result = export_html(session)
        assert "test-trace-001" in result

    def test_contains_step_names(self):
        session = _populated_session()
        result = export_html(session)
        assert "researcher" in result
        assert "web_search" in result
        assert "writer" in result
        assert "editor" in result

    def test_contains_scores(self):
        session = _populated_session()
        result = export_html(session)
        assert "4.5" in result
        assert "3.0" in result
        assert "1.5" in result
        assert "4.0" in result

    def test_contains_heatmap_section(self):
        session = _populated_session()
        result = export_html(session)
        assert "Score Heatmap" in result
        assert "heatmap" in result

    def test_contains_timeline_section(self):
        session = _populated_session()
        result = export_html(session)
        assert "Step-by-Step Timeline" in result
        assert "timeline" in result

    def test_contains_bottleneck_section(self):
        session = _populated_session()
        result = export_html(session)
        assert "Bottleneck Analysis" in result

    def test_bottleneck_badge_for_low_score(self):
        session = _populated_session()
        result = export_html(session)
        # writer has score 1.5, should be flagged as bottleneck
        assert "bottleneck" in result.lower()

    def test_error_step_rendering(self):
        session = TraceSession(trace_id="err-html")
        session.add_step(_step(
            span_id="e1", name="broken_tool", step_type="tool",
            step_index=0, latency_ms=50.0, error="Timeout error",
        ))
        result = export_html(session)
        assert "broken_tool" in result
        assert "Timeout error" in result
        assert "ERR" in result  # heatmap shows ERR for error steps

    def test_empty_session_renders(self):
        session = TraceSession(trace_id="empty-html")
        result = export_html(session)
        assert "<!DOCTYPE html>" in result
        assert "empty-html" in result
        assert "No bottlenecks detected" in result

    def test_self_contained_no_external_links(self):
        session = _populated_session()
        result = export_html(session)
        # Should not reference external CSS or JS
        assert 'href="http' not in result
        assert 'src="http' not in result
        # CSS and JS should be inline
        assert "<style>" in result
        assert "<script>" in result

    def test_html_escaping(self):
        session = TraceSession(trace_id="<script>alert('xss')</script>")
        session.add_step(_step(
            span_id="x1", name='<img onerror="alert(1)">',
            step_type="agent", step_index=0,
            input_data='<script>bad</script>',
            output_data='"safe"',
            score=3.0, score_explanation='Test "quotes" & <tags>',
        ))
        result = export_html(session)
        # Raw HTML should be escaped
        assert "<script>alert" not in result
        assert "&lt;script&gt;" in result
        assert "&amp;" in result

    def test_session_method_matches_function(self):
        session = _populated_session()
        assert session.export_html() == export_html(session)

    def test_latency_formatting(self):
        session = TraceSession(trace_id="latency-test")
        session.add_step(_step(
            span_id="l1", name="fast_step", step_index=0,
            latency_ms=50.0, score=4.0,
        ))
        session.add_step(_step(
            span_id="l2", name="slow_step", step_index=1,
            latency_ms=2500.0, score=3.0,
        ))
        result = export_html(session)
        assert "50ms" in result
        assert "2.50s" in result

    def test_unscored_steps_in_heatmap(self):
        session = TraceSession(trace_id="unscored-hm")
        session.add_step(_step(
            span_id="u1", name="unscored_step", step_index=0,
            latency_ms=100.0,
        ))
        result = export_html(session)
        # Unscored steps show "-" in heatmap
        assert "heatmap-cell" in result

    def test_collapsible_io_sections(self):
        session = _populated_session()
        result = export_html(session)
        assert "collapsible" in result
        assert "Input" in result
        assert "Output" in result

    def test_parent_child_shown_in_timeline(self):
        session = _populated_session()
        result = export_html(session)
        # web_search has parent s1, should show truncated parent id
        assert "Parent:" in result

    def test_mixed_error_and_scored_steps(self):
        session = TraceSession(trace_id="mixed")
        session.add_step(_step(
            span_id="m1", name="good_step", step_type="agent",
            step_index=0, latency_ms=100.0, score=5.0,
            score_explanation="Perfect.",
        ))
        session.add_step(_step(
            span_id="m2", name="bad_step", step_type="tool",
            step_index=1, latency_ms=200.0,
            error="Something broke",
        ))
        session.add_step(_step(
            span_id="m3", name="ok_step", step_type="agent",
            step_index=2, latency_ms=150.0, score=3.0,
            score_explanation="Acceptable.",
        ))
        result = export_html(session)
        assert "good_step" in result
        assert "bad_step" in result
        assert "ok_step" in result
        assert "Something broke" in result
        assert "Perfect." in result

    def test_score_bar_percentages(self):
        """Score bars should reflect the 1-5 scale mapped to 0-100%."""
        session = TraceSession(trace_id="bars")
        session.add_step(_step(
            span_id="b1", name="min_score", step_index=0,
            score=1.0, latency_ms=10.0,
        ))
        session.add_step(_step(
            span_id="b2", name="max_score", step_index=1,
            score=5.0, latency_ms=10.0,
        ))
        result = export_html(session)
        # score 1.0 -> 0%, score 5.0 -> 100%
        assert "width:0.0%" in result
        assert "width:100.0%" in result
