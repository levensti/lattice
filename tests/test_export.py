"""Tests for TraceSession export functionality (JSON and HTML)."""

import json
from agent_trace import TraceSession, StepRecord, export_json, export_html
from agent_trace.export import _session_to_dict, _score_color, _score_percent


def _step(
    span_id: str = "span1",
    name: str = "test_step",
    step_type: str = "agent",
    description: str = "A test step",
    criteria: str = "Works correctly",
    input_data: str = '{"key": "value"}',
    output_data: str = '{"result": "success"}',
    step_index: int = 0,
    latency_ms: float = 100.0,
    parent_span_id: str | None = None,
    score: float | None = None,
    score_explanation: str | None = None,
    error: str | None = None,
) -> StepRecord:
    """Helper to create a StepRecord."""
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


class TestExportJSON:
    """Test JSON export functionality."""

    def test_empty_session(self):
        """Test exporting an empty session."""
        session = TraceSession()
        result = export_json(session)
        data = json.loads(result)

        assert "trace_id" in data
        assert data["summary"]["total_steps"] == 0
        assert data["summary"]["scored_steps"] == 0
        assert data["steps"] == []
        assert data["bottlenecks"] == []

    def test_single_step_no_score(self):
        """Test exporting a single unscored step."""
        session = TraceSession()
        session.add_step(_step(name="search", step_type="tool"))
        result = export_json(session)
        data = json.loads(result)

        assert data["summary"]["total_steps"] == 1
        assert data["summary"]["scored_steps"] == 0
        assert len(data["steps"]) == 1
        assert data["steps"][0]["name"] == "search"

    def test_multiple_steps_with_scores(self):
        """Test exporting multiple steps with scores."""
        session = TraceSession()
        session.add_step(
            _step(
                span_id="s1",
                name="agent_step",
                step_type="agent",
                step_index=0,
                score=4.5,
                score_explanation="Good quality",
            )
        )
        session.add_step(
            _step(
                span_id="s2",
                name="tool_step",
                step_type="tool",
                step_index=1,
                parent_span_id="s1",
                score=3.8,
                score_explanation="Acceptable",
            )
        )
        result = export_json(session)
        data = json.loads(result)

        assert data["summary"]["total_steps"] == 2
        assert data["summary"]["scored_steps"] == 2
        assert data["summary"]["average_score"] == 4.15

    def test_step_with_error(self):
        """Test exporting a step with an error."""
        session = TraceSession()
        session.add_step(
            _step(
                name="failing_step",
                step_index=0,
                error="Connection timeout",
            )
        )
        result = export_json(session)
        data = json.loads(result)

        assert data["summary"]["error_count"] == 1
        assert data["steps"][0]["error"] == "Connection timeout"

    def test_total_latency_calculation(self):
        """Test that total latency is summed correctly."""
        session = TraceSession()
        session.add_step(_step(step_index=0, latency_ms=100.0))
        session.add_step(_step(step_index=1, latency_ms=250.5))
        session.add_step(_step(step_index=2, latency_ms=49.5))
        result = export_json(session)
        data = json.loads(result)

        assert data["summary"]["total_latency_ms"] == 400.0

    def test_bottleneck_fields_in_json(self):
        """Test that bottleneck analysis is included in JSON."""
        session = TraceSession()
        session.add_step(
            _step(
                name="slow_step",
                step_index=0,
                latency_ms=1000.0,
                score=1.5,
            )
        )
        session.add_step(
            _step(
                name="fast_step",
                step_index=1,
                latency_ms=10.0,
                score=4.8,
            )
        )
        result = export_json(session)
        data = json.loads(result)

        # The slow_step should be identified as a bottleneck
        assert len(data["bottlenecks"]) > 0
        bottleneck_names = [b["step_name"] for b in data["bottlenecks"]]
        assert "slow_step" in bottleneck_names

    def test_json_is_valid_and_pretty(self):
        """Test that export_json returns valid, prettified JSON."""
        session = TraceSession()
        session.add_step(_step(name="step1", step_index=0))
        result = export_json(session, indent=4)

        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, dict)

        # Should have newlines (prettified)
        assert "\n" in result


class TestExportHTML:
    """Test HTML export functionality."""

    def test_empty_session_html(self):
        """Test exporting an empty session to HTML."""
        session = TraceSession()
        html = export_html(session)

        assert "<!DOCTYPE html>" in html
        assert "<title>Trace Report" in html
        assert session.trace_id in html
        assert "Score Heatmap" in html
        assert "Step-by-Step Timeline" in html
        assert "Bottleneck Analysis" in html

    def test_html_is_self_contained(self):
        """Test that HTML is self-contained with inline CSS and JS."""
        session = TraceSession()
        session.add_step(_step(name="test"))
        html = export_html(session)

        # Should have inline CSS
        assert "<style>" in html
        assert "background:#0f1117" in html

        # Should have inline JS
        assert "<script>" in html
        assert "DOMContentLoaded" in html

        # Should NOT reference external resources
        assert "href=" not in html or ".css" not in html
        assert "src=" not in html or ".js" not in html

    def test_html_contains_summary(self):
        """Test that HTML includes the summary grid."""
        session = TraceSession()
        session.add_step(_step(step_index=0, score=4.0))
        session.add_step(_step(step_index=1, score=3.5, error=None))
        session.add_step(_step(step_index=2, error="Failed"))
        html = export_html(session)

        # Should contain summary metrics
        assert "Total Steps" in html or "3" in html
        # The summary should have avg score data (label or value)
        assert ("Average Score" in html or "Avg Score" in html or "3.75" in html)
        assert "Errors" in html or "1" in html

    def test_html_contains_heatmap(self):
        """Test that HTML includes the score heatmap."""
        session = TraceSession()
        session.add_step(_step(name="high_score", step_index=0, score=5.0))
        session.add_step(_step(name="low_score", step_index=1, score=1.0))
        html = export_html(session)

        assert "heatmap" in html
        assert "high_score" in html
        assert "low_score" in html

    def test_html_timeline_shows_steps(self):
        """Test that HTML timeline displays each step."""
        session = TraceSession()
        session.add_step(
            _step(
                name="researcher",
                step_type="agent",
                step_index=0,
                latency_ms=500.0,
                score=4.5,
                score_explanation="Found relevant papers",
            )
        )
        session.add_step(
            _step(
                name="web_search",
                step_type="tool",
                step_index=1,
                parent_span_id="s1",
                latency_ms=200.0,
                score=4.0,
            )
        )
        html = export_html(session)

        assert "Timeline" in html
        assert "researcher" in html
        assert "web_search" in html
        assert "agent" in html
        assert "tool" in html
        assert "4.5" in html
        assert "4.0" in html
        assert "Found relevant papers" in html

    def test_html_error_badge_shown(self):
        """Test that errors are highlighted in the HTML."""
        session = TraceSession()
        session.add_step(
            _step(
                name="failing_tool",
                step_index=0,
                error="Network error: timeout",
            )
        )
        html = export_html(session)

        assert "error" in html
        assert "Network error: timeout" in html or "timeout" in html

    def test_html_bottleneck_callouts(self):
        """Test that bottleneck steps are highlighted."""
        session = TraceSession()
        session.add_step(_step(name="fast", step_index=0, score=5.0, latency_ms=50.0))
        session.add_step(_step(name="slow", step_index=1, score=1.5, latency_ms=5000.0))
        html = export_html(session)

        assert "Bottleneck Analysis" in html
        assert "bottleneck" in html or "slow" in html

    def test_html_collapsible_io(self):
        """Test that input/output data is in a collapsible section."""
        session = TraceSession()
        session.add_step(
            _step(
                name="test",
                step_index=0,
                input_data='{"query": "what is AI?"}',
                output_data='{"answer": "AI is..."}',
            )
        )
        html = export_html(session)

        assert "Show Input / Output" in html
        assert "collapsible" in html
        assert "query" in html
        assert "answer" in html

    def test_html_escapes_special_chars(self):
        """Test that HTML properly escapes special characters."""
        session = TraceSession()
        session.add_step(
            _step(
                name="test<script>",
                step_index=0,
                description="Test & check",
                input_data='<img src=x>',
                error='Error: "quotes" & <tags>',
            )
        )
        html = export_html(session)

        # Should be HTML-escaped, not raw
        assert "<script>" not in html or "&lt;script&gt;" in html
        assert "<img" not in html or "&lt;img" in html


class TestScoreColorHelpers:
    """Test the colour helper functions."""

    def test_score_color_excellent(self):
        """Test colour for excellent scores."""
        assert _score_color(5.0) == "#3fb950"
        assert _score_color(4.6) == "#3fb950"

    def test_score_color_good(self):
        """Test colour for good scores."""
        assert _score_color(4.4) == "#58a6ff"
        assert _score_color(3.6) == "#58a6ff"

    def test_score_color_acceptable(self):
        """Test colour for acceptable scores."""
        assert _score_color(3.4) == "#d29922"
        assert _score_color(2.6) == "#d29922"

    def test_score_color_poor(self):
        """Test colour for poor scores."""
        assert _score_color(2.4) == "#f0883e"
        assert _score_color(1.6) == "#f0883e"

    def test_score_color_terrible(self):
        """Test colour for terrible scores."""
        assert _score_color(1.4) == "#f85149"
        assert _score_color(1.0) == "#f85149"

    def test_score_color_none(self):
        """Test colour for missing scores."""
        assert _score_color(None) == "#484f58"

    def test_score_percent_conversion(self):
        """Test score-to-percentage conversion."""
        assert _score_percent(None) == 0.0
        assert _score_percent(1.0) == 0.0
        assert _score_percent(3.0) == 50.0
        assert _score_percent(5.0) == 100.0


class TestTraceSessionMethods:
    """Test the new export methods on TraceSession itself."""

    def test_session_export_json_method(self):
        """Test calling export_json() as a session method."""
        session = TraceSession()
        session.add_step(_step(name="test", step_index=0, score=4.0))
        result = session.export_json()

        data = json.loads(result)
        assert data["summary"]["total_steps"] == 1
        assert data["summary"]["scored_steps"] == 1

    def test_session_export_html_method(self):
        """Test calling export_html() as a session method."""
        session = TraceSession()
        session.add_step(_step(name="test", step_index=0, score=4.0))
        html = session.export_html()

        assert "<!DOCTYPE html>" in html
        assert "test" in html

    def test_export_methods_produce_consistent_output(self):
        """Test that both export methods handle the same data."""
        session = TraceSession()
        session.add_step(
            _step(
                name="agent_call",
                step_type="agent",
                step_index=0,
                score=4.2,
                score_explanation="Excellent response",
                latency_ms=123.45,
            )
        )

        json_str = session.export_json()
        html_str = session.export_html()

        # Both should contain the same core data
        data = json.loads(json_str)
        assert data["summary"]["total_steps"] == 1
        assert "agent_call" in html_str
        assert "4.2" in html_str
        assert "Excellent response" in html_str


class TestSessionToDict:
    """Test the _session_to_dict helper."""

    def test_session_to_dict_structure(self):
        """Test the structure of the dict representation."""
        session = TraceSession()
        session.add_step(_step(name="test", step_index=0, score=3.5))

        data = _session_to_dict(session)

        assert "trace_id" in data
        assert "summary" in data
        assert "steps" in data
        assert "bottlenecks" in data

        assert data["trace_id"] == session.trace_id
        assert data["summary"]["total_steps"] == 1
        assert len(data["steps"]) == 1

    def test_session_to_dict_summary_fields(self):
        """Test that summary contains all expected fields."""
        session = TraceSession()
        session.add_step(_step(step_index=0, score=4.0, latency_ms=100.0))
        session.add_step(_step(step_index=1, score=3.0, latency_ms=50.0))

        data = _session_to_dict(session)
        summary = data["summary"]

        assert "total_steps" in summary
        assert "total_latency_ms" in summary
        assert "average_score" in summary
        assert "error_count" in summary
        assert "scored_steps" in summary

        assert summary["total_steps"] == 2
        assert summary["total_latency_ms"] == 150.0
        assert summary["average_score"] == 3.5
        assert summary["error_count"] == 0
        assert summary["scored_steps"] == 2

    def test_session_to_dict_includes_bottlenecks(self):
        """Test that bottleneck analysis is included."""
        session = TraceSession()
        session.add_step(_step(name="good", step_index=0, score=4.5))
        session.add_step(_step(name="bad", step_index=1, score=1.2))

        data = _session_to_dict(session)

        assert len(data["bottlenecks"]) > 0
        assert any(b["step_name"] == "bad" for b in data["bottlenecks"])
