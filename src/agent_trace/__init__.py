"""agent-trace: Quality debugging framework for multi-agent systems."""

from .bottleneck import BottleneckResult, find_bottlenecks
from .config import AgentTraceConfig, configure, get_config
from .context import StepRecord, TraceSession, get_current_session, trace_session
from .decorators import step
from .judge.scorer import async_score_trace, score_trace
from .logging_utils import configure_logging, print_trace_summary

__all__ = [
    "AgentTraceConfig",
    "BottleneckResult",
    "StepRecord",
    "TraceSession",
    "async_score_trace",
    "configure",
    "configure_logging",
    "find_bottlenecks",
    "get_config",
    "get_current_session",
    "print_trace_summary",
    "score_trace",
    "step",
    "trace_session",
]
