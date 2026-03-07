"""agent-trace: Quality debugging framework for multi-agent systems."""

from .bottleneck import BottleneckResult, find_bottlenecks
from .config import AgentTraceConfig, configure, get_config
from .context import StepRecord, TraceSession, get_current_session, trace_session
from .decorators import step, trace_agent, trace_tool
from .judge.scorer import async_score_trace, score_trace

__all__ = [
    "AgentTraceConfig",
    "BottleneckResult",
    "StepRecord",
    "TraceSession",
    "async_score_trace",
    "configure",
    "find_bottlenecks",
    "get_config",
    "get_current_session",
    "score_trace",
    "step",
    "trace_agent",
    "trace_session",
    "trace_tool",
]
