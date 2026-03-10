"""lattice: Quality debugging framework for multi-agent systems."""

from .bottleneck import BottleneckResult, ImpactType, find_bottlenecks
from .config import LatticeConfig, configure, get_config
from .context import (
    GroupRecord,
    GroupType,
    LoopContext,
    StepRecord,
    TraceSession,
    TransitionRecord,
    copy_trace_context,
    get_current_session,
    trace_activation,
    trace_iterations,
    trace_loop,
    trace_parallel,
    trace_session,
    trace_transition,
)
from .decorators import instrument, step, trace_step
from .judge.providers import JudgeProvider
from .judge.scorer import async_score_session, async_score_trace, score_session, score_trace
from .logging_utils import print_trace_summary

__all__ = [
    "LatticeConfig",
    "BottleneckResult",
    "GroupRecord",
    "GroupType",
    "ImpactType",
    "JudgeProvider",
    "LoopContext",
    "StepRecord",
    "TraceSession",
    "TransitionRecord",
    "async_score_session",
    "async_score_trace",
    "configure",
    "copy_trace_context",
    "find_bottlenecks",
    "get_config",
    "get_current_session",
    "instrument",
    "print_trace_summary",
    "score_session",
    "score_trace",
    "step",
    "trace_activation",
    "trace_iterations",
    "trace_loop",
    "trace_parallel",
    "trace_session",
    "trace_step",
    "trace_transition",
]
