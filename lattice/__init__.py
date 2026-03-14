"""lattice: Quality debugging framework for multi-agent systems."""

from .bottleneck import BottleneckResult, ImpactType, find_bottlenecks
from .context import (
    ActionRecord,
    GroupRecord,
    GroupType,
    LoopContext,
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
from .decorators import action, instrument, trace_action
from .judge.prompt_builder import (
    ActionPromptBuilder,
    JUDGE_SYSTEM_PROMPT,
    RATING_SCALE,
    RESPONSE_FORMAT,
    SessionPromptBuilder,
)
from .judge.providers import JudgeProvider
from .judge.scorer import async_score_session, async_score_trace, BackgroundScorer, score_session, score_trace
from .logging_utils import print_trace_summary
from .store import configure, traces

__all__ = [
    "action",
    "ActionPromptBuilder",
    "ActionRecord",
    "BottleneckResult",
    "GroupRecord",
    "GroupType",
    "ImpactType",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeProvider",
    "LoopContext",
    "RATING_SCALE",
    "RESPONSE_FORMAT",
    "SessionPromptBuilder",
    "TraceSession",
    "TransitionRecord",
    "async_score_session",
    "async_score_trace",
    "BackgroundScorer",
    "configure",
    "copy_trace_context",
    "find_bottlenecks",
    "get_current_session",
    "instrument",
    "print_trace_summary",
    "score_session",
    "score_trace",
    "trace_action",
    "trace_activation",
    "trace_iterations",
    "trace_loop",
    "trace_parallel",
    "trace_session",
    "trace_transition",
    "traces",
]
