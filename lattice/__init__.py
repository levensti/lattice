"""lattice: Quality debugging framework for multi-agent systems."""

from .bottleneck import BottleneckResult, ImpactType, find_bottlenecks
from .context import (
    ActionRecord,
    GroupRecord,
    GroupType,
    JudgeConfig,
    JudgeResult,
    LoopContext,
    TraceSession,
    TransitionRecord,
    copy_trace_context,
    get_current_session,
    judged_session,
    trace_activation,
    trace_iterations,
    trace_loop,
    trace_parallel,
    trace_session,
    trace_transition,
)
from .decorators import action, instrument, trace_action
from .logging_utils import print_trace_summary
from .storage import SQLiteStore, Store
from .storage.store import configure, traces

_HAS_JUDGE_DEPS = True
try:
    from .judge.prompt_builder import (
        ActionPromptBuilder,
        JUDGE_SYSTEM_PROMPT,
        SessionPromptBuilder,
    )
    from .judge.providers import (
        AnthropicProvider,
        ApiType,
        InferenceProvider,
        OpenAIProvider,
        OpenRouterProvider,
    )
    from .judge.scorer import (
        async_score_session,
        async_score_trace,
        BackgroundScorer,
        score_session,
        score_trace,
    )
except ImportError:  # optional dependency (e.g., httpx)
    _HAS_JUDGE_DEPS = False

__all__ = [
    "action",
    "ActionRecord",
    "BottleneckResult",
    "GroupRecord",
    "GroupType",
    "ImpactType",
    "JudgeConfig",
    "JudgeResult",
    "LoopContext",
    "TraceSession",
    "TransitionRecord",
    "configure",
    "copy_trace_context",
    "find_bottlenecks",
    "get_current_session",
    "instrument",
    "judged_session",
    "SQLiteStore",
    "Store",
    "trace_action",
    "trace_activation",
    "trace_iterations",
    "trace_loop",
    "trace_parallel",
    "trace_session",
    "trace_transition",
    "traces",
]

if _HAS_JUDGE_DEPS:
    __all__.extend([
        "ActionPromptBuilder",
        "JUDGE_SYSTEM_PROMPT",
        "SessionPromptBuilder",
        "AnthropicProvider",
        "ApiType",
        "InferenceProvider",
        "OpenAIProvider",
        "OpenRouterProvider",
        "async_score_session",
        "async_score_trace",
        "BackgroundScorer",
        "score_session",
        "score_trace",
    ])
