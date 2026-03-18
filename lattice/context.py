from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .judge.prompt_builder import ActionPromptBuilder
    from .judge.providers import InferenceProvider

GroupType = Literal["loop", "parallel"]

logger = logging.getLogger("lattice")


@dataclass
class JudgeResult:
    """Score produced by a judge for a single action.

    One judge, one prompt, one score. Access via ``action.judge_result``.
    """

    score: float
    explanation: str
    name: str  # descriptive name for this evaluation
    model: str  # model that produced this score


class JudgeConfig:
    """Configuration for the judge that evaluates an action.

    Attach to ``@action(judge=JudgeConfig(...))`` to configure how the action
    is evaluated. The ``system_prompt`` is the rubric — define scoring
    criteria, anchors, response format, and any reference material there.

    Inference settings (model, API key, temperature, etc.) live on the
    :class:`~lattice.judge.providers.InferenceProvider` instance passed
    via ``provider``. ``JudgeConfig`` only controls *evaluation semantics*
    — the rubric and prompt format.

    Examples::

        from lattice.judge.providers import OpenAIProvider, AnthropicProvider

        @action(
            goal="Summarise the paper into 3 bullet points",
            judge=JudgeConfig(
                system_prompt=\"\"\"You evaluate research summaries.

Score 1: Wrong number of bullets or major factual errors.
Score 2: 3 bullets but one is factually wrong.
Score 3: 3 accurate bullets, one vague or incomplete.
Score 4: 3 accurate bullets, minor wording issues.
Score 5: 3 tight, distinct, fully accurate bullets.

Respond with JSON only: {"reasoning": "...", "score": <1-5>, "explanation": "..."}\"\"\",
                provider=AnthropicProvider("claude-opus-4-6"),
            ),
        )
        def summarise(paper): ...

        # Use any OpenAI-compatible endpoint (Fireworks, Sail, etc.)
        @action(
            goal="...",
            judge=JudgeConfig(
                system_prompt="...",
                provider=OpenAIProvider(
                    "accounts/fireworks/my-model",
                    api_base="https://api.fireworks.ai/inference/v1",
                ),
            ),
        )
        def step(): ...

    Args:
        system_prompt: **Required.** The judge's rubric and instructions.
            Define scoring criteria, per-score anchors, response format, and
            any reference material here. This is the system turn sent to the
            judge LLM on every call.
        provider: **Required.** :class:`~lattice.judge.providers.InferenceProvider`
            instance that handles the actual LLM call.
        action_prompt_builder: Custom callable that fully replaces the default
            user prompt. Must accept ``name``, ``goal``,
            ``input_data``, ``output_data`` as keyword arguments and return a
            string.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        provider: InferenceProvider,
        action_prompt_builder: ActionPromptBuilder | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.provider = provider
        self.action_prompt_builder = action_prompt_builder

    def __repr__(self) -> str:
        parts = ["system_prompt=..."]
        if self.provider is not None:
            parts.append(f"provider={self.provider!r}")
        return f"JudgeConfig({', '.join(parts)})"


@dataclass
class ActionRecord:
    """A single traced action within a session."""

    span_id: str
    name: str
    goal: str
    input_data: str
    output_data: str
    action_index: int
    latency_ms: float
    parent_span_id: str | None = None
    score: float | None = None
    score_explanation: str | None = None
    error: str | None = None
    group_id: str | None = None
    iteration: int | None = None
    activation_reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    # Per-action judge config — live object, not persisted (stripped in to_dict)
    judge: JudgeConfig | None = field(default=None, compare=False)
    # Serializable summary of judge config — persisted
    judge_config: dict[str, Any] | None = field(default=None)
    # Judge result — persisted
    judge_result: JudgeResult | None = field(default=None)


@dataclass
class GroupRecord:
    """Metadata about a structural group of actions (loop or parallel)."""

    group_id: str
    group_type: GroupType
    name: str
    parent_span_id: str | None = None


@dataclass
class TransitionRecord:
    """A recorded routing decision between actions."""

    from_span_id: str | None
    to_name: str
    reason: str
    auto: bool = False


def _judge_config_summary(judge: JudgeConfig) -> dict[str, Any]:
    """Extract serializable metadata from a live JudgeConfig."""
    provider = judge.provider
    summary: dict[str, Any] = {
        "system_prompt": judge.system_prompt,
        "provider": type(provider).__name__,
        "model": getattr(provider, "model", None),
        "has_custom_prompt_builder": judge.action_prompt_builder is not None,
    }
    # Capture inference settings from the provider if available
    for attr in ("temperature", "top_p", "timeout", "api_base", "api_type"):
        val = getattr(provider, attr, None)
        if val is not None:
            # api_type is an enum — persist its value
            summary[attr] = val.value if hasattr(val, "value") else val
    return summary


def _format_action_line(action: ActionRecord) -> str:
    status = "ERROR" if action.error else "OK"
    score_str = f"{action.score:.1f}" if action.score is not None else "-"
    parts = [f"{action.name}  {status}  {action.latency_ms:.0f}ms  score={score_str}"]
    if action.iteration is not None:
        parts.append(f"iter={action.iteration}")
    if action.activation_reason:
        parts.append(f"activated: {action.activation_reason}")
    return "  ".join(parts)


def _format_tree(
    node: ActionRecord,
    children: dict[str | None, list["ActionRecord"]],
    lines: list[str],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    lines.append(f"{prefix}{connector}{_format_action_line(node)}")
    detail_prefix = prefix + ("    " if is_last else "\u2502   ")
    if node.error:
        lines.append(f"{detail_prefix}error: {node.error}")
    if node.score_explanation:
        lines.append(f"{detail_prefix}reason: {node.score_explanation}")
    child_nodes = children.get(node.span_id, [])
    for i, child in enumerate(child_nodes):
        _format_tree(
            child, children, lines,
            prefix=prefix + ("    " if is_last else "\u2502   "),
            is_last=(i == len(child_nodes) - 1),
        )


@dataclass
class TraceSession:
    """Groups multiple traced actions into a single debugging session."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workflow_name: str = ""
    goal: str = ""
    input_data: str | None = field(default=None)
    actions: list[ActionRecord] = field(default_factory=list)
    groups: list[GroupRecord] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    session_score: float | None = field(default=None)
    session_score_explanation: str | None = field(default=None)
    created_at: str | None = field(default=None)
    _action_counter: int = field(default=0, repr=False)

    def next_index(self) -> int:
        idx = self._action_counter
        self._action_counter += 1
        return idx

    def add_action(self, action: ActionRecord) -> None:
        # Auto-capture the first root action's input as the workflow input
        if self.input_data is None and action.parent_span_id is None and action.input_data:
            self.input_data = action.input_data
        self.actions.append(action)

    def add_group(self, group: GroupRecord) -> None:
        self.groups.append(group)

    def add_transition(self, transition: TransitionRecord) -> None:
        self.transitions.append(transition)

    @property
    def bottlenecks(self):
        """Lazily compute and return bottleneck analysis results."""
        from .bottleneck import find_bottlenecks
        return find_bottlenecks(self)

    def __str__(self) -> str:
        """Human-readable summary of the trace session."""
        if not self.actions:
            return "(no actions recorded)"

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 60)
        title = self.workflow_name or "Trace Summary"
        lines.append(f"  {title}  (trace_id={self.trace_id})")
        if self.goal:
            lines.append(f"  Goal: {self.goal}")
        lines.append("=" * 60)

        # Build parent-child tree
        children_map: dict[str | None, list[ActionRecord]] = {}
        for a in self.actions:
            children_map.setdefault(a.parent_span_id, []).append(a)
        roots = children_map.get(None, [])

        for i, root in enumerate(roots):
            _format_tree(root, children_map, lines, prefix="  ", is_last=(i == len(roots) - 1))

        total_ms = sum(a.latency_ms for a in self.actions)
        lines.append("-" * 60)
        lines.append(f"  Total actions: {len(self.actions)}  |  Total time: {total_ms:.0f}ms")

        if self.groups:
            lines.append("  Groups:")
            for g in self.groups:
                symbol = "\u27f3" if g.group_type == "loop" else "\u2225"
                group_steps = [a for a in self.actions if a.group_id == g.group_id]
                action_count = len(group_steps)
                if g.group_type == "loop":
                    iters = {a.iteration for a in group_steps if a.iteration is not None}
                    iter_info = f", {len(iters)} iterations" if iters else ""
                    lines.append(f"    {symbol} {g.name} ({action_count} steps{iter_info})")
                else:
                    lines.append(f"    {symbol} {g.name} ({action_count} branches)")

        if self.transitions:
            merged: dict[tuple[str | None, str], TransitionRecord] = {}
            for t in self.transitions:
                key = (t.from_span_id, t.to_name)
                existing = merged.get(key)
                if existing is None or (existing.auto and not t.auto):
                    merged[key] = t
            display = [t for t in merged.values() if not t.auto or t.reason]
            if display:
                lines.append("  Transitions:")
                for t in display:
                    from_name = "?"
                    if t.from_span_id:
                        from_step = next(
                            (a for a in self.actions if a.span_id == t.from_span_id), None
                        )
                        from_name = from_step.name if from_step else t.from_span_id
                    reason = f': "{t.reason}"' if t.reason else ""
                    lines.append(f"    {from_name} \u2192 {t.to_name}{reason}")

        if self.session_score is not None:
            lines.append(f"  Session score: {self.session_score:.1f}")
            if self.session_score_explanation:
                lines.append(f"  Verdict: {self.session_score_explanation}")
        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the session to a plain dict (suitable for JSON)."""
        d = asdict(self)
        d.pop("_action_counter", None)
        for action, action_dict in zip(self.actions, d.get("actions", [])):
            # Snapshot judge config summary into the dict copy (not the
            # original ActionRecord) so serialization has no side effects.
            if action.judge is not None and action_dict.get("judge_config") is None:
                action_dict["judge_config"] = _judge_config_summary(action.judge)
            # judge contains non-serializable objects (callables, provider
            # instances); strip it — judge_config is persisted instead.
            action_dict.pop("judge", None)
        return d


_current_session: ContextVar[TraceSession | None] = ContextVar(
    "lattice_session", default=None
)
_current_span_id: ContextVar[str | None] = ContextVar(
    "lattice_span_id", default=None
)
_current_group_id: ContextVar[str | None] = ContextVar(
    "lattice_group_id", default=None
)
_current_iteration: ContextVar[int | None] = ContextVar(
    "lattice_iteration", default=None
)
_current_activation_reason: ContextVar[str | None] = ContextVar(
    "lattice_activation_reason", default=None
)


class LoopContext:
    """Handle returned by :func:`trace_loop` for managing iterations."""

    def __init__(self, name: str, group_id: str):
        self.name = name
        self.group_id = group_id
        self._iteration_count = 0

    @contextmanager
    def iteration(self):
        """Mark the boundary of one loop iteration.

        All ``@action`` calls inside this context are tagged with the
        current iteration number.
        """
        iter_num = self._iteration_count
        self._iteration_count += 1
        token = _current_iteration.set(iter_num)
        try:
            yield iter_num
        finally:
            _current_iteration.reset(token)

    @property
    def iteration_count(self) -> int:
        return self._iteration_count


class trace_iterations:
    """Wrap an iterable with loop tracing, avoiding nested context managers.

    Equivalent to ``trace_loop`` + ``loop.iteration()`` in a single
    ``for`` statement::

        for i in trace_iterations("react", range(5)):
            thought = think(state)
            action = act(thought)
            state = observe(action)

    Access :attr:`iteration_count` after the loop finishes to see how
    many iterations actually ran (useful with early ``break``).
    """

    def __init__(self, name: str, iterable):
        self.name = name
        self._iterable = iterable
        self._iteration_count = 0
        self._group_id: str | None = None

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def __iter__(self):
        session = _current_session.get()
        group_id = uuid.uuid4().hex
        self._group_id = group_id
        parent_span = _current_span_id.get()

        if session:
            session.add_group(GroupRecord(
                group_id=group_id,
                group_type="loop",
                name=self.name,
                parent_span_id=parent_span,
            ))

        group_token = _current_group_id.set(group_id)
        try:
            for item in self._iterable:
                iter_token = _current_iteration.set(self._iteration_count)
                self._iteration_count += 1
                try:
                    yield item
                finally:
                    _current_iteration.reset(iter_token)
        finally:
            _current_group_id.reset(group_token)


@contextmanager
def trace_loop(name: str):
    """Group actions into a named loop with numbered iterations.

    Usage::

        with trace_loop("react") as loop:
            while not done:
                with loop.iteration():
                    thought = think(state)
                    action = act(thought)
                    state = observe(action)
    """
    session = _current_session.get()
    group_id = uuid.uuid4().hex
    parent_span = _current_span_id.get()

    if session:
        session.add_group(GroupRecord(
            group_id=group_id,
            group_type="loop",
            name=name,
            parent_span_id=parent_span,
        ))

    token = _current_group_id.set(group_id)
    loop_ctx = LoopContext(name, group_id)
    try:
        yield loop_ctx
    finally:
        _current_group_id.reset(token)


@contextmanager
def trace_parallel(name: str):
    """Mark actions as running concurrently within this context.

    Works with ``asyncio.gather`` (context is automatically copied to tasks)
    and with threads when combined with :func:`copy_trace_context`.

    Usage::

        with trace_parallel("search_fanout"):
            results = await asyncio.gather(
                search_web(q), search_db(q), search_cache(q)
            )
    """
    session = _current_session.get()
    group_id = uuid.uuid4().hex
    parent_span = _current_span_id.get()

    if session:
        session.add_group(GroupRecord(
            group_id=group_id,
            group_type="parallel",
            name=name,
            parent_span_id=parent_span,
        ))

    token = _current_group_id.set(group_id)
    try:
        yield group_id
    finally:
        _current_group_id.reset(token)


def trace_transition(*, to: str, reason: str = "") -> None:
    """Record a routing decision from the current action.

    Call inside a ``@action``-decorated function to capture why control
    is being transferred to a particular next action.

    Usage::

        @action(name="router", ...)
        def router(state):
            if state.needs_retry:
                trace_transition(to="retry", reason="validation failed")
                return retry(state)
            trace_transition(to="finalize", reason="checks passed")
            return finalize(state)
    """
    session = _current_session.get()
    if session:
        from_span = _current_span_id.get()
        session.add_transition(TransitionRecord(
            from_span_id=from_span,
            to_name=to,
            reason=reason,
        ))


@contextmanager
def trace_activation(*, reason: str):
    """Provide an activation reason for actions without a parent caller.

    Used in blackboard / event-driven architectures where agents
    self-activate based on shared state changes.

    Usage::

        with trace_activation(reason="knowledge_base updated by agent_a"):
            agent_b.process(blackboard)
    """
    token = _current_activation_reason.set(reason)
    try:
        yield
    finally:
        _current_activation_reason.reset(token)


@contextmanager
def trace_session(
    trace_id: str | None = None,
    *,
    workflow_name: str = "",
    goal: str,
    persist: bool = True,
    judge: "InferenceProvider | None" = None,
    judge_system_prompt: str | None = None,
):
    """Context manager that groups ``@action``-decorated calls into a single trace.

    Scoring semantics:

    - ``JudgeConfig`` on individual actions controls **per-action** judging.
      Those configs carry their own provider + rubric and are evaluated at
      session exit.
    - The ``judge=`` argument on :func:`trace_session` controls **session-level**
      judging only — i.e. “given the inputs, outputs, and trajectory, how
      good was this overall workflow?”  It is never reused for per-action
      judging.

    Args:
        trace_id: Custom trace ID (auto-generated if omitted).
        workflow_name: Human-readable name for this workflow.
        goal: **Required.** The desired outcome of the workflow — used
            for session-level scoring via :func:`score_session`.
        persist: If ``True`` (default), automatically save the completed
            session to the local SQLite store when the context exits.
        judge: Optional :class:`~lattice.judge.providers.InferenceProvider`.
            When set, the final output of the workflow is judged against
            ``goal`` on exit, producing a session score. It is **not**
            used for per-action scoring.
        judge_system_prompt: Override the default judge system prompt.
            Only used when *judge* is set.
    """
    session = TraceSession(
        trace_id=trace_id or uuid.uuid4().hex,
        workflow_name=workflow_name,
        goal=goal,
    )
    logger.info("Trace session started: %s (trace_id=%s)", workflow_name or "(unnamed)", session.trace_id)
    session_token = _current_session.set(session)
    span_token = _current_span_id.set(None)
    try:
        yield session
    finally:
        _current_session.reset(session_token)
        _current_span_id.reset(span_token)

        # First, score any actions that have their own JudgeConfig.  Each
        # action carries its own provider + rubric; we never reuse the
        # session-level judge for per-action evaluation.
        if session.actions:
            try:
                from .judge.scorer import _score_single_action

                for a in session.actions:
                    if a.error is None and a.judge is not None:
                        _score_single_action(a, a.judge.provider, a.judge.system_prompt)
            except Exception:
                logger.warning(
                    "Per-action scoring failed for trace %s",
                    session.trace_id,
                    exc_info=True,
                )

        # Then, optionally score the overall trajectory against the session goal.
        if judge is not None and session.goal and session.actions:
            try:
                from .judge.scorer import score_session as _score_session
                from .judge.prompt_builder import JUDGE_SYSTEM_PROMPT

                sys_prompt = judge_system_prompt or JUDGE_SYSTEM_PROMPT
                _score_session(session, provider=judge, system_prompt=sys_prompt)
            except Exception:
                logger.warning(
                    "Session-level scoring failed for trace %s",
                    session.trace_id,
                    exc_info=True,
                )

        if persist:
            try:
                from .storage.store import save_session
                save_session(session)
            except Exception:
                logger.warning(
                    "Failed to persist trace %s to local store",
                    session.trace_id,
                    exc_info=True,
                )

        logger.info(
            "Trace session ended (trace_id=%s, steps=%d)",
            session.trace_id,
            len(session.actions),
        )


def judged_session(
    *,
    workflow_name: str = "",
    goal: str,
    judge: "InferenceProvider",
    persist: bool = True,
    trace_id: str | None = None,
    judge_system_prompt: str | None = None,
):
    """Stricter variant of :func:`trace_session` that requires a judge.

    Applications that always want session-level judging should use this helper
    instead of :func:`trace_session`.
    """
    if judge is None:
        raise TypeError("judged_session requires judge=InferenceProvider(...), got None")
    if not goal:
        raise TypeError("judged_session requires goal='...'.")
    return trace_session(
        trace_id=trace_id,
        workflow_name=workflow_name,
        goal=goal,
        persist=persist,
        judge=judge,
        judge_system_prompt=judge_system_prompt,
    )


def get_current_session() -> TraceSession | None:
    return _current_session.get()


def copy_trace_context() -> contextvars.Context:
    """Copy the current trace context for use in a new thread.

    ``ContextVar`` values are not automatically propagated to threads.
    Use this to capture the context and run traced actions within it::

        from concurrent.futures import ThreadPoolExecutor

        ctx = copy_trace_context()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(ctx.run, my_step, arg1, arg2)
    """
    return contextvars.copy_context()
