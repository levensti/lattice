from __future__ import annotations

import asyncio
import json
import logging
import re
from ..context import ActionRecord, JudgeConfig, TraceSession
from .prompt_builder import (
    JUDGE_SYSTEM_PROMPT,
    SessionPromptBuilder,
    ActionPromptBuilder,
    build_judge_prompt,
    build_session_judge_prompt,
)
from .providers import InferenceProvider

logger = logging.getLogger("lattice")


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract (score, explanation) from the judge LLM's response."""
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Attempt structured JSON parse
    try:
        data = json.loads(cleaned)
        return float(data["score"]), data.get("explanation", "")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.debug("JSON parse failed for judge response, trying regex fallbacks")

    # Fallback: regex for "score": N
    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if match:
        score = float(match.group(1))
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', text)
        explanation = exp_match.group(1) if exp_match else ""
        logger.debug("Parsed judge score via regex fallback: %.1f", score)
        return score, explanation

    # Fallback: N/M pattern (e.g. "3/5", "7/10")
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*\d+", text)
    if match:
        score = float(match.group(1))
        logger.debug("Parsed judge score via N/M fallback: %.1f", score)
        return score, text

    logger.warning("Could not parse judge response: %s", text[:200])
    return 0.0, f"Could not parse judge response: {text[:200]}"


def _build_prompt_for_action(
    action: ActionRecord,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> str:
    builder = action_prompt_builder or build_judge_prompt
    return builder(
        name=action.name,
        goal=action.goal,
        input_data=action.input_data,
        output_data=action.output_data,
    )


def _score_single_action(
    action: ActionRecord,
    provider: InferenceProvider,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    config = action.judge
    prov = config.provider if config else provider
    sys_prompt = config.system_prompt if config else system_prompt
    builder = (config.action_prompt_builder if config else None) or action_prompt_builder
    raw = prov.judge(sys_prompt, _build_prompt_for_action(action, builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f", action.name, action.score)


async def _async_score_single_action(
    action: ActionRecord,
    provider: InferenceProvider,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    config = action.judge
    prov = config.provider if config else provider
    sys_prompt = config.system_prompt if config else system_prompt
    builder = (config.action_prompt_builder if config else None) or action_prompt_builder
    raw = await prov.ajudge(sys_prompt, _build_prompt_for_action(action, builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f", action.name, action.score)


def score_trace(
    session: TraceSession,
    *,
    provider: InferenceProvider,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> list[ActionRecord]:
    """Score every action in the session synchronously. Updates actions in place.

    Args:
        provider: **Required.** :class:`InferenceProvider` used for actions
            that don't have a per-action ``JudgeConfig.provider`` set.
        system_prompt: Override the default system prompt (used as fallback
            for actions that don't set ``JudgeConfig.system_prompt``).
        action_prompt_builder: Callable that builds the user prompt for each
            action. Must accept keyword arguments ``name``,
            ``goal``, ``input_data``, ``output_data`` and return a string.
    """
    for a in session.actions:
        if a.error is None:
            _score_single_action(a, provider, system_prompt, action_prompt_builder)
    return session.actions


async def async_score_trace(
    session: TraceSession,
    *,
    provider: InferenceProvider,
    max_concurrency: int = 5,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> list[ActionRecord]:
    """Score every action concurrently. Updates actions in place.

    See :func:`score_trace` for how the judge LLM and prompts are configured.
    """
    sem = asyncio.Semaphore(max_concurrency)
    scorable = [a for a in session.actions if a.error is None]

    async def _bounded(action: ActionRecord) -> None:
        async with sem:
            await _async_score_single_action(action, provider, system_prompt, action_prompt_builder)

    await asyncio.gather(*[_bounded(a) for a in scorable])
    return session.actions


def score_session(
    session: TraceSession,
    *,
    provider: InferenceProvider,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    session_prompt_builder: SessionPromptBuilder | None = None,
) -> tuple[float, str]:
    """Judge the final output of the workflow against the session goal.

    Looks at the last action's output and evaluates it against ``session.goal``.
    Returns ``(score, explanation)`` and stores them on the session.

    Args:
        provider: **Required.** :class:`InferenceProvider` for the judge LLM.
        system_prompt: Override the system prompt sent to the judge LLM.
        session_prompt_builder: Callable that builds the user prompt.
            Must accept keyword arguments ``goal``, ``final_output``,
            ``workflow_name`` and return a string.

    Raises:
        ValueError: If the session has no goal or no actions.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.actions:
        raise ValueError("Session has no actions — nothing to judge.")

    last_action = session.actions[-1]

    builder = session_prompt_builder or build_session_judge_prompt
    prompt = builder(
        goal=session.goal,
        final_output=last_action.output_data,
        workflow_name=session.workflow_name,
        input_data=session.input_data or "",
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = provider.judge(system_prompt, prompt)
    score, explanation = _parse_judge_response(raw)
    session.session_score = score
    session.session_score_explanation = explanation
    logger.info("Session score: %.1f — %s", score, explanation)
    return score, explanation


class BackgroundScorer:
    """Score sessions off the critical path using a background worker.

    The critical path calls :meth:`submit` — a non-blocking
    ``queue.put_nowait()`` that returns immediately. A background asyncio
    worker drains the queue and scores each session as it arrives.

    At shutdown, call :meth:`drain` to wait for any in-flight scoring to
    finish before your process exits. That wait is not on the critical path.

    Intended to be used as a long-lived application-level singleton.
    Individual requests call :meth:`submit` and move on — no request
    ever waits for the judge.  At shutdown, call :meth:`cancel` to stop
    immediately (un-scored sessions stay in SQLite for offline scoring)::

        # ── app startup (once) ──────────────────────────────
        scorer = BackgroundScorer(
            provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
        )
        await scorer.start()

        # ── per-request hot path (many times) ───────────────
        async def handle_request(query):
            with trace_session(goal="...") as session:
                result = await run_agent(query)   # @action steps inside
            scorer.submit(session)                # non-blocking
            return result                         # returns immediately

        # ── app shutdown (immediate, no blocking) ───────────
        await scorer.cancel()

    Or as an async context manager scoped to the application lifetime::

        async with BackgroundScorer(
            provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
        ) as scorer:
            await serve_forever(scorer)   # submit() inside each request
        # cancel() called automatically on exit — does not block

    Args:
        provider: **Required.** :class:`InferenceProvider` instance for the
            judge LLM.
        max_concurrency: Maximum parallel judge calls per session.
        system_prompt: Override the judge system prompt.
        action_prompt_builder: Custom prompt builder for each action.
    """

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        max_concurrency: int = 5,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
        action_prompt_builder: ActionPromptBuilder | None = None,
        session_prompt_builder: SessionPromptBuilder | None = None,
        persist: bool = True,
    ) -> None:
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._system_prompt = system_prompt
        self._action_prompt_builder = action_prompt_builder
        self._session_prompt_builder = session_prompt_builder
        self._persist = persist
        self._queue: asyncio.Queue[TraceSession] | None = None
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background worker.

        Called automatically on first :meth:`submit` if not started
        explicitly. Safe to call multiple times.
        """
        if self._queue is not None:
            return
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())

    def submit(self, session: TraceSession) -> None:
        """Enqueue *session* for background scoring. Non-blocking.

        Lazily starts the background worker if :meth:`start` has not
        been called yet (requires a running event loop).
        """
        if self._queue is None:
            loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            self._worker_task = loop.create_task(self._worker())
        self._queue.put_nowait(session)

    async def drain(self) -> None:
        """Wait for all submitted sessions to finish scoring.

        Safe to call at shutdown — this is not on the critical path.
        """
        if self._queue is not None:
            await self._queue.join()

    async def cancel(self) -> None:
        """Stop the worker immediately, dropping any un-scored sessions.

        Use this at server shutdown when you don't want to block.
        Un-scored sessions are still in SQLite and can be scored later
        offline via :func:`score_trace`.
        """
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def close(self) -> None:
        """Drain pending sessions then shut down the background worker.

        If you don't want to wait, use :meth:`cancel` instead.
        """
        await self.drain()
        await self.cancel()

    async def _worker(self) -> None:
        while True:
            session = await self._queue.get()
            try:
                sem = asyncio.Semaphore(self._max_concurrency)
                scorable = [a for a in session.actions if a.error is None]

                async def _bounded(action: ActionRecord) -> None:
                    async with sem:
                        await _async_score_single_action(
                            action, self._provider, self._system_prompt, self._action_prompt_builder
                        )

                await asyncio.gather(*[_bounded(a) for a in scorable])
                logger.info(
                    "BackgroundScorer: finished scoring %d actions for session %s",
                    len(scorable), session.trace_id,
                )
                # Score the session end-to-end if it has a goal
                if session.goal and session.actions:
                    await async_score_session(
                        session,
                        provider=self._provider,
                        system_prompt=self._system_prompt,
                        session_prompt_builder=self._session_prompt_builder,
                    )
                if self._persist:
                    try:
                        from ..storage.store import save_session
                        save_session(session)
                    except Exception:
                        logger.warning(
                            "BackgroundScorer: failed to persist scored session %s",
                            session.trace_id, exc_info=True,
                        )
            except Exception:
                logger.exception(
                    "BackgroundScorer: error scoring session %s", session.trace_id
                )
            finally:
                self._queue.task_done()

    async def __aenter__(self) -> "BackgroundScorer":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.cancel()


async def async_score_session(
    session: TraceSession,
    *,
    provider: InferenceProvider,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    session_prompt_builder: SessionPromptBuilder | None = None,
) -> tuple[float, str]:
    """Async version of :func:`score_session`.

    See :func:`score_session` for parameter documentation.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.actions:
        raise ValueError("Session has no actions — nothing to judge.")

    last_action = session.actions[-1]

    builder = session_prompt_builder or build_session_judge_prompt
    prompt = builder(
        goal=session.goal,
        final_output=last_action.output_data,
        workflow_name=session.workflow_name,
        input_data=session.input_data or "",
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = await provider.ajudge(system_prompt, prompt)
    score, explanation = _parse_judge_response(raw)
    session.session_score = score
    session.session_score_explanation = explanation
    logger.info("Session score: %.1f — %s", score, explanation)
    return score, explanation
