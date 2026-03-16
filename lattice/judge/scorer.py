from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from ..context import ActionRecord, JudgeConfig, TraceSession
from .prompt_builder import (
    JUDGE_SYSTEM_PROMPT,
    SessionPromptBuilder,
    ActionPromptBuilder,
    build_judge_prompt,
    build_session_judge_prompt,
)
from .providers import JudgeProvider, create_provider, resolve_provider

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
        pass

    # Fallback: regex for "score": N
    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if match:
        score = float(match.group(1))
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', text)
        explanation = exp_match.group(1) if exp_match else text
        return score, explanation

    # Fallback: N/5 pattern
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", text)
    if match:
        return float(match.group(1)), text

    return 0.0, f"Could not parse judge response: {text[:200]}"


def _build_prompt_for_action(
    action: ActionRecord,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> str:
    builder = action_prompt_builder or build_judge_prompt
    return builder(
        name=action.name,
        description=action.description,
        goal=action.goal,
        input_data=action.input_data,
        output_data=action.output_data,
    )


def _provider_from_config(config: JudgeConfig, fallback: JudgeProvider | None) -> JudgeProvider:
    """Resolve a JudgeProvider from a JudgeConfig, falling back to *fallback*."""
    if config.provider is not None:
        return config.provider
    if config.model is not None:
        return resolve_provider(
            config.model,
            config.api_key,
            temperature=config.temperature if config.temperature is not None else 0.1,
            top_p=config.top_p,
        )
    if fallback is not None:
        return fallback
    raise ValueError(
        "JudgeConfig has no model or provider, and no global fallback provider is configured."
    )


def _score_single_action(
    action: ActionRecord,
    provider: JudgeProvider | None,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    config = action.judge
    prov = _provider_from_config(config, provider) if config else provider
    if prov is None:
        raise ValueError(
            f"No judge provider configured for action '{action.name}'. "
            "Pass provider=, model=, set OPENAI_API_KEY, or attach judge= to the action."
        )
    sys_prompt = config.system_prompt if config else system_prompt
    builder = (config.action_prompt_builder if config else None) or action_prompt_builder
    raw = prov.judge(sys_prompt, _build_prompt_for_action(action, builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f/5", action.name, action.score)


async def _async_score_single_action(
    action: ActionRecord,
    provider: JudgeProvider | None,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    config = action.judge
    prov = _provider_from_config(config, provider) if config else provider
    if prov is None:
        raise ValueError(
            f"No judge provider configured for action '{action.name}'. "
            "Pass provider=, model=, set OPENAI_API_KEY, or attach judge= to the action."
        )
    sys_prompt = config.system_prompt if config else system_prompt
    builder = (config.action_prompt_builder if config else None) or action_prompt_builder
    raw = await prov.ajudge(sys_prompt, _build_prompt_for_action(action, builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f/5", action.name, action.score)


def _get_provider(
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> JudgeProvider | None:
    """Resolve the global fallback provider, or return None if unconfigured.

    Returns None when no global provider is set — this is fine as long as
    every action has per-action ``judges`` configured.
    """
    if provider is not None:
        return provider
    if model is not None:
        return resolve_provider(model, api_key)
    # Fall back to environment-based default
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return create_provider("openai", env_key, "gpt-4o", "https://api.openai.com/v1")
    return None


def score_trace(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> list[ActionRecord]:
    """Score every action in the session synchronously. Updates actions in place.

    The global judge LLM is resolved in order of priority:

    1. Pass an explicit *provider* instance.
    2. Pass *model* (and optionally *api_key*) — the provider is resolved
       automatically from the model name.
    3. Fall back to ``OPENAI_API_KEY`` environment variable with ``gpt-4o``.

    Actions decorated with ``judge=JudgeConfig(...)`` use that config's
    ``system_prompt`` and provider instead of the global ones.

    The global provider is optional when every action has a per-action judge.

    Prompt customization:

    - *system_prompt*: Override the default system prompt (used as fallback
      for actions that don't set ``JudgeConfig.system_prompt``).
    - *action_prompt_builder*: Callable that builds the user prompt for each
      action. Must accept keyword arguments ``name``, ``description``, ``goal``,
      ``input_data``, ``output_data`` and return a string.
    """
    prov = _get_provider(provider, model, api_key)
    for a in session.actions:
        if a.error is None:
            _score_single_action(a, prov, system_prompt, action_prompt_builder)
    return session.actions


async def async_score_trace(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_concurrency: int = 5,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> list[ActionRecord]:
    """Score every action concurrently. Updates actions in place.

    See :func:`score_trace` for how the judge LLM and prompts are configured.
    """
    prov = _get_provider(provider, model, api_key)
    sem = asyncio.Semaphore(max_concurrency)
    scorable = [a for a in session.actions if a.error is None]

    async def _bounded(action: ActionRecord) -> None:
        async with sem:
            await _async_score_single_action(action, prov, system_prompt, action_prompt_builder)

    await asyncio.gather(*[_bounded(a) for a in scorable])
    return session.actions


def score_session(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    session_prompt_builder: SessionPromptBuilder | None = None,
) -> tuple[float, str]:
    """Judge the final output of the workflow against the session goal.

    Looks at the last action's output and evaluates it against ``session.goal``.
    Returns ``(score, explanation)`` and stores them on the session.

    See :func:`score_trace` for how the judge LLM is resolved.

    Prompt customization:

    - *system_prompt*: Override the system prompt sent to the judge LLM.
    - *session_prompt_builder*: Callable that builds the user prompt.
      Must accept keyword arguments ``goal``, ``final_output``,
      ``workflow_name`` and return a string.

    Raises:
        ValueError: If the session has no goal or no actions.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.actions:
        raise ValueError("Session has no actions — nothing to judge.")

    prov = _get_provider(provider, model, api_key)
    last_action = session.actions[-1]

    builder = session_prompt_builder or build_session_judge_prompt
    prompt = builder(
        goal=session.goal,
        final_output=last_action.output_data,
        workflow_name=session.workflow_name,
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = prov.judge(system_prompt, prompt)
    score, explanation = _parse_judge_response(raw)
    session.session_score = score
    session.session_score_explanation = explanation
    logger.info("Session score: %.1f/5 — %s", score, explanation)
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
        scorer = BackgroundScorer(model="gpt-4o")
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

        async with BackgroundScorer(model="gpt-4o") as scorer:
            await serve_forever(scorer)   # submit() inside each request
        # cancel() called automatically on exit — does not block

    Args:
        provider: Explicit :class:`JudgeProvider` instance.
        model: Model name — provider is resolved automatically.
        api_key: API key for the judge provider.
        max_concurrency: Maximum parallel judge calls per session.
        system_prompt: Override the judge system prompt.
        action_prompt_builder: Custom prompt builder for each action.
    """

    def __init__(
        self,
        *,
        provider: JudgeProvider | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_concurrency: int = 5,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
        action_prompt_builder: ActionPromptBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._max_concurrency = max_concurrency
        self._system_prompt = system_prompt
        self._action_prompt_builder = action_prompt_builder
        self._queue: asyncio.Queue[TraceSession] | None = None
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background worker. Must be called before :meth:`submit`."""
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())

    def submit(self, session: TraceSession) -> None:
        """Enqueue *session* for background scoring. Non-blocking."""
        if self._queue is None:
            raise RuntimeError(
                "BackgroundScorer not started — call `await scorer.start()` first."
            )
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
        prov = _get_provider(self._provider, self._model, self._api_key)
        while True:
            session = await self._queue.get()
            try:
                sem = asyncio.Semaphore(self._max_concurrency)
                scorable = [a for a in session.actions if a.error is None]

                async def _bounded(action: ActionRecord) -> None:
                    async with sem:
                        await _async_score_single_action(
                            action, prov, self._system_prompt, self._action_prompt_builder
                        )

                await asyncio.gather(*[_bounded(a) for a in scorable])
                logger.info(
                    "BackgroundScorer: finished scoring session %s (%d actions)",
                    session.trace_id, len(scorable),
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
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
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

    prov = _get_provider(provider, model, api_key)
    last_action = session.actions[-1]

    builder = session_prompt_builder or build_session_judge_prompt
    prompt = builder(
        goal=session.goal,
        final_output=last_action.output_data,
        workflow_name=session.workflow_name,
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = await prov.ajudge(system_prompt, prompt)
    score, explanation = _parse_judge_response(raw)
    session.session_score = score
    session.session_score_explanation = explanation
    logger.info("Session score: %.1f/5 — %s", score, explanation)
    return score, explanation
