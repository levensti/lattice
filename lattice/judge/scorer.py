from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from ..context import ActionRecord, TraceSession
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


def _score_single_action(
    action: ActionRecord,
    provider: JudgeProvider,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    raw = provider.judge(system_prompt, _build_prompt_for_action(action, action_prompt_builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f/5", action.name, action.score)


async def _async_score_single_action(
    action: ActionRecord,
    provider: JudgeProvider,
    system_prompt: str,
    action_prompt_builder: ActionPromptBuilder | None = None,
) -> None:
    logger.info("Scoring action: %s", action.name)
    raw = await provider.ajudge(system_prompt, _build_prompt_for_action(action, action_prompt_builder))
    action.score, action.score_explanation = _parse_judge_response(raw)
    logger.info("Scored action: %s → %.1f/5", action.name, action.score)


def _get_provider(
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> JudgeProvider:
    if provider is not None:
        return provider
    if model is not None:
        return resolve_provider(model, api_key)
    # Fall back to environment-based default
    env_key = os.environ.get("OPENAI_API_KEY")
    if not env_key:
        raise ValueError(
            "No judge provider configured. Pass provider=, model=, "
            "or set OPENAI_API_KEY."
        )
    return create_provider("openai", env_key, "gpt-4o", "https://api.openai.com/v1")


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

    The judge LLM can be specified in three ways (highest priority first):

    1. Pass an explicit *provider* instance.
    2. Pass *model* (and optionally *api_key*) — the provider is resolved
       automatically from the model name.
    3. Fall back to ``OPENAI_API_KEY`` environment variable with ``gpt-4o``.

    Prompt customization:

    - *system_prompt*: Override the system prompt sent to the judge LLM.
    - *action_prompt_builder*: Callable that builds the user prompt for each action.
      Must accept keyword arguments ``name``, ``description``, ``goal``,
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
