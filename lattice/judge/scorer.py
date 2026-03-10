from __future__ import annotations

import asyncio
import json
import logging
import re
from ..config import get_config
from ..context import StepRecord, TraceSession
from .prompt_builder import JUDGE_SYSTEM_PROMPT, build_judge_prompt, build_session_judge_prompt
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


def _build_prompt_for_step(step: StepRecord) -> str:
    return build_judge_prompt(
        name=step.name,
        description=step.description,
        goal=step.goal,
        input_data=step.input_data,
        output_data=step.output_data,
    )


def _score_single_step(step: StepRecord, provider: JudgeProvider) -> None:
    logger.info("Scoring step: %s", step.name)
    raw = provider.judge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)
    logger.info("Scored step: %s → %.1f/5", step.name, step.score)


async def _async_score_single_step(step: StepRecord, provider: JudgeProvider) -> None:
    logger.info("Scoring step: %s", step.name)
    raw = await provider.ajudge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)
    logger.info("Scored step: %s → %.1f/5", step.name, step.score)


def _get_provider(
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> JudgeProvider:
    if provider is not None:
        return provider
    if model is not None:
        return resolve_provider(model, api_key)
    cfg = get_config()
    if not cfg.judge_api_key:
        raise ValueError(
            "No judge API key configured. Pass model= to the scoring "
            "function, call configure(), or set OPENAI_API_KEY."
        )
    return create_provider(
        cfg.judge_provider, cfg.judge_api_key, cfg.judge_model, cfg.judge_api_base,
        timeout=cfg.judge_timeout,
    )


def score_trace(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> list[StepRecord]:
    """Score every step in the session synchronously. Updates steps in place.

    The judge LLM can be specified in three ways (highest priority first):

    1. Pass an explicit *provider* instance.
    2. Pass *model* (and optionally *api_key*) — the provider is resolved
       automatically from the model name.
    3. Fall back to the global config set via :func:`configure`.
    """
    prov = _get_provider(provider, model, api_key)
    for step in session.steps:
        if step.error is None:
            _score_single_step(step, prov)
    return session.steps


async def async_score_trace(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_concurrency: int = 5,
) -> list[StepRecord]:
    """Score every step concurrently. Updates steps in place.

    See :func:`score_trace` for how the judge LLM is resolved.
    """
    prov = _get_provider(provider, model, api_key)
    sem = asyncio.Semaphore(max_concurrency)
    scorable = [s for s in session.steps if s.error is None]

    async def _bounded(step: StepRecord) -> None:
        async with sem:
            await _async_score_single_step(step, prov)

    await asyncio.gather(*[_bounded(s) for s in scorable])
    return session.steps


def score_session(
    session: TraceSession,
    *,
    provider: JudgeProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[float, str]:
    """Judge the final output of the workflow against the session goal.

    Looks at the last step's output and evaluates it against ``session.goal``.
    Returns ``(score, explanation)`` and stores them on the session.

    See :func:`score_trace` for how the judge LLM is resolved.

    Raises:
        ValueError: If the session has no goal or no steps.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.steps:
        raise ValueError("Session has no steps — nothing to judge.")

    prov = _get_provider(provider, model, api_key)
    last_step = session.steps[-1]
    prompt = build_session_judge_prompt(
        goal=session.goal,
        final_output=last_step.output_data,
        workflow_name=session.workflow_name,
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = prov.judge(JUDGE_SYSTEM_PROMPT, prompt)
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
) -> tuple[float, str]:
    """Async version of :func:`score_session`.

    See :func:`score_trace` for how the judge LLM is resolved.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.steps:
        raise ValueError("Session has no steps — nothing to judge.")

    prov = _get_provider(provider, model, api_key)
    last_step = session.steps[-1]
    prompt = build_session_judge_prompt(
        goal=session.goal,
        final_output=last_step.output_data,
        workflow_name=session.workflow_name,
    )
    logger.info("Scoring session: %s", session.workflow_name or session.trace_id)
    raw = await prov.ajudge(JUDGE_SYSTEM_PROMPT, prompt)
    score, explanation = _parse_judge_response(raw)
    session.session_score = score
    session.session_score_explanation = explanation
    logger.info("Session score: %.1f/5 — %s", score, explanation)
    return score, explanation
