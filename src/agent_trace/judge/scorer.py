from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ..config import JudgeConfig, _merge_judge_configs, get_config
from ..context import StepRecord, TraceSession
from .prompt_builder import JUDGE_SYSTEM_PROMPT, build_judge_prompt, build_session_judge_prompt
from .providers import create_provider

logger = logging.getLogger("agent_trace")


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


def _resolve_config(
    step_config: JudgeConfig | None,
    call_config: JudgeConfig | None,
) -> JudgeConfig:
    """Merge three layers: step > call > global."""
    global_config = get_config().to_judge_config()
    return _merge_judge_configs(step_config, call_config, global_config)


def _provider_from_config(config: JudgeConfig):
    """Create a provider from a fully-resolved ``JudgeConfig``."""
    if not config.api_key:
        raise ValueError(
            "No judge API key configured. "
            "Call configure(judge_api_key=...) or set OPENAI_API_KEY."
        )
    return create_provider(
        provider_name=config.provider or "openai",
        api_key=config.api_key,
        model=config.model or "gpt-4o",
        api_base=config.api_base or "https://api.openai.com/v1",
        temperature=config.temperature if config.temperature is not None else 0.1,
        top_k=config.top_k,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )


def _get_provider(
    provider: Any | None,
    call_config: JudgeConfig | None,
    step_config: JudgeConfig | None = None,
):
    """Return a provider, respecting the three-layer config hierarchy.

    If a pre-built ``provider`` is passed, it is used as-is (escape hatch).
    Otherwise a provider is constructed from the merged config.
    """
    if provider is not None:
        return provider
    resolved = _resolve_config(step_config, call_config)
    return _provider_from_config(resolved)


def _score_single_step(
    step: StepRecord,
    provider: Any | None,
    call_config: JudgeConfig | None,
) -> None:
    prov = _get_provider(provider, call_config, step.judge_config)
    logger.info("Scoring step: %s", step.name)
    raw = prov.judge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)
    logger.info("Scored step: %s → %.1f/5", step.name, step.score)


async def _async_score_single_step(
    step: StepRecord,
    provider: Any | None,
    call_config: JudgeConfig | None,
) -> None:
    prov = _get_provider(provider, call_config, step.judge_config)
    logger.info("Scoring step: %s", step.name)
    raw = await prov.ajudge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)
    logger.info("Scored step: %s → %.1f/5", step.name, step.score)


# ── Public API ────────────────────────────────────────────────────────


def score_trace(
    session: TraceSession,
    *,
    provider: Any | None = None,
    judge_config: JudgeConfig | None = None,
) -> list[StepRecord]:
    """Score every step in the session synchronously.

    Config resolution per step: ``@step(judge_config=...)`` >
    ``judge_config`` passed here > global ``configure()`` defaults.

    Pass a pre-built ``provider`` to bypass config resolution entirely.
    """
    for step in session.steps:
        if step.error is None:
            _score_single_step(step, provider, judge_config)
    return session.steps


async def async_score_trace(
    session: TraceSession,
    *,
    provider: Any | None = None,
    judge_config: JudgeConfig | None = None,
    max_concurrency: int = 5,
) -> list[StepRecord]:
    """Score every step concurrently. Updates steps in place."""
    sem = asyncio.Semaphore(max_concurrency)
    scorable = [s for s in session.steps if s.error is None]

    async def _bounded(step: StepRecord) -> None:
        async with sem:
            await _async_score_single_step(step, provider, judge_config)

    await asyncio.gather(*[_bounded(s) for s in scorable])
    return session.steps


def score_session(
    session: TraceSession,
    *,
    provider: Any | None = None,
    judge_config: JudgeConfig | None = None,
) -> tuple[float, str]:
    """Judge the final output of the workflow against the session goal.

    Looks at the last step's output and evaluates it against ``session.goal``.
    Returns ``(score, explanation)`` and stores them on the session.
    """
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.steps:
        raise ValueError("Session has no steps — nothing to judge.")

    prov = _get_provider(provider, judge_config)
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
    provider: Any | None = None,
    judge_config: JudgeConfig | None = None,
) -> tuple[float, str]:
    """Async version of :func:`score_session`."""
    if not session.goal:
        raise ValueError("Session has no goal set — nothing to judge against.")
    if not session.steps:
        raise ValueError("Session has no steps — nothing to judge.")

    prov = _get_provider(provider, judge_config)
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
