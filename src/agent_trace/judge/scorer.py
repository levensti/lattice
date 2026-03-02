from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..config import get_config
from ..context import StepRecord, TraceSession
from .prompt_builder import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from .providers import create_provider


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
        criteria=step.criteria,
        input_data=step.input_data,
        output_data=step.output_data,
    )


def _score_single_step(step: StepRecord, provider: Any) -> None:
    raw = provider.judge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)


async def _async_score_single_step(step: StepRecord, provider: Any) -> None:
    raw = await provider.ajudge(JUDGE_SYSTEM_PROMPT, _build_prompt_for_step(step))
    step.score, step.score_explanation = _parse_judge_response(raw)


def _get_provider(provider: Any | None):
    if provider is not None:
        return provider
    cfg = get_config()
    if not cfg.judge_api_key:
        raise ValueError(
            "No judge API key configured. "
            "Call configure(judge_api_key=...) or set OPENAI_API_KEY."
        )
    return create_provider(
        cfg.judge_provider, cfg.judge_api_key, cfg.judge_model, cfg.judge_api_base,
    )


def score_trace(
    session: TraceSession,
    *,
    provider: Any | None = None,
) -> list[StepRecord]:
    """Score every step in the session synchronously. Updates steps in place."""
    prov = _get_provider(provider)
    for step in session.steps:
        if step.error is None:
            _score_single_step(step, prov)
    return session.steps


async def async_score_trace(
    session: TraceSession,
    *,
    provider: Any | None = None,
    max_concurrency: int = 5,
) -> list[StepRecord]:
    """Score every step concurrently. Updates steps in place."""
    prov = _get_provider(provider)
    sem = asyncio.Semaphore(max_concurrency)
    scorable = [s for s in session.steps if s.error is None]

    async def _bounded(step: StepRecord) -> None:
        async with sem:
            await _async_score_single_step(step, prov)

    await asyncio.gather(*[_bounded(s) for s in scorable])
    return session.steps
