from __future__ import annotations

from typing import Protocol


JUDGE_SYSTEM_PROMPT = (
    "You are a quality judge for an action in a multi-agent system. "
    "Your job is to assess whether the output of this action meets its goal. "
    "Be objective, concise, and fair."
)

MAX_FIELD_CHARS = 4000

RATING_SCALE = (
    "  1 = Completely fails the goal\n"
    "  2 = Major issues, mostly unusable\n"
    "  3 = Acceptable but with notable gaps\n"
    "  4 = Good with only minor issues\n"
    "  5 = Excellent, fully meets the goal\n"
)

RESPONSE_FORMAT = (
    'Respond with ONLY a JSON object in this exact format (no markdown fencing):\n'
    '{"reasoning": "<think step by step before scoring>", "score": <number 1-5>, "explanation": "<one or two sentences>"}'
)


class ActionPromptBuilder(Protocol):
    """Callable that builds the user prompt for judging a single action."""

    def __call__(
        self,
        *,
        name: str,
        description: str,
        goal: str,
        input_data: str,
        output_data: str,
    ) -> str: ...


class SessionPromptBuilder(Protocol):
    """Callable that builds the user prompt for judging a session."""

    def __call__(
        self,
        *,
        goal: str,
        final_output: str,
        workflow_name: str,
    ) -> str: ...


def build_judge_prompt(
    *,
    name: str,
    description: str,
    goal: str,
    input_data: str,
    output_data: str,
    criteria: dict[str, str] | None = None,
    reference: str | None = None,
) -> str:
    """Build the user prompt sent to the judge LLM for a single action.

    When *criteria* are provided the judge evaluates each one and returns
    per-criterion scores in a single call. When *reference* is provided it is
    injected as a calibration anchor for the judge.
    """
    goal_block = goal or "Assess overall quality, relevance, and correctness."

    parts: list[str] = [
        "Evaluate the following action output.\n",
        f"**Action name:** {name}",
        f"**Description:** {description or 'No description provided.'}\n",
        f"**Action goal:**\n{goal_block}\n",
        f"**Input to the action:**\n{input_data[:MAX_FIELD_CHARS]}\n",
        f"**Output from the action:**\n{output_data[:MAX_FIELD_CHARS]}\n",
    ]

    if reference:
        parts.append(f"**Reference answer (ground truth):**\n{reference[:MAX_FIELD_CHARS]}\n")

    if criteria:
        parts.append("**Evaluation criteria:**")
        for crit_name, crit_desc in criteria.items():
            parts.append(f"  - {crit_name}: {crit_desc}")
        parts.append("")
        parts.append("Rate the output on a scale of 1 to 5 for **each criterion**:\n" + RATING_SCALE)

        # Build per-criterion placeholders for the response format
        criteria_json = ", ".join(
            f'"{n}": {{"score": <1-5>, "explanation": "<sentence>"}}'
            for n in criteria
        )
        response_format = (
            'Respond with ONLY a JSON object in this exact format (no markdown fencing):\n'
            f'{{"reasoning": "<think step by step before scoring>", "criteria": {{{criteria_json}}}, '
            '"score": <overall average 1-5>, "explanation": "<one or two sentences overall>"}}'
        )
    else:
        parts.append("Rate the quality of this output on a scale of 1 to 5:\n" + RATING_SCALE)
        response_format = RESPONSE_FORMAT

    parts.append(response_format)
    return "\n".join(parts)


def build_session_judge_prompt(
    *,
    goal: str,
    final_output: str,
    workflow_name: str = "",
) -> str:
    """Build the user prompt for judging the overall session outcome."""
    name_line = f"**Workflow:** {workflow_name}\n" if workflow_name else ""

    return (
        f"Evaluate whether the following workflow achieved its goal.\n\n"
        f"{name_line}"
        f"**Goal:** {goal}\n\n"
        f"**Final output:**\n{final_output[:MAX_FIELD_CHARS]}\n\n"
        f"Rate how well the final output achieves the goal on a scale of 1 to 5:\n"
        f"{RATING_SCALE}\n"
        f"{RESPONSE_FORMAT}"
    )
