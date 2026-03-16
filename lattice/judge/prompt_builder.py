from __future__ import annotations

from typing import Protocol


JUDGE_SYSTEM_PROMPT = (
    "You are a quality judge for an action in a multi-agent system. "
    "Your job is to assess whether the output of this action meets its goal. "
    "Be objective, concise, and fair.\n\n"
    "Rate the output on a scale of 1 to 5:\n"
    "  1 = Completely fails the goal\n"
    "  2 = Major issues, mostly unusable\n"
    "  3 = Acceptable but with notable gaps\n"
    "  4 = Good with only minor issues\n"
    "  5 = Excellent, fully meets the goal\n\n"
    'Respond with JSON only: {"reasoning": "<think step by step>", "score": <1-5>, "explanation": "<one or two sentences>"}'
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
    'Respond with JSON only: {"reasoning": "<think step by step>", "score": <1-5>, "explanation": "<one or two sentences>"}'
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
) -> str:
    """Build the user prompt sent to the judge LLM for a single action."""
    goal_block = goal or "Assess overall quality, relevance, and correctness."

    return (
        f"Evaluate the following action output.\n\n"
        f"**Action name:** {name}\n"
        f"**Description:** {description or 'No description provided.'}\n\n"
        f"**Action goal:**\n{goal_block}\n\n"
        f"**Input to the action:**\n{input_data[:MAX_FIELD_CHARS]}\n\n"
        f"**Output from the action:**\n{output_data[:MAX_FIELD_CHARS]}"
    )


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
        f"**Final output:**\n{final_output[:MAX_FIELD_CHARS]}"
    )
