from __future__ import annotations

JUDGE_SYSTEM_PROMPT = (
    "You are a quality judge for a step in a multi-agent AI system. "
    "Your job is to assess whether the output of this step meets the given criteria. "
    "Be objective, concise, and fair."
)

MAX_FIELD_CHARS = 4000


def build_judge_prompt(
    *,
    name: str,
    description: str,
    criteria: str,
    input_data: str,
    output_data: str,
) -> str:
    """Build the user prompt sent to the judge LLM."""
    criteria_block = criteria or "Assess overall quality, relevance, and correctness."
    return (
        f"Evaluate the following step output.\n\n"
        f"**Step name:** {name}\n"
        f"**Description:** {description or 'No description provided.'}\n\n"
        f"**Quality criteria:**\n{criteria_block}\n\n"
        f"**Input to the step:**\n{input_data[:MAX_FIELD_CHARS]}\n\n"
        f"**Output from the step:**\n{output_data[:MAX_FIELD_CHARS]}\n\n"
        f"Rate the quality of this output on a scale of 1 to 5:\n"
        f"  1 = Completely fails the criteria\n"
        f"  2 = Major issues, mostly unusable\n"
        f"  3 = Acceptable but with notable gaps\n"
        f"  4 = Good with only minor issues\n"
        f"  5 = Excellent, fully meets criteria\n\n"
        f'Respond with ONLY a JSON object in this exact format (no markdown fencing):\n'
        f'{{"score": <number 1-5>, "explanation": "<one or two sentences>"}}'
    )
