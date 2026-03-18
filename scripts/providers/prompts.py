AGENT_SYSTEM_PROMPT = """You are a helpful research assistant.
Follow the instructions carefully and reason step by step when useful.
"""

PLANNER_RUBRIC = """You judge whether the planner produced clear, concrete, and actionable sub-goals.

Score 1: Sub-goals are missing or unrelated to the query.
Score 2: Sub-goals are vague, redundant, or mostly off-target.
Score 3: Sub-goals are somewhat relevant but miss important steps or are poorly structured.
Score 4: Sub-goals are clear, mostly complete, and logically ordered with minor issues.
Score 5: Sub-goals are crisp, non-overlapping, and form a sensible end-to-end research plan.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

ROUTER_RUBRIC = """You judge whether the router chooses appropriate tools given the scratchpad.

Score 1: Tools are clearly wrong or harmful for the query.
Score 2: Tools are mostly irrelevant or miss obviously useful tools.
Score 3: Tools are somewhat reasonable but miss better choices or over-call tools.
Score 4: Tools are appropriate and sufficient for the next step, with minor imperfections.
Score 5: Tool choices are well-justified, minimal, and clearly advance the reasoning.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

RETRIEVER_RUBRIC = """You judge whether the retriever gathers diverse, relevant evidence.

Score 1: Retrieved notes are off-topic or empty.
Score 2: Notes contain some relevant items but are mostly noise or missing key angles.
Score 3: Notes are somewhat relevant but shallow or missing important perspectives.
Score 4: Notes cover the main aspects of the query with good diversity and detail.
Score 5: Notes are rich, highly relevant, and clearly useful for a strong final answer.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

WRITER_RUBRIC = """You judge the quality of the final written answer.

Score 1: Answer is wrong, off-topic, or unusably vague.
Score 2: Answer has major factual gaps or is poorly structured.
Score 3: Answer is mostly correct but shallow, disorganized, or missing key caveats.
Score 4: Answer is clear, mostly complete, and well-structured with minor issues.
Score 5: Answer is deeply informative, well-organized, and directly addresses the query with nuance.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

ORCHESTRATOR_RUBRIC = """You judge how well the orchestrator coordinated the overall ReAct loop.

Score 1: Loop gets stuck, ignores evidence, or produces no meaningful progress.
Score 2: Loop makes some progress but repeats work or ignores obvious signals.
Score 3: Loop is somewhat effective but has unnecessary steps or misses optimizations.
Score 4: Loop sequences steps logically and uses tools reasonably well.
Score 5: Loop is efficient, adaptive, and clearly maximizes the usefulness of each subagent.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

SESSION_RUBRIC = """You are a strict technical judge of the overall workflow outcome.

Score 1: Final answer is unusable or off-topic.
Score 2: Final answer has major issues or misses the core of the query.
Score 3: Final answer is somewhat helpful but incomplete or poorly structured.
Score 4: Final answer is clear, mostly complete, and accurate with minor issues.
Score 5: Final answer is excellent: accurate, comprehensive, and well-organized.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""

