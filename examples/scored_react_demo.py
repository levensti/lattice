"""End-to-end ReAct demo with per-action scoring using Lattice + OpenAI.

Runs a small ReAct-style loop with several @action-decorated subagents,
records a traced session in SQLite, scores each action and the overall
session using an OpenAI-backed judge, and prints the results.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/scored_react_demo.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from lattice import (
    JudgeConfig,
    OpenAIProvider,
    action,
    find_bottlenecks,
    judged_session,
    trace_iterations,
    trace_parallel,
)


# ── Rubrics ──────────────────────────────────────────────────────────────────

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


# ── OpenAI configuration ────────────────────────────────────────────────────

MODEL = "gpt-4o"

client = OpenAI()


# ── LLM helper ──────────────────────────────────────────────────────────────


def chat(*, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ── Lattice judge provider ──────────────────────────────────────────────────

judge_provider = OpenAIProvider(model=MODEL, api_key=os.environ["OPENAI_API_KEY"])


# ── Subagents / tools ───────────────────────────────────────────────────────


@action(
    goal="Produce a set of concrete sub-goals from the user's query",
    judge=JudgeConfig(system_prompt=PLANNER_RUBRIC, provider=judge_provider),
)
def planner(user_query: str) -> str:
    return chat(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=(
            "You are a planning agent.\n"
            f"User query: {user_query}\n\n"
            "Break this into 3-5 numbered sub-goals for a research pipeline."
        ),
    )


@action(
    goal="Choose which tools to call next given the current scratchpad",
    judge=JudgeConfig(system_prompt=ROUTER_RUBRIC, provider=judge_provider),
)
def tool_router(scratchpad: str) -> str:
    return chat(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=(
            "You are a tool router in a ReAct loop.\n"
            "Given the current scratchpad and remaining questions, decide which tools to call next.\n"
            "Available tools: web_search, code_search, doc_summarizer.\n\n"
            f"Scratchpad:\n{scratchpad}\n\n"
            'Respond with a JSON object with keys: "thought", "tools" (list of tool names to call).\n'
        ),
    )


@action(
    goal="Search the web for relevant information",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
)
def _tool_web_search(query: str) -> str:
    return chat(
        system_prompt="You are a search abstraction.",
        user_prompt=(
            "Pretend you are a web search tool that returns concise bullet points.\n"
            f"Query: {query}\n"
            "Return 3-5 bullet points of relevant information. Do NOT browse the real web."
        ),
    )


@action(
    goal="Search the codebase for relevant APIs and modules",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
)
def _tool_code_search(query: str) -> str:
    return chat(
        system_prompt="You are a code search abstraction.",
        user_prompt=(
            "Pretend you are a code search tool over a large codebase.\n"
            f"Query: {query}\n"
            "Return a few bullet points describing relevant APIs or modules that might help."
        ),
    )


@action(
    goal="Summarize documentation into key takeaways",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
)
def _tool_doc_summarizer(text: str) -> str:
    return chat(
        system_prompt="You are a documentation summarizer.",
        user_prompt=(
            "Pretend you are a documentation summarizer.\n"
            "Summarize the following text into 3-4 key takeaways:\n\n"
            f"{text}"
        ),
    )


@action(
    goal="Gather multi-source evidence relevant to the query",
    judge=JudgeConfig(system_prompt=RETRIEVER_RUBRIC, provider=judge_provider),
)
def multi_source_retriever(user_query: str) -> Dict[str, str]:
    """Fan out to multiple tools in a traced parallel block and return combined notes."""
    notes: Dict[str, str] = {}

    with trace_parallel("multi_source_retrieval"):
        notes["web_search_primary"] = _tool_web_search(user_query)
        notes["web_search_secondary"] = _tool_web_search(user_query + " practical examples")
        notes["code_search"] = _tool_code_search(user_query)

    return notes


@action(
    goal="Synthesize a final answer from notes and reasoning",
    judge=JudgeConfig(system_prompt=WRITER_RUBRIC, provider=judge_provider),
)
def synthesizer(user_query: str, notes: Dict[str, str], scratchpad: str) -> str:
    return chat(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=(
            "You are the final synthesis agent in a ReAct-style pipeline.\n"
            "You have:\n"
            f"- User query: {user_query}\n"
            f"- Tool notes (JSON):\n{json.dumps(notes, indent=2)}\n"
            f"- Scratchpad of previous reasoning:\n{scratchpad}\n\n"
            "Write a clear, structured answer that directly addresses the user's query.\n"
            "Use headings and bullet points where helpful."
        ),
    )


# ── ReAct-style orchestrator ────────────────────────────────────────────────

MAX_ITERATIONS = 3


@action(
    goal="Run a ReAct-style loop to answer the user's question",
    judge=JudgeConfig(system_prompt=ORCHESTRATOR_RUBRIC, provider=judge_provider),
)
def react_orchestrator(user_query: str) -> Dict[str, Any]:
    """Coordinate planner, router, retriever, and synthesizer in a small ReAct loop."""
    scratchpad = ""
    history: list[Dict[str, Any]] = []

    plan = planner(user_query)
    scratchpad += f"PLAN:\n{plan}\n\n"
    history.append({"step": "plan", "content": plan})

    notes: Dict[str, str] = {}

    for i in trace_iterations("react_loop", range(MAX_ITERATIONS)):
        route_raw = tool_router(scratchpad)
        history.append({"step": f"route_{i}", "content": route_raw})

        try:
            route_obj = json.loads(route_raw)
        except json.JSONDecodeError:
            scratchpad += f"[loop {i}] Router produced invalid JSON, stopping.\n"
            break

        thought = route_obj.get("thought", "")
        tools = route_obj.get("tools", [])
        scratchpad += f"[loop {i}] THOUGHT: {thought}\n"
        scratchpad += f"[loop {i}] TOOLS: {tools}\n"

        if any(t in {"web_search", "code_search", "doc_summarizer"} for t in tools):
            notes = multi_source_retriever(user_query)
            history.append({"step": f"retrieval_{i}", "content": notes})
            scratchpad += f"[loop {i}] RETRIEVAL_NOTES:\n{json.dumps(notes, indent=2)}\n\n"
        else:
            scratchpad += f"[loop {i}] Router did not request known tools; stopping.\n"
            break

    final_answer = synthesizer(user_query, notes=notes, scratchpad=scratchpad)
    history.append({"step": "final_answer", "content": final_answer})

    return {
        "plan": plan,
        "history": history,
        "scratchpad": scratchpad,
        "answer": final_answer,
    }


def main() -> None:
    user_query = "How could I use retrieval-augmented generation to debug complex multi-agent systems?"

    with judged_session(
        goal="Answer the user's research question accurately and comprehensively",
        workflow_name="ReAct multi-agent demo",
        judge=judge_provider,
        judge_system_prompt=SESSION_RUBRIC,
    ) as session:
        result = react_orchestrator(user_query)

    print(session)

    print(f"Session score: {session.session_score}/5")
    print(f"Explanation:   {session.session_score_explanation}")

    bottlenecks = find_bottlenecks(session)
    if bottlenecks:
        print("\nBottlenecks:")
        for b in bottlenecks:
            print(f"  {b.action_name}: {b.score}/5 ({b.impact}) - {b.explanation}")


if __name__ == "__main__":
    main()
