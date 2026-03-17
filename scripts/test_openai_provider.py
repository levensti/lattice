"""ReAct-style multi-agent demo using Lattice + OpenAI.

Run with:

    export OPENAI_API_KEY="sk-..."
    uv run python scripts/react_openai_demo.py

This will:
- run a small ReAct-style loop with several @action-decorated subagents
- record a traced session in SQLite
- score each action and the overall session using an OpenAI-backed judge
- print the final answer, session score, and bottlenecks
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

from openai import OpenAI

from lattice import (
    action,
    JudgeConfig,
    trace_session,
    trace_iterations,
    trace_parallel,
    traces,
    OpenAIProvider,
)
from lattice.bottleneck import find_bottlenecks


# ── OpenAI configuration ─────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment before running this script.")

MODEL = "gpt-5-mini-2025-08-07"

client = OpenAI(api_key=OPENAI_API_KEY)


# ── LLM helper ───────────────────────────────────────────────────────────────


def chat(*, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ── Lattice judge provider ───────────────────────────────────────────────────

judge_provider = OpenAIProvider(
    model=MODEL,
    api_key=OPENAI_API_KEY,
)


# ── Scoring configuration ────────────────────────────────────────────────────

SESSION_RUBRIC = """You are a strict technical judge scoring an answer from 1–5.

Score 1: Completely incorrect, off-topic, or missing.
Score 2: Major inaccuracies or omissions; barely useful.
Score 3: Mostly correct but with notable gaps or lack of depth.
Score 4: Correct and helpful with minor gaps.
Score 5: Exceptionally clear, thorough, and accurate.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""


# ── Subagents / tools ────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a helpful research assistant.
Follow the instructions carefully and reason step by step when useful.
"""


@action(
    goal="Produce a set of concrete sub-goals from the user's query",
    role="planner",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
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
    role="router",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
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


def _tool_web_search(query: str) -> str:
    return chat(
        system_prompt="You are a search abstraction.",
        user_prompt=(
            "Pretend you are a web search tool that returns concise bullet points.\n"
            f"Query: {query}\n"
            "Return 3-5 bullet points of relevant information. Do NOT browse the real web."
        ),
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
    role="retriever",
    tags=["parallel", "multi-source"],
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
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
    role="writer",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
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


# ── ReAct-style orchestrator ─────────────────────────────────────────────────


def _log_action(name: str, *, done: bool = False) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    tag = "DONE" if done else "START"
    print(f"  [{ts} UTC] {tag:>5}  {name}")


MAX_ITERATIONS = 3


@action(
    goal="Run a ReAct-style loop to answer the user's question",
    role="orchestrator",
    judge=JudgeConfig(system_prompt=SESSION_RUBRIC, provider=judge_provider),
)
def react_orchestrator(user_query: str) -> Dict[str, Any]:
    """Coordinate planner, router, retriever, and synthesizer in a small ReAct loop."""
    scratchpad = ""
    history: list[Dict[str, Any]] = []

    _log_action("planner")
    plan = planner(user_query)
    _log_action("planner", done=True)
    scratchpad += f"PLAN:\n{plan}\n\n"
    history.append({"step": "plan", "content": plan})

    notes: Dict[str, str] = {}

    for i in trace_iterations("react_loop", range(MAX_ITERATIONS)):
        _log_action(f"tool_router (iter {i})")
        route_raw = tool_router(scratchpad)
        _log_action(f"tool_router (iter {i})", done=True)
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
            _log_action(f"multi_source_retriever (iter {i})")
            notes = multi_source_retriever(user_query)
            _log_action(f"multi_source_retriever (iter {i})", done=True)
            history.append({"step": f"retrieval_{i}", "content": notes})
            scratchpad += f"[loop {i}] RETRIEVAL_NOTES:\n{json.dumps(notes, indent=2)}\n\n"
        else:
            scratchpad += f"[loop {i}] Router did not request known tools; stopping.\n"
            break

    _log_action("synthesizer")
    final_answer = synthesizer(user_query, notes=notes, scratchpad=scratchpad)
    _log_action("synthesizer", done=True)
    history.append({"step": "final_answer", "content": final_answer})

    return {
        "plan": plan,
        "history": history,
        "scratchpad": scratchpad,
        "answer": final_answer,
    }


def run_react_session(query: str):
    with trace_session(
        goal="Answer the user's research question accurately and comprehensively",
        workflow_name="ReAct multi-agent demo (OpenAI)",
        judge=judge_provider,
        judge_system_prompt=SESSION_RUBRIC,
    ) as session:
        result = react_orchestrator(query)
    return session, result


def main() -> None:
    user_query = "How could I use retrieval-augmented generation to debug complex multi-agent systems?"

    print("Running ReAct-style session...\n")
    session, result = run_react_session(user_query)

    print("=== FINAL ANSWER (truncated) ===\n")
    print(result["answer"][:2000])
    print("\n" + "=" * 80 + "\n")

    # Action scores were computed inline by @action(judge=...);
    # session trajectory score was computed on trace_session exit.
    print("Action scores:")
    for a in session.actions:
        score = f"{a.score}/5" if a.score is not None else "n/a"
        print(f"  {a.name}: {score}")

    if session.session_score is not None:
        print(f"\nSession score: {session.session_score:.1f}/5")
        print(f"Explanation: {session.session_score_explanation}")

    print("\nBottlenecks:")
    b_list = list(find_bottlenecks(session))
    if not b_list:
        print("  None detected")
    else:
        for b in b_list:
            print(f"  {b.action_name}: {b.score}/5 ({b.impact}) — {b.explanation}")

    print("\nRecent traces:")
    for t in traces(last=3):
        print(f"- {t.trace_id} | {t.workflow_name} | actions={len(t.actions)} | score={t.session_score}")

    print(
        "\nTo explore visually, run:\n"
        "  uv run python -m lattice dashboard\n"
        "and open http://localhost:8080 in your browser."
    )


if __name__ == "__main__":
    main()
