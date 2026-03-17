"""ReAct-style multi-agent demo using Lattice + Sail.

Run with:

    export SAIL_API_KEY="your-sail-key"
    uv run python scripts/react_sail_react_demo.py

This will:
- run a small ReAct-style loop with several @action-decorated subagents
- record a traced session in SQLite
- score each action and the overall session using a Sail-backed judge
- print the final answer, session score, and bottlenecks
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

import httpx

from lattice import (
    action,
    trace_session,
    trace_iterations,
    trace_parallel,
    score_session,
    traces,
)
from lattice.bottleneck import find_bottlenecks
from lattice.judge.providers import InferenceProvider, ApiType
from lattice.judge.scorer import _score_single_action


# ── Sail configuration ─────────────────────────────────────────────────────────

SAIL_API_KEY = os.environ.get("SAIL_API_KEY")
if not SAIL_API_KEY:
    raise RuntimeError("Please set SAIL_API_KEY in your environment before running this script.")

SAIL_API_BASE = "https://api.sailresearch.com/v1"

AGENT_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-20b"


# ── SailProvider — custom InferenceProvider for Sail's Responses API ──────────


class SailProvider(InferenceProvider):
    """Calls Sail's Responses API (``/v1/responses``).

    Sail supports the OpenAI Responses wire protocol with extra fields:
    - ``background``: whether the request returns immediately for polling.
    - ``metadata.completion_window``: scheduling hint (e.g. ``"asap"``, ``"15m"``).

    See https://docs.sailresearch.com/quickstart for details.
    """

    api_type = ApiType.RESPONSES

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        timeout: float = 120.0,
        temperature: float = 0.1,
        background: bool = False,
        completion_window: str = "asap",
    ):
        if not api_key:
            raise ValueError("api_key is required for SailProvider")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.background = background
        self.completion_window = completion_window

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "background": self.background,
            "metadata": {"completion_window": self.completion_window},
        }

    @staticmethod
    def _extract(resp: httpx.Response) -> str:
        try:
            for block in resp.json()["output"]:
                if block.get("type") == "message":
                    for content in block["content"]:
                        if content.get("type") == "output_text":
                            return content["text"]
            raise ValueError("No output_text block found in Sail response")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Unexpected Sail response from {resp.url}: {resp.text[:500]}"
            ) from exc

    def _responses(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{SAIL_API_BASE}/responses",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return self._extract(resp)

    async def _aresponses(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{SAIL_API_BASE}/responses",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return self._extract(resp)

    def __repr__(self) -> str:
        return f"SailProvider(model={self.model!r})"


# ── Sail helpers ──────────────────────────────────────────────────────────────


def sail_chat_completions(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    """Thin wrapper around Sail's Chat Completions endpoint for agent calls."""
    url = f"{SAIL_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SAIL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Error calling Sail at {url}: {exc}. "
            "Check SAIL_API_BASE and your network connectivity."
        ) from exc


# ── Lattice judge provider (via Sail Responses API) ───────────────────────────

judge_provider = SailProvider(
    model=JUDGE_MODEL,
    api_key=SAIL_API_KEY,
    temperature=0.1,
    background=False,
    completion_window="asap",
)


# ── Subagents / tools ─────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a helpful research assistant.
Follow the instructions carefully and reason step by step when useful.
"""


@action(goal="Produce a set of concrete sub-goals from the user's query", role="planner")
def planner(user_query: str) -> str:
    prompt = (
        "You are a planning agent.\n"
        f"User query: {user_query}\n\n"
        "Break this into 3-5 numbered sub-goals for a research pipeline."
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
    )


@action(goal="Choose which tools to call next given the current scratchpad", role="router")
def tool_router(scratchpad: str) -> str:
    prompt = (
        "You are a tool router in a ReAct loop.\n"
        "Given the current scratchpad and remaining questions, decide which tools to call next.\n"
        "Available tools: web_search, code_search, doc_summarizer.\n\n"
        f"Scratchpad:\n{scratchpad}\n\n"
        "Respond with a JSON object with keys: \"thought\", \"tools\" (list of tool names to call).\n"
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
    )


def _tool_web_search(query: str) -> str:
    prompt = (
        "Pretend you are a web search tool that returns concise bullet points.\n"
        f"Query: {query}\n"
        "Return 3-5 bullet points of relevant information. Do NOT browse the real web."
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt="You are a search abstraction.",
        user_prompt=prompt,
    )


def _tool_code_search(query: str) -> str:
    prompt = (
        "Pretend you are a code search tool over a large codebase.\n"
        f"Query: {query}\n"
        "Return a few bullet points describing relevant APIs or modules that might help."
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt="You are a code search abstraction.",
        user_prompt=prompt,
    )


def _tool_doc_summarizer(text: str) -> str:
    prompt = (
        "Pretend you are a documentation summarizer.\n"
        "Summarize the following text into 3-4 key takeaways:\n\n"
        f"{text}"
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt="You are a documentation summarizer.",
        user_prompt=prompt,
    )


@action(
    goal="Gather multi-source evidence relevant to the query",
    role="retriever",
    tags=["parallel", "multi-source"],
)
def multi_source_retriever(user_query: str) -> Dict[str, str]:
    """Fan out to multiple tools in a traced parallel block and return combined notes."""
    notes: Dict[str, str] = {}

    with trace_parallel("multi_source_retrieval"):
        notes["web_search_primary"] = _tool_web_search(user_query)
        notes["web_search_secondary"] = _tool_web_search(user_query + " practical examples")
        notes["code_search"] = _tool_code_search(user_query)
        # Up to 5 branches; uncomment if you want more parallelism:
        # notes["doc_summary_1"] = _tool_doc_summarizer("Some internal design doc about agents...")
        # notes["doc_summary_2"] = _tool_doc_summarizer("Another doc about evaluation and logging...")

    return notes


@action(goal="Synthesize a final answer from notes and reasoning", role="writer")
def synthesizer(user_query: str, notes: Dict[str, str], scratchpad: str) -> str:
    prompt = (
        "You are the final synthesis agent in a ReAct-style pipeline.\n"
        "You have:\n"
        f"- User query: {user_query}\n"
        f"- Tool notes (JSON):\n{json.dumps(notes, indent=2)}\n"
        f"- Scratchpad of previous reasoning:\n{scratchpad}\n\n"
        "Write a clear, structured answer that directly addresses the user's query.\n"
        "Use headings and bullet points where helpful."
    )
    return sail_chat_completions(
        model=AGENT_MODEL,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
    )


# ── ReAct-style orchestrator ──────────────────────────────────────────────────


def _log_action(name: str, *, done: bool = False) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    tag = "DONE" if done else "START"
    print(f"  [{ts} UTC] {tag:>5}  {name}")


MAX_ITERATIONS = 3


@action(goal="Run a ReAct-style loop to answer the user's question", role="orchestrator")
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


# ── Scoring configuration ─────────────────────────────────────────────────────

SESSION_RUBRIC = """You are a strict technical judge scoring an answer from 1–5.

Score 1: Completely incorrect, off-topic, or missing.
Score 2: Major inaccuracies or omissions; barely useful.
Score 3: Mostly correct but with notable gaps or lack of depth.
Score 4: Correct and helpful with minor gaps.
Score 5: Exceptionally clear, thorough, and accurate.

Respond with JSON: {"score": <1-5>, "explanation": "..."}.
"""


def run_react_session(query: str):
    with trace_session(
        goal="Answer the user's research question accurately and comprehensively",
        workflow_name="ReAct multi-agent demo (Sail)",
    ) as session:
        result = react_orchestrator(query)
    return session, result


def score_entire_session(sess) -> None:
    print("Scoring individual actions with Sail judge...\n")
    for a in sess.actions:
        if a.error is not None:
            print(f"    {a.name}: skipped (error)")
            continue
        _log_action(f"judge → {a.name}")
        _score_single_action(a, judge_provider, SESSION_RUBRIC)
        _log_action(f"judge → {a.name}", done=True)
        print(f"    {a.name}: {a.score}/5")

    print()
    _log_action("judge → session")
    overall_score, explanation = score_session(
        sess,
        provider=judge_provider,
        system_prompt=SESSION_RUBRIC,
    )
    _log_action("judge → session", done=True)
    print(f"\nSession score: {overall_score:.1f}/5")
    print(f"Explanation: {explanation}\n")

    print("Potential bottlenecks:")
    b_list = list(find_bottlenecks(sess))
    if not b_list:
        print("- None detected")
    else:
        for b in b_list:
            print(f"- {b.action_name}: {b.score}/5 ({b.impact}) — {b.explanation}")


def main() -> None:
    user_query = "How could I use retrieval-augmented generation to debug complex multi-agent systems?"

    print("Running ReAct-style session...\n")
    session, result = run_react_session(user_query)

    print("=== FINAL ANSWER (truncated) ===\n")
    print(result["answer"][:2000])
    print("\n" + "=" * 80 + "\n")

    score_entire_session(session)

    print("\nRecent traces:")
    for t in traces(last=3):
        print(f"- {t.trace_id} | {t.workflow_name} | actions={len(t.actions)} | score={t.session_score}")

    print(
        "\nTo explore visually, run:\n"
        "  uv run python -m lattice dashboard\n"
        "and open http://localhost:8787 in your browser."
    )


if __name__ == "__main__":
    main()

