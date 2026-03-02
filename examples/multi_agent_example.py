"""Example: Multi-agent pipeline with agent-trace quality debugging.

Simulates a research → write → edit pipeline and shows how to trace,
score, and find bottlenecks.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/multi_agent_example.py
"""

from agent_trace import (
    configure,
    find_bottlenecks,
    score_trace,
    trace_agent,
    trace_session,
    trace_tool,
)

configure(
    judge_model="gpt-4o",
    otel_enabled=False,
)


@trace_tool(
    name="web_search",
    description="Search the web for information",
    criteria="Must return results relevant to the query with at least 2 distinct sources",
)
def web_search(query: str) -> list[dict]:
    return [
        {"title": "Python docs", "url": "https://docs.python.org", "snippet": "Official docs"},
        {"title": "Real Python", "url": "https://realpython.com", "snippet": "Tutorials"},
    ]


@trace_agent(
    name="researcher",
    description="Researches a topic by searching and summarizing findings",
    criteria="Must synthesize information from multiple sources into a coherent summary with citations",
)
def researcher(topic: str) -> str:
    results = web_search(topic)
    sources = ", ".join(r["title"] for r in results)
    return f"Research on '{topic}': Based on {sources}, Python is a versatile language."


@trace_agent(
    name="writer",
    description="Writes a polished article from research notes",
    criteria="Must produce a well-structured article with introduction, body, and conclusion",
)
def writer(research: str) -> str:
    # Intentionally weak output to demonstrate bottleneck detection
    return "Python is good."


@trace_agent(
    name="editor",
    description="Reviews and improves the article",
    criteria="Must fix grammar, improve clarity, and ensure the article is publication-ready",
)
def editor(article: str) -> str:
    return (
        f"Edited: {article} Python is a versatile, beginner-friendly "
        "programming language used worldwide."
    )


def main():
    with trace_session() as session:
        research = researcher("Python programming")
        draft = writer(research)
        final = editor(draft)

    print(f"Final output: {final}\n")
    print(f"Traced {len(session.steps)} steps\n")

    try:
        score_trace(session)

        print("=== Step Scores ===")
        for step in session.steps:
            print(f"  [{step.step_index}] {step.name}: {step.score}/5 — {step.score_explanation}")

        print("\n=== Bottleneck Analysis ===")
        for b in find_bottlenecks(session):
            print(f"  {b.step_name} (score={b.score}, impact={b.impact}): {b.explanation}")
    except ValueError as e:
        print(f"Skipping scoring (no API key): {e}")
        print("Set OPENAI_API_KEY to enable LLM-based scoring.")


if __name__ == "__main__":
    main()
