"""Example: Pipeline architecture with Lattice quality debugging.

Simulates a research -> write -> edit pipeline and shows how to trace,
score, and find bottlenecks using the unified ``@action`` decorator.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/multi_agent_example.py
"""

from lattice import (
    find_bottlenecks,
    score_trace,
    action,
    trace_session,
)


@action(goal="Must return results relevant to the query with at least 2 distinct sources", tags=["io", "external"])
def web_search(query: str) -> list[dict]:
    return [
        {"title": "Python docs", "url": "https://docs.python.org", "snippet": "Official docs"},
        {"title": "Real Python", "url": "https://realpython.com", "snippet": "Tutorials"},
    ]


@action(goal="Must synthesize information from multiple sources into a coherent summary with citations", tags=["llm"])
def researcher(topic: str) -> str:
    results = web_search(topic)
    sources = ", ".join(r["title"] for r in results)
    return f"Research on '{topic}': Based on {sources}, Python is a versatile language."


@action(goal="Must produce a well-structured article with introduction, body, and conclusion", tags=["llm"])
def writer(research: str) -> str:
    return "Python is good."


@action(goal="Must fix grammar, improve clarity, and ensure the article is publication-ready", tags=["llm"])
def editor(article: str) -> str:
    return (
        f"Edited: {article} Python is a versatile, beginner-friendly "
        "programming language used worldwide."
    )


def main():
    with trace_session(goal="Produce a high-quality, publication-ready article about the given topic") as session:
        research = researcher("Python programming")
        draft = writer(research)
        final = editor(draft)

    print(f"Final output: {final}\n")
    print(f"Traced {len(session.actions)} actions\n")

    try:
        score_trace(session, model="gpt-4o")

        print("=== Action Scores ===")
        for s in session.actions:
            print(f"  [{s.action_index}] {s.name}: {s.score}/5 — {s.score_explanation}")

        print("\n=== Bottleneck Analysis ===")
        for b in find_bottlenecks(session):
            print(f"  {b.action_name} (score={b.score}, impact={b.impact}): {b.explanation}")
    except ValueError as e:
        print(f"Skipping scoring (no API key): {e}")


if __name__ == "__main__":
    main()
