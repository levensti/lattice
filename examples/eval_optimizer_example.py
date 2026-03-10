"""Example: Evaluator-optimizer loop with Lattice.

A generator produces output, an evaluator scores it, and the generator
retries with feedback until quality is sufficient.

Usage:
    python examples/eval_optimizer_example.py
"""

from lattice import (
    print_trace_summary,
    step,
    trace_iterations,
    trace_session,
)

DRAFTS = [
    "Python good.",
    "Python is a programming language that is popular.",
    "Python is a versatile, high-level programming language known for its "
    "readability and broad ecosystem of libraries.",
]


@step(goal="Produce a clear, informative paragraph about the topic", role="generator")
def generate(topic: str, feedback: str | None = None, attempt: int = 0) -> str:
    idx = min(attempt, len(DRAFTS) - 1)
    return DRAFTS[idx]


@step(goal="Accurately assess quality and provide actionable feedback", role="evaluator")
def evaluate(text: str) -> dict:
    if len(text) > 50:
        return {"pass": True, "score": 4, "feedback": "Good quality."}
    if len(text) > 20:
        return {"pass": False, "score": 2, "feedback": "Too brief, add more detail."}
    return {"pass": False, "score": 1, "feedback": "Unacceptable, rewrite completely."}


def main():
    topic = "Python programming"
    result = None
    feedback = None

    with trace_session(workflow_name="Eval-Optimizer", goal="Produce a clear, informative paragraph that passes quality evaluation") as session:
        iters = trace_iterations("generate_eval", range(5))
        for attempt in iters:
            draft = generate(topic, feedback=feedback, attempt=attempt)
            verdict = evaluate(draft)

            if verdict["pass"]:
                result = draft
                break
            feedback = verdict["feedback"]

    print_trace_summary(session)
    print(f"Converged after {iters.iteration_count} iterations")
    print(f"Final output: {result}")


if __name__ == "__main__":
    main()
