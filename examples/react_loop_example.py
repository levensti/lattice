"""Example: ReAct loop architecture with agent-trace.

Simulates a single agent in a think -> act -> observe cycle, calling
tools iteratively until it reaches an answer.

Usage:
    python examples/react_loop_example.py
"""

from agent_trace import (
    print_trace_summary,
    step,
    trace_iterations,
    trace_session,
)


@step(goal="Decide which tool to call next", role="think")
def think(state: dict) -> dict:
    if not state.get("search_done"):
        return {"action": "search", "query": state["question"]}
    if not state.get("verified"):
        return {"action": "verify", "claim": state["answer"]}
    return {"action": "finish", "result": state["answer"]}


@step(goal="Execute the chosen tool accurately", role="act")
def act(plan: dict) -> dict:
    if plan["action"] == "search":
        return {"type": "search_result", "data": f"Results for: {plan['query']}"}
    if plan["action"] == "verify":
        return {"type": "verification", "data": f"Verified: {plan['claim']}"}
    return {"type": "final", "data": plan["result"]}


@step(goal="Update state correctly from tool output", role="observe")
def observe(state: dict, result: dict) -> dict:
    new_state = dict(state)
    if result["type"] == "search_result":
        new_state["answer"] = f"Python is a programming language. ({result['data']})"
        new_state["search_done"] = True
    elif result["type"] == "verification":
        new_state["verified"] = True
    return new_state


def main():
    state = {"question": "What is Python?"}

    with trace_session(workflow_name="ReAct Agent") as session:
        iters = trace_iterations("react", range(5))
        for _ in iters:
            plan = think(state)
            if plan["action"] == "finish":
                break
            result = act(plan)
            state = observe(state, result)

    print_trace_summary(session)
    print(f"Completed in {iters.iteration_count} iterations")
    print(f"Final answer: {state.get('answer', 'No answer')}")


if __name__ == "__main__":
    main()
