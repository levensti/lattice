"""Example: Retrofitting agent-trace onto existing code.

Shows three ways to add tracing without modifying original source:
  1. instrument() — wrap an existing function
  2. trace_step  — context manager for arbitrary code blocks
  3. @step on a class — self is automatically excluded from inputs

Usage:
    python examples/retrofit_example.py
"""

from agent_trace import instrument, print_trace_summary, step, trace_session, trace_step


# === Existing code (imagine this lives in another module) ================

class ResearchAgent:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.history: list[str] = []

    def search(self, query: str) -> list[str]:
        return [f"Result 1 for '{query}'", f"Result 2 for '{query}'"]

    def summarize(self, results: list[str]) -> str:
        return f"Summary of {len(results)} items using {self.model_name}"

    def run(self, topic: str) -> str:
        results = self.search(topic)
        summary = self.summarize(results)
        self.history.append(summary)
        return summary


def external_api_call(query: str) -> dict:
    """Pretend this is a third-party SDK function you can't modify."""
    return {"status": "ok", "data": [f"API result for '{query}'"]}


# === Adding tracing from the outside ====================================

def main():
    agent = ResearchAgent()

    # 1. instrument() — wrap existing methods without modifying the class
    agent.search = instrument(agent.search, goal="Return relevant search results")
    agent.summarize = instrument(agent.summarize, goal="Produce a coherent summary")

    with trace_session(workflow_name="Retrofit Demo", goal="Complete a research pipeline and fetch supplementary data") as session:
        # 2. trace_step — context manager for code blocks
        with trace_step("agent_run", goal="Complete research pipeline", input_data="Python") as ts:
            result = agent.run("Python")
            ts.set_output(result)

        # trace_step also works for third-party calls
        with trace_step("api_call", goal="Fetch supplementary data") as ts:
            api_result = external_api_call("Python advanced")
            ts.set_output(api_result)

    print_trace_summary(session)
    print(f"Agent result: {result}")
    print(f"API result: {api_result}")

    # 3. @step on a class method — self is excluded from traced inputs
    print("\n--- Class method decoration ---")

    class TracedAgent:
        def __init__(self):
            self.state = {"large": "object", "with": "lots", "of": "fields"}

        @step(goal="Process the input efficiently")
        def process(self, data: str) -> str:
            return f"Processed: {data}"

    with trace_session(workflow_name="Class Method Demo", goal="Process input through a traced class method") as session:
        agent2 = TracedAgent()
        agent2.process("hello")

    print_trace_summary(session)
    print(f"Input recorded: {session.steps[0].input_data}")
    print("(Note: 'self' is excluded from the traced input)")


if __name__ == "__main__":
    main()
