"""Example: State-machine / graph architecture with Lattice.

A router dispatches to different agents based on the current state.
Transitions are auto-recorded from the call graph — use trace_transition
only when you want to annotate *why* a particular path was taken.

Usage:
    python examples/state_machine_example.py
"""

import lattice

from lattice.context import trace_session, trace_transition
from lattice.decorators import action


@action(goal="Check input for required fields and format")
def validate(data: dict) -> dict:
    errors = []
    if not data.get("name"):
        errors.append("missing name")
    if not data.get("email"):
        errors.append("missing email")
    return {"valid": len(errors) == 0, "errors": errors, "data": data}


@action(goal="Add derived fields to the validated record")
def enrich(data: dict) -> dict:
    data["username"] = data["name"].lower().replace(" ", "_")
    data["domain"] = data["email"].split("@")[-1]
    return data


@action(goal="Persist the record successfully")
def save(data: dict) -> dict:
    return {"saved": True, "id": 42, **data}


@action(goal="Produce a helpful error response")
def handle_error(errors: list[str]) -> dict:
    return {"saved": False, "errors": errors, "suggestion": "Please fix the errors and retry."}


@action(goal="Route to the correct next step based on state")
def router(state: dict) -> dict:
    if state.get("step") == "start":
        result = validate(state["data"])
        if result["valid"]:
            enriched = enrich(result["data"])
            return save(enriched)
        else:
            trace_transition(to="handle_error", reason=f"validation failed: {result['errors']}")
            return handle_error(result["errors"])

    return {"error": f"Unknown step: {state.get('step')}"}


def main():
    lattice.configure(db_path="./.lattice/traces.db")
    print("--- Valid input ---")
    with trace_session(workflow_name="State Machine (valid)", goal="Process the input record through validation, enrichment, and persistence") as session:
        result = router({"step": "start", "data": {"name": "Alice", "email": "alice@example.com"}})
    print(session)
    print(f"Result: {result}\n")

    print("--- Invalid input ---")
    with trace_session(workflow_name="State Machine (invalid)", goal="Correctly reject invalid input with helpful error messages") as session:
        result = router({"step": "start", "data": {"name": "", "email": ""}})
    print(session)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
