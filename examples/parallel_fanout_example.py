"""Example: Parallel fan-out / fan-in architecture with Lattice.

Multiple search agents run concurrently, and an aggregator synthesizes
the results.

Usage:
    python examples/parallel_fanout_example.py
"""

import asyncio

from lattice import (
    print_trace_summary,
    action,
    trace_parallel,
    trace_session,
)


@action(goal="Return relevant web results", role="searcher")
async def search_web(query: str) -> list[str]:
    await asyncio.sleep(0.01)
    return [f"Web result 1 for '{query}'", f"Web result 2 for '{query}'"]


@action(goal="Return relevant database records", role="searcher")
async def search_db(query: str) -> list[str]:
    await asyncio.sleep(0.01)
    return [f"DB record for '{query}'"]


@action(goal="Return cached results if available", role="searcher")
async def search_cache(query: str) -> list[str]:
    await asyncio.sleep(0.005)
    return []


@action(goal="Merge results from all sources into a coherent summary", role="aggregator")
async def aggregate(results: list[list[str]]) -> str:
    flat = [item for sublist in results for item in sublist]
    if not flat:
        return "No results found."
    return f"Found {len(flat)} results: " + "; ".join(flat)


async def main():
    query = "Python asyncio"

    with trace_session(workflow_name="Parallel Search", goal="Find and summarize relevant results from multiple sources") as session:
        with trace_parallel("search_fanout"):
            results = await asyncio.gather(
                search_web(query),
                search_db(query),
                search_cache(query),
            )

        summary = await aggregate(list(results))

    print_trace_summary(session)
    print(f"Result: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
