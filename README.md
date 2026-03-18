# Lattice

Quality debugging for multi-agent systems. Trace every action, score with LLM-as-a-judge, find exactly where quality breaks down.

```python
import os
from lattice import action, trace_session, score_trace, OpenAIProvider

@action(goal="Return factual claims with sources")
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@action(goal="Write a structured article with intro, body, and conclusion")
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

score_trace(session, provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]))
print(session)
```

```
============================================================
  Trace Summary  (trace_id=7f3a...)
  Goal: Produce a well-researched article
============================================================
  ├── researcher  OK  823ms  score=4.0
  │   reason: Returns 3 factual claims with Wikipedia and arXiv sources
  └── writer  OK  651ms  score=2.0
      reason: Missing conclusion, body is a single paragraph
------------------------------------------------------------
  Total actions: 2  |  Total time: 1474ms
============================================================
```

- **Zero config to start** — `@action` + `trace_session` is all you need. Traces auto-persist to local SQLite.
- **Any architecture** — sequential pipelines, ReAct loops, parallel fan-outs, state machines, orchestrator-subagent hierarchies.
- **Structural analysis** — find cascading failures, non-converging loops, and weak parallel branches, not just individual bad scores.
- **Non-blocking scoring** — per-action judging runs in background threads. Production scoring uses a daemon queue.
- **One dependency** — just `httpx`. No server, no account, no setup.

## Install

```bash
pip install lattice
```

Requires Python 3.10+. For development:

```bash
git clone https://github.com/your-org/lattice.git
cd lattice
pip install -e ".[dev]"
```

## Quick Start

### 1. Trace your workflow

Annotate functions with `@action` and wrap the workflow in `trace_session`. No API keys needed — traces auto-persist to a local SQLite database.

```python
from lattice import action, trace_session

@action
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@action
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

print(session)
```

This records inputs, outputs, latency, and parent-child relationships for every action. Browse results with `python -m lattice dashboard`.

### 2. Score and find bottlenecks

Add a `goal` to each action and score with any LLM provider:

```python
import os
from lattice import action, trace_session, score_trace, OpenAIProvider

@action(goal="Return at least 3 factual claims with sources")
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@action(goal="Must have introduction, body, and conclusion")
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

# Score every action against its goal
score_trace(session, provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]))

# Find structural issues
for b in session.bottlenecks:
    print(f"{b.action_name}: {b.score:.1f} ({b.impact}) — {b.explanation}")
```

## Tracing

### `@action` — for functions you own

```python
@action(goal="Must cite sources")
def research(topic): ...

# goal is optional when you just want tracing without scoring
@action
def preprocess(data): ...
```

Works on sync and async functions. Works on methods (`self` and `cls` are excluded from traced inputs).

### `trace_action` — for code blocks you can't decorate

```python
with trace_action("api_call", goal="Return relevant results") as ts:
    result = third_party_api.search(query)
    ts.set_output(result)
```

Also supports inline scoring with `judge=`, just like `@action`.

### `instrument()` — for existing functions

Wrap a function without modifying its source:

```python
agent.search = instrument(agent.search, goal="Find documents")
```

## Architecture Support

### Loops

```python
with trace_session(goal="Answer the question accurately") as session:
    for _ in trace_iterations("react", range(10)):
        thought = think(state)
        if thought["action"] == "finish":
            break
        result = act(thought)
        state = observe(result)
```

`session.bottlenecks` detects when loops fail to converge (scores don't improve across iterations).

### Parallel fan-out

```python
with trace_session(goal="Find and summarize relevant results") as session:
    with trace_parallel("search_fanout"):
        results = await asyncio.gather(
            search_web(query), search_db(query), search_cache(query),
        )
    summary = aggregate(results)
```

`session.bottlenecks` detects when one branch scores significantly worse than the others.

### More patterns

| Pattern                    | How                                                        | Example                        |
| -------------------------- | ---------------------------------------------------------- | ------------------------------ |
| State machine              | `trace_transition(to=..., reason=...)`                     | `state_machine_example.py`     |
| Orchestrator + subagents   | Automatic from the call graph                              | `multi_agent_example.py`       |
| Evaluator-optimizer        | `trace_iterations` with generator + evaluator actions      | `eval_optimizer_example.py`    |
| Blackboard / event-driven  | `trace_activation(reason=...)`                             | —                              |
| Thread-based parallelism   | `copy_trace_context()`                                     | —                              |
| Retrofitting existing code | `instrument()` + `trace_action`                            | `retrofit_example.py`          |

## Scoring

Lattice is unopinionated about scoring — your rubric defines the scale, criteria, and format. There are four ways to score, depending on your needs:

### Per-action scoring (inline, non-blocking)

Attach a `JudgeConfig` to any `@action` or `trace_action`. The action returns immediately; scoring runs in a daemon thread on session exit:

```python
from lattice import JudgeConfig, AnthropicProvider

@action(
    goal="Summarise the paper into 3 bullet points",
    judge=JudgeConfig(
        system_prompt="""You evaluate research summaries.

Score 1: Wrong number of bullets or major factual errors.
Score 3: 3 accurate bullets, one vague or incomplete.
Score 5: 3 tight, distinct, fully accurate bullets.

Respond with JSON: {"reasoning": "...", "score": <1-5>, "explanation": "..."}""",
        provider=AnthropicProvider("claude-opus-4-6", api_key=os.environ["ANTHROPIC_API_KEY"]),
    ),
)
def summarise(paper): ...
```

When `judge=` is set, its `system_prompt` and `provider` override the global judge for that action only.

### Session-level scoring (end-to-end, non-blocking)

Score the entire workflow's output against its goal by passing `judge=` to `trace_session`. The session is persisted immediately, then re-persisted with the score once scoring completes in the background:

```python
provider = OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

with trace_session(goal="Produce a publication-ready article", judge=provider) as session:
    notes = researcher("quantum computing")
    article = writer(notes)
# session.session_score is populated shortly after exit
```

Or score explicitly after the fact:

```python
from lattice import score_session

score, explanation = score_session(session, provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]))
```

### Batch scoring (after the fact)

Score every action in a completed session — useful for offline analysis or re-scoring with a different rubric:

```python
from lattice import score_trace

provider = OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

score_trace(session, provider=provider)

# With a custom global rubric
score_trace(
    session,
    provider=provider,
    system_prompt="You are a strict technical evaluator. ...",
)
```

Use `async_score_trace` for concurrent scoring across actions.

### Background scoring (production)

For production workloads, use `BackgroundScorer` to score off the critical path entirely. It scores all actions and the session end-to-end, then persists results:

```python
from lattice import BackgroundScorer, OpenAIProvider

scorer = BackgroundScorer(provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]))
await scorer.start()

# Per-request hot path — submit() is non-blocking
async def handle_request(query):
    with trace_session(goal="...") as session:
        result = await run_agent(query)
    scorer.submit(session)   # returns immediately
    return result

# At shutdown
await scorer.cancel()
```

Or as an async context manager:

```python
async with BackgroundScorer(provider=OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])) as scorer:
    await serve_forever(scorer)
```

## Providers

### Built-in

| Class                | Wire protocol                 |
| -------------------- | ----------------------------- |
| `OpenAIProvider`     | Chat Completions or Responses |
| `AnthropicProvider`  | Messages                      |
| `OpenRouterProvider` | Chat Completions              |

```python
import os
from lattice import OpenAIProvider, AnthropicProvider, OpenRouterProvider

OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
AnthropicProvider("claude-sonnet-4-20250514", api_key=os.environ["ANTHROPIC_API_KEY"])
OpenRouterProvider("google/gemini-2.0-flash", api_key=os.environ["OPENROUTER_API_KEY"])
```

`OpenAIProvider` supports custom endpoints (Fireworks, Together, etc.) via `api_base=` and the newer Responses API via `api_type=ApiType.RESPONSES`.

### Custom providers

Subclass `InferenceProvider`, set `api_type`, and implement the matching method pair:

```python
from lattice import InferenceProvider, ApiType

class MyProvider(InferenceProvider):
    api_type = ApiType.CHAT_COMPLETIONS

    def _chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        return my_llm_call(system_prompt, user_prompt)

    async def _achat_completions(self, system_prompt: str, user_prompt: str) -> str:
        return await my_async_llm_call(system_prompt, user_prompt)
```

| `ApiType`          | Sync method           | Async method           |
| ------------------ | --------------------- | ---------------------- |
| `CHAT_COMPLETIONS` | `_chat_completions()` | `_achat_completions()` |
| `RESPONSES`        | `_responses()`        | `_aresponses()`        |
| `MESSAGES`         | `_messages()`         | `_amessages()`         |

## Bottleneck Analysis

After scoring, find structural issues that individual scores don't surface:

```python
for b in session.bottlenecks:
    print(f"{b.action_name}: score={b.score:.1f}, impact={b.impact}")
    print(f"  {b.explanation}")
```

Or call directly:

```python
from lattice import find_bottlenecks
results = find_bottlenecks(session)
```

| Impact                  | Meaning                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `"error"`               | Action raised an exception                                           |
| `"lowest_score"`        | Worst-scoring action, more than 1 stddev below the session mean      |
| `"quality_cascade"`     | Low-scoring parent whose output degraded a child action's quality    |
| `"loop_no_convergence"` | Scores didn't improve across loop iterations                         |
| `"weakest_branch"`      | Worst parallel branch, significantly below group average             |

## Storage & Dashboard

Traces are automatically saved to a local SQLite database (`~/.lattice/traces.db`) when a `trace_session` exits — no setup required.

### Querying traces

```python
import lattice

lattice.traces()                         # all traces, most recent first
lattice.traces(last=5)                   # 5 most recent
lattice.traces(workflow="summarize")     # filter by workflow name
lattice.traces(trace_id="abc123")        # specific trace
```

### Dashboard

Browse traces visually with the built-in local dashboard:

```bash
python -m lattice dashboard              # starts at http://localhost:8080
python -m lattice dashboard --port 3000  # custom port
```

Seed the store with demo data to explore:

```bash
python scripts/seed_sqllite_traces.py
python -m lattice dashboard
```

### Configuration

```python
import lattice

# Change the database location
lattice.configure(db_path="/custom/path/traces.db")

# Disable auto-persist for a specific session
with trace_session(goal="...", persist=False) as session:
    ...
```

### Custom storage

The default `SQLiteStore` can be swapped for any backend that subclasses `Store`:

```python
from lattice.storage import Store
from lattice import configure

class PostgresStore(Store):
    def save_session(self, session): ...
    def load_sessions(self, *, workflow=None, last=None, trace_id=None): ...

configure(backend=PostgresStore(os.environ["DATABASE_URL"]))
```

## Examples

All examples run without API keys (using mock LLM calls) unless noted otherwise.

| What you want to do                 | Example                       | Key concepts                                          |
| ----------------------------------- | ----------------------------- | ----------------------------------------------------- |
| Get started with a pipeline         | `multi_agent_example.py`      | `@action`, `score_trace`, `find_bottlenecks`          |
| Add tracing to existing code        | `retrofit_example.py`         | `instrument()`, `trace_action`                        |
| Trace a ReAct loop                  | `react_loop_example.py`       | `trace_iterations`                                    |
| Trace parallel execution            | `parallel_fanout_example.py`  | `trace_parallel`, `asyncio.gather`                    |
| Trace a state machine               | `state_machine_example.py`    | `trace_transition`                                    |
| Generator + evaluator loop          | `eval_optimizer_example.py`   | `trace_iterations` with early `break`                 |
| Full demo with real LLM calls       | `scored_react_demo.py`        | Per-action `JudgeConfig`, `judged_session` (needs API key) |

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a PR.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
