# Lattice

Lattice is a lightweight quality debugging framework for multi-agent systems. Trace each action in your workflow, score it against a defined goal using an LLM judge, and find exactly where quality degrades.

Designed to plug into existing codebases with minimal changes — add a few lines of code and get full visibility into your pipeline in minutes. Perfect for modern agent architectures, including sequential pipelines, ReAct loops, parallel fan-outs, evaluator-optimizer patterns, state machines, and orchestrator-subagent hierarchies.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from lattice import action, trace_session, score_trace, find_bottlenecks, JudgeConfig

@action(goal="Must return at least 3 factual claims with sources")
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@action(goal="Must have introduction, body, and conclusion")
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

score_trace(session, model="gpt-4o")
for b in find_bottlenecks(session):
    print(f"{b.action_name}: {b.score}/5 ({b.impact}) — {b.explanation}")
```

Every `@action` and `trace_session` requires a `goal` — this is what the judge evaluates against. Traces are automatically saved to a local SQLite database so you can query and visualize them later.

## Tracing

### `@action` — for functions you own

```python
@action(goal="Must cite sources")
def research(topic): ...

@action(goal="Decide the next action", role="think")
def think(state): ...
```

Works on sync and async functions. Works on methods (`self` and `cls` are excluded from traced inputs).

### `trace_action` / `instrument()` — for code you don't own

```python
with trace_action("api_call", goal="Return relevant results") as ts:
    result = third_party_api.search(query)
    ts.set_output(result)

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

`find_bottlenecks()` detects when loops fail to converge.

### Parallel fan-out

```python
with trace_session(goal="Find and summarize relevant results") as session:
    with trace_parallel("search_fanout"):
        results = await asyncio.gather(
            search_web(query), search_db(query), search_cache(query),
        )
    summary = aggregate(results)
```

`find_bottlenecks()` detects when one branch scores significantly worse than the others.

### More patterns

| Pattern                    | How                                                        | Example                     |
| -------------------------- | ---------------------------------------------------------- | --------------------------- |
| State machine              | `trace_transition(to=..., reason=...)`                     | `state_machine_example.py`  |
| Orchestrator + subagents   | Automatic from the call graph                              | `multi_agent_example.py`    |
| Evaluator-optimizer        | `trace_iterations` with `role="generator"` / `"evaluator"` | `eval_optimizer_example.py` |
| Blackboard / event-driven  | `trace_activation(reason=...)`                             | —                           |
| Thread-based parallelism   | `copy_trace_context()`                                     | —                           |
| Retrofitting existing code | `instrument()` + `trace_action`                             | `retrofit_example.py`       |

## Scoring

Pass `model=` to `score_trace` — the provider and API key are resolved automatically:

```python
score_trace(session, model="gpt-4o")                   # uses OPENAI_API_KEY
score_trace(session, model="claude-sonnet-4-20250514")  # uses ANTHROPIC_API_KEY
score_trace(session, model="google/gemini-2.0-flash")   # uses OPENROUTER_API_KEY
```

| Model name                      | Routes to  | Env var              |
| ------------------------------- | ---------- | -------------------- |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI     | `OPENAI_API_KEY`     |
| `claude-*`                      | Anthropic  | `ANTHROPIC_API_KEY`  |
| Names containing `/`            | OpenRouter | `OPENROUTER_API_KEY` |
| Anything else                   | OpenRouter | `OPENROUTER_API_KEY` |

You can also pass `api_key=` explicitly or a custom `provider=` instance. Use `async_score_trace` for concurrent scoring.

### Custom rubric per action

Attach a `JudgeConfig` to any `@action` to give that step its own rubric and model. The `system_prompt` is the rubric — define scoring criteria, per-score anchors, and any reference material there:

```python
from lattice import JudgeConfig

@action(
    goal="Summarise the paper into 3 bullet points",
    judge=JudgeConfig(
        model="claude-opus-4-6",
        system_prompt="""You evaluate research summaries.

Score 1: Wrong number of bullets or major factual errors.
Score 2: 3 bullets but one is factually wrong.
Score 3: 3 accurate bullets, one vague or incomplete.
Score 4: 3 accurate bullets, minor wording issues.
Score 5: 3 tight, distinct, fully accurate bullets.

Respond with JSON only: {"reasoning": "...", "score": <1-5>, "explanation": "..."}""",
    ),
)
def summarise(paper): ...
```

When a `judge=` is set on an action, its `system_prompt` and model override the global judge for that step only. Actions without a `judge=` fall back to the global judge passed to `score_trace`.

### Global system prompt

Override the default rubric for all actions at once:

```python
score_trace(
    session,
    model="gpt-4o",
    system_prompt="""You are a strict technical evaluator. Score 1-5.
Respond: {"reasoning": "...", "score": <1-5>, "explanation": "..."}""",
)
```

### Background scoring (production)

Score off the critical path so the judge never blocks your request handler:

```python
scorer = BackgroundScorer(model="gpt-4o")
await scorer.start()

async def handle_request(query):
    with trace_session(goal="...") as session:
        result = await run_agent(query)
    scorer.submit(session)   # non-blocking
    return result

await scorer.cancel()        # at shutdown
```

## Bottleneck Analysis

```python
for b in find_bottlenecks(session):
    print(f"{b.action_name}: score={b.score}, impact={b.impact}")
```

| Impact                  | Meaning                                                  |
| ----------------------- | -------------------------------------------------------- |
| `"error"`               | Action raised an exception                               |
| `"lowest_score"`        | Worst-scoring action in the session                      |
| `"largest_drop"`        | Biggest quality drop from the preceding action           |
| `"below_average"`       | Below the session average                                |
| `"loop_no_convergence"` | Scores didn't improve across loop iterations             |
| `"weakest_branch"`      | Worst parallel branch, significantly below group average |

## Trace Persistence & Dashboard

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
python -m lattice dashboard              # starts at http://127.0.0.1:8787
python -m lattice dashboard --port 8080  # custom port
```

### Configuration

```python
# Change the database location
lattice.configure(db_path="/custom/path/traces.db")

# Disable auto-persist for a specific session
with trace_session(goal="...", persist=False) as session:
    ...
```

### Custom storage

The default `SQLiteStore` can be swapped for any class that implements the `Store` protocol (`lattice.storage`):

```python
from lattice.storage import Store
from lattice import configure

class PostgresStore(Store):
    def save_session(self, session): ...
    def load_sessions(self, *, workflow=None, last=None, trace_id=None): ...

configure(backend=PostgresStore(os.environ["DATABASE_URL"]))
```

`SQLiteStore` is also importable directly if you need to instantiate it explicitly:

```python
from lattice.storage import SQLiteStore

configure(backend=SQLiteStore("/custom/path/traces.db"))
```

## Examples

See `examples/`:

- **`multi_agent_example.py`** — Pipeline with scoring and bottleneck analysis
- **`react_loop_example.py`** — ReAct loop with `trace_iterations`
- **`parallel_fanout_example.py`** — Async fan-out with aggregation
- **`eval_optimizer_example.py`** — Generator + evaluator refinement loop
- **`state_machine_example.py`** — Router with transitions
- **`retrofit_example.py`** — Adding tracing to existing code

## License

MIT
