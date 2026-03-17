# Lattice

Lattice is a quality debugging framework for multi-agent systems. Annotate the actions in your workflow, score them with LLM-as-a-judge, and find exactly where quality degrades — without blocking your critical path.

Designed for progressive adoption: start with tracing (zero config), add scoring when you're ready, and scale to production with background scoring. Works with any agent architecture — sequential pipelines, ReAct loops, parallel fan-outs, evaluator-optimizer patterns, state machines, and orchestrator-subagent hierarchies.

## Install

```bash
pip install -e .
```

## Quick Start

### 1. Trace your workflow

Annotate functions with `@action` and wrap the workflow in `trace_session`. No API keys needed — traces are saved to a local SQLite database automatically.

```python
from lattice import action, trace_session, print_trace_summary

@action
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@action
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

print_trace_summary(session)
```

This records inputs, outputs, latency, and parent-child relationships for every action. Browse results with `python -m lattice dashboard`.

### 2. Add scoring

When you're ready to evaluate quality, add a `goal` to each action and score with any LLM provider:

```python
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
score_trace(session, provider=OpenAIProvider("gpt-4o"))

# Find structural issues
for b in session.bottlenecks:
    print(f"{b.action_name}: {b.score:.1f} ({b.impact}) — {b.explanation}")
```

### 3. Score inline (non-blocking)

For tighter feedback loops, attach a `JudgeConfig` directly to actions. Scoring happens in a background thread — the caller is never blocked:

```python
from lattice import action, trace_session, JudgeConfig, AnthropicProvider

judge = JudgeConfig(
    system_prompt="Score 1-5. Respond: {\"score\": N, \"explanation\": \"...\"}",
    provider=AnthropicProvider("claude-sonnet-4-20250514"),
)

@action(goal="Return factual claims with sources", judge=judge)
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")  # returns immediately, score computed in background
```

## Tracing

### `@action` — for functions you own

```python
@action(goal="Must cite sources")
def research(topic): ...

@action(goal="Decide the next action", role="think")
def think(state): ...

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

| Pattern                    | How                                                        | Example                     |
| -------------------------- | ---------------------------------------------------------- | --------------------------- |
| State machine              | `trace_transition(to=..., reason=...)`                     | `state_machine_example.py`  |
| Orchestrator + subagents   | Automatic from the call graph                              | `multi_agent_example.py`    |
| Evaluator-optimizer        | `trace_iterations` with `role="generator"` / `"evaluator"` | `eval_optimizer_example.py` |
| Blackboard / event-driven  | `trace_activation(reason=...)`                             | —                           |
| Thread-based parallelism   | `copy_trace_context()`                                     | —                           |
| Retrofitting existing code | `instrument()` + `trace_action`                            | `retrofit_example.py`       |

## Scoring

Lattice is unopinionated about scoring — your rubric defines the scale, criteria, and format. There are three ways to score, depending on your needs:

### Inline scoring (per-action, non-blocking)

Attach a `JudgeConfig` to any `@action` or `trace_action`. The action returns immediately; scoring runs in a daemon thread:

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
        provider=AnthropicProvider("claude-opus-4-6"),
    ),
)
def summarise(paper): ...
```

When `judge=` is set, its `system_prompt` and `provider` override the global judge for that action only.

### Session-level scoring (end-to-end, non-blocking)

Score the entire workflow's output against its goal by passing `judge=` to `trace_session`. The session is persisted immediately, then re-persisted with the score once scoring completes in the background:

```python
provider = OpenAIProvider("gpt-4o")

with trace_session(goal="Produce a publication-ready article", judge=provider) as session:
    notes = researcher("quantum computing")
    article = writer(notes)
# session.session_score is populated shortly after exit
```

Or score explicitly after the fact:

```python
from lattice import score_session

score, explanation = score_session(session, provider=OpenAIProvider("gpt-4o"))
```

### Batch scoring (after the fact)

Score every action in a completed session — useful for offline analysis or re-scoring with a different rubric:

```python
from lattice import score_trace

score_trace(session, provider=OpenAIProvider("gpt-4o"))

# With a custom global rubric
score_trace(
    session,
    provider=OpenAIProvider("gpt-4o"),
    system_prompt="You are a strict technical evaluator. ...",
)
```

Use `async_score_trace` for concurrent scoring across actions.

### Background scoring (production)

For production workloads, use `BackgroundScorer` to score off the critical path entirely. It scores all actions and the session end-to-end, then persists results:

```python
from lattice import BackgroundScorer, OpenAIProvider

scorer = BackgroundScorer(provider=OpenAIProvider("gpt-4o"))
await scorer.start()

# Per-request hot path — submit() is non-blocking
async def handle_request(query):
    with trace_session(goal="...") as session:
        result = await run_agent(query)
    scorer.submit(session)   # returns immediately
    return result

# At shutdown — does not block
await scorer.cancel()
```

Or as an async context manager:

```python
async with BackgroundScorer(provider=OpenAIProvider("gpt-4o")) as scorer:
    await serve_forever(scorer)
```

### Providers

| Class                | Wire protocol                 | Typical env var              |
| -------------------- | ----------------------------- | -----------------------------|
| `OpenAIProvider`     | Chat Completions or Responses | `OPENAI_API_KEY`             |
| `AnthropicProvider`  | Messages                      | `ANTHROPIC_API_KEY`          |
| `OpenRouterProvider` | Chat Completions              | `OPENROUTER_API_KEY`         |

```python
import os
from lattice import OpenAIProvider, AnthropicProvider, OpenRouterProvider

OpenAIProvider("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
AnthropicProvider("claude-sonnet-4-20250514", api_key=os.environ["ANTHROPIC_API_KEY"])
OpenRouterProvider("google/gemini-2.0-flash", api_key=os.environ["OPENROUTER_API_KEY"])
```

`OpenAIProvider` supports custom endpoints (Fireworks, Together, etc.) via `api_base=` and the newer Responses API via `api_type=ApiType.RESPONSES`.

### Bring your own provider

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
python -m lattice dashboard              # starts at http://localhost:8080
python -m lattice dashboard --port 3000  # custom port
```

### Demo data

Seed the store with rich example traces and explore the dashboard:

```bash
python scripts/seed_sqllite_traces.py
python -m lattice dashboard
```

Then visit [http://localhost:8080](http://localhost:8080) to explore multi-agent workflows, RAG pipelines, and more.

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

See `examples/`:

- **`multi_agent_example.py`** — Pipeline with scoring and bottleneck analysis
- **`react_loop_example.py`** — ReAct loop with `trace_iterations`
- **`parallel_fanout_example.py`** — Async fan-out with aggregation
- **`eval_optimizer_example.py`** — Generator + evaluator refinement loop
- **`state_machine_example.py`** — Router with transitions
- **`retrofit_example.py`** — Adding tracing to existing code without modifying source

## License

MIT
