# Lattice

Find the bottleneck in multi-agent system.

Lattice is a lightweight debugging framework for developers with existing agent pipelines. It traces each step, scores it against a defined goal using an LLM judge, and surfaces the steps where quality degrades.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from lattice import step, trace_session, score_trace, find_bottlenecks

@step(goal="Must return at least 3 factual claims with sources")
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@step(goal="Must have introduction, body, and conclusion")
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session(goal="Produce a well-researched article") as session:
    notes = researcher("quantum computing")
    article = writer(notes)

score_trace(session, model="gpt-4o")
for b in find_bottlenecks(session):
    print(f"{b.step_name}: {b.score}/5 ({b.impact}) — {b.explanation}")
```

No `configure()` call needed. Every `@step` and `trace_session` requires a `goal` — this is what the judge evaluates against.

## Tracing

### `@step` — for functions you own

```python
@step(goal="Must cite sources")
def research(topic): ...

@step(goal="Decide the next action", role="think")
def think(state): ...
```

Works on sync and async functions. Works on methods (`self` and `cls` are excluded from traced inputs).

### `trace_step` / `instrument` — for code you don't own

```python
with trace_step("api_call", goal="Return relevant results") as ts:
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
| Retrofitting existing code | `instrument()` + `trace_step`                              | `retrofit_example.py`       |

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

## Bottleneck Analysis

```python
for b in find_bottlenecks(session):
    print(f"{b.step_name}: score={b.score}, impact={b.impact}")
```

| Impact                  | Meaning                                                  |
| ----------------------- | -------------------------------------------------------- |
| `"error"`               | Step raised an exception                                 |
| `"lowest_score"`        | Worst-scoring step in the session                        |
| `"largest_drop"`        | Biggest quality drop from the preceding step             |
| `"below_average"`       | Below the session average                                |
| `"loop_no_convergence"` | Scores didn't improve across loop iterations             |
| `"weakest_branch"`      | Worst parallel branch, significantly below group average |

## Configuration

`configure()` is optional — you only need it for OpenTelemetry export or to set global judge defaults.

```python
configure(
    otel_endpoint="localhost:4317",  # enables OTel span export
    judge_model="gpt-4o",           # global default (provider auto-detected)
)
```

OTel spans include `lattice.name`, `lattice.input`, `lattice.output`, `lattice.latency_ms`, and topology metadata. Spans nest automatically. If `opentelemetry-sdk` is not installed, tracing still works — you just won't get OTel spans.

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
