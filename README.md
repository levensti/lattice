# agent-trace

Quality debugging framework for multi-agent systems. Find out **which step is the bottleneck** — whether it's a simple chain, a ReAct loop, a parallel fan-out, or a state machine.

agent-trace gives you:

- **A single decorator** (`@step`) to annotate any function — name is inferred, `self` is excluded, zero boilerplate
- **Multiple integration levels** — decorator, context manager (`trace_step`), or runtime wrapping (`instrument`) for code you don't own
- **Topology primitives** — `trace_loop`, `trace_parallel`, `trace_transition`, and `trace_activation` to describe how your agents are structured
- **Automatic LLM-based scoring** — each step is judged against its goal without you writing evaluation prompts
- **Topology-aware bottleneck analysis** — loop convergence failures and parallel branch imbalance detection

## Installation

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from agent_trace import step, trace_session, score_trace, find_bottlenecks

@step(goal="Must return at least 3 factual claims with sources")
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@step(goal="Must have introduction, body, and conclusion")
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

with trace_session() as session:
    notes = researcher("quantum computing")
    article = writer(notes)

score_trace(session, model="gpt-4o")
for b in find_bottlenecks(session):
    print(f"{b.step_name}: {b.score}/5 ({b.impact}) — {b.explanation}")
```

No `configure()` call needed. Pass `model=` to `score_trace` and the provider, API base, and API key are inferred automatically (see [Scoring](#scoring)).

## Three Ways to Trace

### 1. `@step` decorator — for functions you own

```python
@step                                    # bare — name inferred
def think(state): ...

@step(goal="Must cite sources")          # with goal
def research(topic): ...

@step(name="custom", role="generator")   # explicit name and role
def my_func(): ...
```

Works on sync and async functions. Works on methods — `self` and `cls` are automatically excluded from traced inputs.

### 2. `trace_step` context manager — for code you don't own

```python
from agent_trace import trace_step

with trace_step("external_search", goal="Return relevant results") as ts:
    result = third_party_api.search(query)
    ts.set_output(result)
```

### 3. `instrument()` — for runtime wrapping

```python
from agent_trace import instrument

# Wrap without modifying source
traced_search = instrument(search_api, goal="Return results")
results = traced_search(query)

# Or instrument a method on an instance
agent.search = instrument(agent.search, goal="Find documents")
```

## Architecture Support

### Pipeline (sequential chain)

Sequential function calls produce a flat trace — the default behavior.

```python
with trace_session() as session:
    notes = researcher("topic")
    draft = writer(notes)
    final = editor(draft)
```

### ReAct Loop (single agent)

Use `trace_iterations` for a flat loop, or `trace_loop` + `loop.iteration()` when you need more control:

```python
from agent_trace import trace_iterations

with trace_session() as session:
    for _ in trace_iterations("react", range(10)):
        thought = think(state)
        if thought["action"] == "finish":
            break
        result = act(thought)
        state = observe(result)
```

`find_bottlenecks()` detects when loops fail to converge.

### Parallel Fan-out / Fan-in

```python
from agent_trace import trace_parallel

with trace_session() as session:
    with trace_parallel("search_fanout"):
        results = await asyncio.gather(
            search_web(query), search_db(query), search_cache(query),
        )
    summary = aggregate(results)
```

`find_bottlenecks()` detects when one branch scores significantly worse than the others.

### Evaluator-Optimizer Loop

```python
@step(goal="...", role="generator")
def generate(topic, feedback=None): ...

@step(goal="...", role="evaluator")
def evaluate(text): ...

with trace_session() as session:
    for attempt in trace_iterations("refine", range(5)):
        draft = generate(topic, feedback=feedback)
        verdict = evaluate(draft)
        if verdict["pass"]:
            break
        feedback = verdict["feedback"]
```

### State Machine / Graph

Transitions are auto-recorded from the call graph. Use `trace_transition` only when you want to annotate *why* a path was taken:

```python
from agent_trace import trace_transition

@step(goal="Route to the correct next step")
def router(state):
    result = validate(state)
    if result["valid"]:
        return process(result)
    # Only annotate the non-obvious path
    trace_transition(to="handle_error", reason="validation failed")
    return handle_error(result["errors"])
```

### Orchestrator + Subagents

Parent-child relationships are tracked automatically from the call graph:

```python
@step(goal="Coordinate the research workflow", role="orchestrator")
def orchestrator(task):
    code = coder(task)
    review = reviewer(code)
    return synthesize(code, review)
```

### Blackboard / Event-driven

```python
from agent_trace import trace_activation

with trace_activation(reason="knowledge_base updated by agent_a"):
    agent_b.process(blackboard)
```

### Thread-based Parallelism

```python
from agent_trace import copy_trace_context
from concurrent.futures import ThreadPoolExecutor

ctx = copy_trace_context()
with ThreadPoolExecutor() as pool:
    future = pool.submit(ctx.run, my_step, arg1, arg2)
```

## API Reference

### Configuration

`configure()` is **optional**. Tracing works without it. You only need it for OTel export or to set global defaults for the judge model.

```python
configure(
    service_name="my-app",          # OpenTelemetry service name
    otel_endpoint="localhost:4317", # OTLP gRPC endpoint (required to enable OTel)
    judge_model="gpt-4o",           # model for the judge LLM (provider auto-detected)
    judge_api_key="sk-...",         # defaults to env var based on model (see below)
)
```

When you set `judge_model`, the provider and API base are **inferred automatically**:

| Model name | Routes to | Env var |
|---|---|---|
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI | `OPENAI_API_KEY` |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `openai/gpt-4o`, `google/gemini-2.0-flash`, … | OpenRouter | `OPENROUTER_API_KEY` |
| Anything else | OpenRouter | `OPENROUTER_API_KEY` |

You can still override explicitly with `judge_provider=` and `judge_api_base=` if needed.

### The `@step` Decorator

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | no | Step name (defaults to function name) |
| `description` | no | What this step does |
| `goal` | no | Quality goal for the judge |
| `step_id` | no | Custom ID (auto-generated if omitted) |
| `tags` | no | Labels for grouping/filtering |
| `role` | no | Semantic role (e.g. `"generator"`, `"evaluator"`, `"think"`) |

### Topology Primitives

| Primitive | Purpose | Use case |
|-----------|---------|----------|
| `trace_iterations(name, iterable)` | Flat loop tracing (single `for` statement) | ReAct, eval-optimizer |
| `trace_loop(name)` | Loop with manual `.iteration()` control | Advanced loop patterns |
| `trace_parallel(name)` | Mark steps as concurrent | Fan-out/fan-in, MoA |
| `trace_transition(to, reason)` | Annotate a routing decision | State machines, orchestrators |
| `trace_activation(reason)` | Explain why a step fired without a parent | Blackboard, event-driven |
| `copy_trace_context()` | Propagate context to threads | Thread-based parallelism |

### Session Management

```python
with trace_session() as session:
    result = my_step("hello")

print(session.steps)        # list of StepRecord
print(session.groups)       # list of GroupRecord (loops, parallel)
print(session.transitions)  # list of TransitionRecord
```

### Scoring

The simplest way to score is to pass `model=` directly — the provider and API key are resolved automatically:

```python
score_trace(session, model="gpt-4o")              # uses OPENAI_API_KEY
score_trace(session, model="claude-sonnet-4-20250514")  # uses ANTHROPIC_API_KEY
score_trace(session, model="google/gemini-2.0-flash")   # uses OPENROUTER_API_KEY

await async_score_trace(session, model="gpt-4o", max_concurrency=5)
```

You can also pass `api_key=` explicitly, or bring your own provider:

```python
score_trace(session, model="gpt-4o", api_key="sk-...")

from agent_trace.judge.providers import AnthropicJudgeProvider
provider = AnthropicJudgeProvider(api_key="sk-ant-...")
score_trace(session, provider=provider)
```

### Bottleneck Analysis

```python
for b in find_bottlenecks(session):
    print(f"{b.step_name}: score={b.score}, impact={b.impact}")
```

| Impact | Meaning |
|--------|---------|
| `"error"` | Step raised an exception |
| `"lowest_score"` | Worst-scoring step |
| `"largest_drop"` | Biggest quality drop from preceding step |
| `"below_average"` | Below the session average |
| `"loop_no_convergence"` | Scores didn't improve across loop iterations |
| `"weakest_branch"` | Worst parallel branch, significantly below group average |

## OpenTelemetry

agent-trace creates standard OTel spans with these custom attributes:

- `agent_trace.name` — step name
- `agent_trace.input` / `agent_trace.output` — serialized I/O
- `agent_trace.latency_ms` — wall-clock time
- `agent_trace.tags` — comma-separated tags
- `agent_trace.role` — semantic role
- `agent_trace.group_id` — group membership (loop or parallel)
- `agent_trace.iteration` — iteration number

Spans nest automatically. If `opentelemetry-sdk` is not installed, the decorators still work — you just won't get OTel spans.

## Examples

See the `examples/` directory:

- **`multi_agent_example.py`** — Pipeline: research → write → edit
- **`react_loop_example.py`** — ReAct loop with `trace_iterations`
- **`parallel_fanout_example.py`** — Async fan-out with aggregation
- **`eval_optimizer_example.py`** — Generator + evaluator refinement loop
- **`state_machine_example.py`** — Router with auto-transitions
- **`retrofit_example.py`** — Adding tracing to existing code with `instrument()` and `trace_step`

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
