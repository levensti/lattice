# agent-trace

Quality debugging framework for multi-agent systems. Find out **which agent step is the bottleneck** in your pipeline.

agent-trace gives you:

- **Decorators** (`@trace_agent`, `@trace_tool`) to annotate every step with a name, description, and quality criteria
- **OpenTelemetry integration** — inputs, outputs, and timing are exported as spans
- **Automatic LLM-based scoring** — each step is judged against its criteria without you writing evaluation prompts
- **Bottleneck analysis** — steps are ranked by quality so you can pinpoint the weakest link

## Installation

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from agent_trace import configure, trace_agent, trace_session, score_trace, find_bottlenecks

configure(judge_model="gpt-4o")  # uses OPENAI_API_KEY env var

@trace_agent(
    name="researcher",
    description="Researches a topic using web search",
    criteria="Must return at least 3 factual claims with sources",
)
def researcher(topic: str) -> str:
    return call_llm(f"Research {topic}")

@trace_agent(
    name="writer",
    description="Writes an article from research notes",
    criteria="Must have introduction, body, and conclusion",
)
def writer(notes: str) -> str:
    return call_llm(f"Write article from: {notes}")

# Run the pipeline inside a trace session
with trace_session() as session:
    notes = researcher("quantum computing")
    article = writer(notes)

# Score each step with an LLM judge
score_trace(session)

# Find the weakest step
for b in find_bottlenecks(session):
    print(f"{b.step_name}: {b.score}/5 ({b.impact}) — {b.explanation}")
```

## API Reference

### Configuration

```python
configure(
    service_name="my-app",          # OpenTelemetry service name
    otel_endpoint="localhost:4317", # OTLP gRPC endpoint (None = console)
    otel_enabled=True,              # set False to disable tracing
    judge_provider="openai",        # "openai" or "anthropic"
    judge_model="gpt-4o",           # model for the judge LLM
    judge_api_key="sk-...",         # defaults to OPENAI_API_KEY env var
    judge_api_base="https://api.openai.com/v1",  # for OpenAI-compatible APIs
    judge_max_concurrency=5,        # max parallel judge calls (async)
)
```

### Decorators

```python
@trace_agent(name="planner", description="...", criteria="...")
def my_agent(input: str) -> str: ...

@trace_tool(name="search", description="...", criteria="...")
def my_tool(query: str) -> list: ...
```

Both decorators support sync and async functions. Parameters:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | yes | Human-readable step name |
| `description` | no | What this step does |
| `criteria` | no | Quality criteria for the judge |
| `step_id` | no | Custom ID (auto-generated if omitted) |

### Session Management

```python
with trace_session() as session:
    # all decorated calls here are recorded into `session`
    result = my_agent("hello")

print(session.steps)  # list of StepRecord
```

### Scoring

```python
# Synchronous
score_trace(session)

# Async (with concurrency control)
await async_score_trace(session, max_concurrency=5)

# Bring your own provider
from agent_trace.judge.providers import AnthropicJudgeProvider
provider = AnthropicJudgeProvider(api_key="sk-ant-...")
score_trace(session, provider=provider)
```

### Bottleneck Analysis

```python
bottlenecks = find_bottlenecks(session)
for b in bottlenecks:
    print(f"{b.step_name}: score={b.score}, impact={b.impact}")
    # impact is one of: "error", "lowest_score", "largest_drop", "below_average"
```

## How It Works

1. **Trace**: Decorators capture inputs, outputs, timing, and parent-child relationships for each step, emitting OpenTelemetry spans.

2. **Score**: `score_trace()` sends each step's input, output, and criteria to an LLM judge that rates quality 1–5. The judge prompt is auto-generated from your `criteria` string — you don't need to write evaluation prompts.

3. **Analyze**: `find_bottlenecks()` ranks steps by score (ascending), flags errors, detects the largest quality drops between steps, and breaks ties by latency.

## OpenTelemetry

agent-trace creates standard OTel spans with these custom attributes:

- `agent_trace.name` — step name
- `agent_trace.type` — `"agent"` or `"tool"`
- `agent_trace.input` — serialized input
- `agent_trace.output` — serialized output
- `agent_trace.latency_ms` — wall-clock time in milliseconds

Spans nest automatically: if agent A calls tool B, the tool span is a child of the agent span.

If `opentelemetry-sdk` is not installed, the decorators still work — you just won't get OTel spans.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
