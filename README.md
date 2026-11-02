# AgentEval-Bench

Deterministic evaluation utilities for agent tool selection, structured arguments, execution trajectories, and fault handling. Python 3.11+, Pydantic v2. The core library is framework-neutral: Pydantic AI, LangChain/LangGraph, and native OpenAI/Anthropic clients can adapt their traces to `TrajectoryStep` / `ToolCall` without introducing runtime dependencies on those frameworks.

## Install and run

```bash
python -m pip install -e '.[dev]'
agenteval run --config agenteval/templates/golden_dataset.json
```

A suite contains expected tool names, required argument values, optional JSON-schema-like argument definitions, step limits, and recorded calls for replay. Exit code `0` means the average accuracy threshold and every case gate passed; `1` means the gate failed; `2` means the suite could not be loaded.

```json
{
  "accuracy_threshold": 0.95,
  "cases": [{
    "id": "weather",
    "input_prompt": "Weather in Oslo?",
    "expected_tool_sequence": ["weather_lookup"],
    "required_arguments": {"city": "Oslo"},
    "argument_schemas": {"weather_lookup": {
      "type": "object", "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }},
    "max_allowed_steps": 2,
    "observed_tool_calls": [{"name": "weather_lookup", "kwargs": {"city": "Oslo"}}]
  }]
}
```

## Use with a live agent

Convert your framework trace into the stable schema and evaluate asynchronously:

```python
from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep

async def agent_adapter(prompt: str) -> TrajectoryStep:
    trace = await my_agent.run(prompt)  # framework-specific call
    calls = [ToolCall(name=c.name, kwargs=c.arguments, latency_ms=c.latency_ms)
             for c in trace.tool_calls]
    return TrajectoryStep(input_state=prompt, model_response=trace.output,
                          tool_calls=calls, token_count=trace.total_tokens,
                          cost_usd=trace.cost_usd)

runner = EvaluationRunner(agent_adapter)
results = await runner.run([TestCase(
    id="case-1", input_prompt="...", expected_tool_sequence=["lookup"],
    required_arguments={"id": 7}, max_allowed_steps=3,
)])
```

The adapter boundary is intentionally small; preserve provider-specific metadata in your own trace store or `details` payloads rather than coupling the evaluator to a framework. `EvalResult.total_cost_usd` is available for adapter-side cost accounting.

## Evaluators and mock API

- `ToolAccuracyEvaluator`: ordered tool-sequence comparison, strict Pydantic payload validation, extra-key rejection, and required-value checks.
- `TrajectoryEvaluator`: bounded optimal/actual efficiency and detection of identical calls repeated three or more times (configurable).
- `ResilienceEvaluator`: deterministic jitter, 3000ms-configurable latency spikes, partial stream truncation, and injected `http_500`, `rate_limit`, `malformed_json`, `timeout`, and connection-reset scenarios.
- `ToolRegistry`: async registry for sync or async handlers with argument models, delay/jitter/spike injection, timeouts, and partial-stream mock responses.
- `create_app(registry)`: FastAPI mock endpoints (`/health`, `/tools`, `/tools/{name}`, `/tools/{name}/stream`).

Example registry:

```python
from agenteval.mock.tool_registry import ToolRegistry
from agenteval.mock.api_server import create_app

registry = ToolRegistry()
registry.register("web_search", lambda query, limit=5: {"query": query, "hits": []})
app = create_app(registry)  # serve with: uvicorn module:app
```

Run the mock service with `docker compose up --build`; it listens on port 8000. Register your tool handlers in an application module before exposing `create_app(registry)`; the default service has no tools registered by design.

## Pytest gate

The pytest plugin is registered on installation:

```bash
pytest --agent-eval-gate --accuracy-threshold=0.95 --agent-eval-config=eval_suite.yaml
```

The gate replays `observed_tool_calls` from the configured suite. Use the Python adapter API for evaluations against a running agent.

## Semantic intent grading

Deterministic exact scoring remains available as `deterministic_accuracy_score` (also mirrored to the legacy `tool_accuracy_score`). Optionally enable the LLM judge for semantically equivalent query text or reordered-but-progressing trajectories:

```bash
set OPENAI_API_KEY=...
agenteval run --config eval_suite.yaml --semantic-judge --judge-provider openai --judge-model gpt-4o-mini
# Or use Anthropic with ANTHROPIC_API_KEY and --judge-provider anthropic
```

The evaluator returns both deterministic and semantic scores. Schema violations, mismatches in unmarked required arguments, and unapproved tool substitutions still fail closed. Only mark natural-language fields such as `query` under `semantic_arguments` for fuzzy grading; keep IDs and other exact values unmarked. Configure `allowed_additional_tools` only when extra progress steps are valid. The judge sends expected/observed tool-call data and arguments to the selected API, so enable this only when that data-sharing is acceptable. API usage may incur provider costs.

```yaml
semantic_arguments: [query]
required_arguments:
  query: best coffee in London
```

## Budgets and network chaos

Use per-case gates in the suite:

```yaml
max_token_budget: 12000
max_cost_usd: 0.05
```

The agent adapter reports actual usage through `TrajectoryStep(token_count=..., cost_usd=..., latency_ms=...)`; any budget breach fails the case even if tool accuracy is perfect. Recorded CLI fixtures can populate `observed_token_count`, `observed_cost_usd`, and `observed_latency_ms`.

`ResilienceEvaluator.calculate_delay_ms()` produces seeded reproducible jitter and long-tail spikes; `inject_latency()` applies the delay asynchronously. The mock registry accepts `jitter_ms`, `spike_probability`, `latency_spike_ms`, and `random_seed`. `stream_with_chaos()` / `POST /tools/{name}/stream` can truncate a response after a configured number of chunks to simulate a partial streaming disconnect. `connection_reset` is also an injectable failure scenario.

## Native MCP validation

Install the official Python MCP client SDK as an optional extra:

```bash
python -m pip install -e '.[mcp]'
```

`MCPEvaluator` uses the official `mcp` client session over stdio or legacy SSE, initializes the JSON-RPC session, lists tools, compares their published JSON schemas against expected schemas, and can call tools:

```python
import asyncio
from agenteval.evaluators.mcp import MCPEvaluator, MCPCall

result = await MCPEvaluator().validate_stdio(
    command="python", args=["my_mcp_server.py"],
    expected_schemas={"search": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    calls=[MCPCall(name="search", arguments={"query": "coffee shops"})],
)
assert result.passed, result.discrepancies

# Legacy MCP SSE transport:
# await MCPEvaluator().validate_sse("http://localhost:8000/sse", expected_schemas={...})
```

## Rich terminal and HTML reports

The CLI shows case progress, a Rich results table, and a colored failure-diff tree. Produce an offline, standalone dashboard (inline CSS/JS; no CDN) with:

```bash
agenteval run --config eval_suite.yaml --html-report=summary.html
```

The report includes interactive per-case details and accuracy, latency, and token-usage charts.

## GitHub Actions

```yaml
name: agent-evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: python -m pip install -e '.[dev]'
      - run: agenteval run --config agenteval/templates/golden_dataset.json
```

## Tracing

`EvaluationTracer` records one span per case and one per tool call, with accuracy, semantic score, latency, token count, cost, and tool latency attributes. With no exporter configured it remains a no-op (or uses a provider configured by the embedding application). To export from the CLI through OTLP/HTTP:

```bash
python -m pip install -e '.[telemetry]'
# Set OTEL_EXPORTER_OTLP_ENDPOINT to your collector, e.g. http://localhost:4318
agenteval run --config eval_suite.yaml
```

The CLI detects the standard `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, creates an OTLP provider only when no provider already exists, and flushes queued spans before exit. Langfuse can receive OTLP traces through its OpenTelemetry endpoint and configured headers; credentials are never defaulted or logged.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```
