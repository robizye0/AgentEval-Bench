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
- `ResilienceEvaluator`: reusable failure injection for `http_500`, `rate_limit`, `malformed_json`, and `timeout` scenarios.
- `ToolRegistry`: async registry for sync or async handlers with argument models, delay injection, failure injection, and timeouts.
- `create_app(registry)`: FastAPI mock endpoints (`/health`, `/tools`, `/tools/{name}`).

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

`EvaluationTracer` emits spans through the installed OpenTelemetry API when available and becomes a no-op otherwise. Install the optional telemetry extra and configure an OTLP exporter/provider in your application. Langfuse can be used through its OpenTelemetry integration or by translating the same `record_result` attributes into the Langfuse SDK; no credentials or network exporter are implicitly configured.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```
