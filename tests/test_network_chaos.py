import asyncio

import pytest

from agenteval.evaluators.resilience import ResilienceEvaluator


def test_latency_chaos_is_seeded_and_includes_configured_spike():
    chaos = ResilienceEvaluator()
    params = {
        "base_latency_ms": 50,
        "jitter_ms": 10,
        "spike_probability": 1.0,
        "spike_latency_ms": 3000,
        "seed": 7,
    }
    first = chaos.calculate_delay_ms(**params)
    assert first == chaos.calculate_delay_ms(**params)
    assert first >= 3000


def test_stream_chaos_drops_after_partial_chunks():
    async def scenario():
        seen = []
        with pytest.raises(ConnectionResetError):
            async for chunk in ResilienceEvaluator.stream_with_chaos(
                [b"a", b"b", b"c"], drop_after_chunks=2
            ):
                seen.append(chunk)
        return seen

    assert asyncio.run(scenario()) == [b"a", b"b"]


def test_unsupported_chaos_scenario_is_rejected():
    with pytest.raises(ValueError):
        ResilienceEvaluator.validate_scenario("random_corruption")
