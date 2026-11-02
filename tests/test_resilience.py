import asyncio

from agenteval.evaluators.resilience import InjectedFailure, ResilienceEvaluator


def test_resilience_score_passes_when_agent_recovers_from_injected_fault():
    async def agent(fail):
        try:
            fail()
        except InjectedFailure as exc:
            return {"recovered": exc.scenario == "http_500"}

    result = asyncio.run(ResilienceEvaluator().evaluate(agent, "http_500"))
    assert result.recovered is True
    assert result.score == 1.0


def test_resilience_score_fails_when_fault_escapes_agent():
    async def agent(fail):
        fail()

    result = asyncio.run(ResilienceEvaluator().evaluate(agent, "rate_limit"))
    assert result.recovered is False
    assert result.score == 0.0
