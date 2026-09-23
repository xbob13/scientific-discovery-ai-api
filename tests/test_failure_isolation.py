import pytest

from research_lab import orchestrator
from research_lab.orchestrator import ResearchCycle, run_cycle


@pytest.mark.asyncio
async def test_cycle_does_not_use_stale_corpus_when_all_sources_fail(monkeypatch, session):
    class FailingAdapter:
        base_url = "https://example.invalid"

        def __init__(self, client):
            self.client = client

        async def harvest(self, query, limit):
            raise RuntimeError("provider unavailable")

    monkeypatch.setitem(orchestrator.ADAPTERS, "openalex", FailingAdapter)
    result = await run_cycle(session, ResearchCycle(question="test", sources=("openalex",)))

    assert result["candidates"] == []
    assert result["connections_created"] == 0
    assert result["client_briefs_published"] == 0
    assert result["sources"]["openalex"]["error"] == "RuntimeError"
