import pytest

import research_lab.orchestrator as orchestrator
from research_lab.models import Source
from research_lab.orchestrator import ResearchCycle, run_cycle


@pytest.mark.asyncio
async def test_zero_result_source_is_a_successful_auditable_outcome(session, monkeypatch):
    class EmptyAdapter:
        base_url = "https://example.org"

        def __init__(self, client):
            self.client = client

        async def harvest(self, query, limit):
            return []

    monkeypatch.setitem(orchestrator.ADAPTERS, "empty", EmptyAdapter)
    result = await run_cycle(
        session,
        ResearchCycle(
            question="specific material with no matching records",
            sources=("empty",),
            cycle_window="test-zero-result",
        ),
    )
    assert result["sources"]["empty"] == {"records": 0, "new_versions": 0}
    assert session.query(Source).filter_by(name="empty").one().last_success_at is not None
