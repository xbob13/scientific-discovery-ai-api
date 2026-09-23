def test_cycle_does_not_use_stale_corpus_when_all_sources_fail(monkeypatch, session):
    from research_lab.orchestrator import ResearchCycle, run_cycle

    class FailingAdapter:
        base_url = "https://example.invalid"

        async def harvest(self, query, limit):
            raise RuntimeError("provider unavailable")

    monkeypatch.setitem(__import__("research_lab.orchestrator", fromlist=["ADAPTERS"]).ADAPTERS, "openalex", FailingAdapter)
    result = await run_cycle(session, ResearchCycle(question="test", sources=("openalex",)))

    assert result["candidates"] == []
    assert result["connections_created"] == 0
    assert result["client_briefs_published"] == 0
    assert result["sources"]["openalex"]["error"] == "RuntimeError"
