import httpx
import pytest

import scripts.sync_dataset_registry as sync_registry


@pytest.mark.asyncio
async def test_optimade_outage_is_reported_without_blocking_registry(monkeypatch):
    async def fail_discovery(session, client):
        raise httpx.ConnectError("OPTIMADE unavailable")

    monkeypatch.setattr(sync_registry, "discover_optimade_providers", fail_discovery)

    result = await sync_registry.sync()

    assert result["core_datasets"] >= 6
    assert result["optimade_providers"] == 0
    assert result["optimade_error"].startswith("ConnectError")
