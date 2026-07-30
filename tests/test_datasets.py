import httpx
import pytest

from research_lab.datasets import discover_optimade_providers, seed_dataset_registry
from research_lab.models import DatasetDefinition


def test_core_dataset_registry_is_idempotent(session):
    assert seed_dataset_registry(session) >= 6
    first_count = session.query(DatasetDefinition).count()
    seed_dataset_registry(session)
    assert session.query(DatasetDefinition).count() == first_count


@pytest.mark.asyncio
async def test_optimade_provider_discovery_tracks_access_boundaries(session):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://providers.optimade.org/v1/links"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "example",
                        "attributes": {
                            "name": "Example Materials",
                            "base_url": "https://example.org/optimade",
                            "homepage": "https://example.org",
                        },
                    },
                    {
                        "id": "unavailable",
                        "attributes": {"name": "Unavailable", "base_url": None},
                    },
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await discover_optimade_providers(session, client) == 1
    item = session.query(DatasetDefinition).filter_by(code="optimade-example").one()
    assert item.access_tier == "provider_specific"
    assert item.redistribution_policy == "metadata_only_until_reviewed"
