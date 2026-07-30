import httpx
import pytest

from research_lab.adapters.datacite import DataCiteAdapter


@pytest.mark.asyncio
async def test_datacite_adapter_harvests_cited_materials_datasets():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["resource-type-id"] == "dataset"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "10.1234/example",
                        "attributes": {
                            "doi": "10.1234/example",
                            "titles": [{"title": "Membrane permeability measurements"}],
                            "descriptions": [
                                {
                                    "description": (
                                        "Measured water flux and fouling resistance for polymer membranes."
                                    )
                                }
                            ],
                            "creators": [{"name": "Example Research Group"}],
                            "published": "2026-01-02",
                            "url": "https://doi.org/10.1234/example",
                            "rightsList": [
                                {"rightsUri": "https://creativecommons.org/licenses/by/4.0/"}
                            ],
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await DataCiteAdapter(client).harvest("membrane fouling", 5)
    assert len(records) == 1
    assert records[0].doi == "10.1234/example"
    assert "water flux" in records[0].abstract
    assert not records[0].peer_reviewed


def test_datacite_adapter_tolerates_non_object_optional_metadata():
    record = object.__new__(DataCiteAdapter)._parse(
        {
            "id": "10.1234/odd",
            "attributes": {
                "doi": "10.1234/odd",
                "titles": ["unexpected", {"title": "Usable title"}],
                "creators": ["unexpected"],
                "descriptions": ["unexpected"],
                "rightsList": ["unexpected"],
                "url": "https://doi.org/10.1234/odd",
            },
        }
    )
    assert record.title == "Usable title"
    assert record.authors == []
    assert record.license is None
