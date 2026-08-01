from types import SimpleNamespace

import httpx

from research_lab.institutions import discover_research_institutions
from research_lab.models import ResearchInstitution


async def test_openalex_institution_discovery_persists_ranked_network(session, monkeypatch):
    settings = SimpleNamespace(
        openalex_base_url="https://openalex.example",
        openalex_api_key=None,
    )
    monkeypatch.setattr("research_lab.institutions.get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/works":
            assert request.url.params["group_by"] == "authorships.institutions.id"
            return httpx.Response(
                200,
                json={
                    "group_by": [
                        {
                            "key": "https://openalex.org/I123",
                            "key_display_name": "Example Materials University",
                            "count": 184,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "https://openalex.org/I123",
                "display_name": "Example Materials University",
                "country_code": "US",
                "type": "education",
                "homepage_url": "https://materials.example.edu",
                "ids": {"ror": "https://ror.org/12345"},
                "summary_stats": {"2yr_mean_citedness": 5.2},
                "topics": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_research_institutions(
            session,
            "advanced membrane materials",
            limit=10,
            client=client,
            seed_verified_channels=False,
        )
    assert result["failures"] == {}
    institution = session.get(ResearchInstitution, result["discovered"][0])
    assert institution.name == "Example Materials University"
    assert institution.works_count == 184
    assert institution.ror_id == "https://ror.org/12345"
