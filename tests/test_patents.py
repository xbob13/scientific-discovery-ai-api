from types import SimpleNamespace

import httpx
from pydantic import SecretStr

from research_lab.patents import search_patents


async def test_patentsview_results_are_normalized_and_persisted(session, monkeypatch):
    settings = SimpleNamespace(
        patentsview_api_key=SecretStr("api-key"),
        patentsview_search_url="https://patents.example/api/v1/patent/",
        epo_ops_consumer_key=None,
        epo_ops_consumer_secret=None,
    )
    monkeypatch.setattr("research_lab.patents.get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "api-key"
        return httpx.Response(
            200,
            json={
                "patents": [
                    {
                        "patent_id": "1234567",
                        "patent_title": "Antifouling membrane coating",
                        "patent_abstract": "A durable hydrophilic surface treatment.",
                        "patent_date": "2025-01-02",
                        "assignee_organization": ["Example Materials"],
                        "cpc_group_id": ["B01D71/00"],
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await search_patents(
            session,
            "antifouling membrane",
            ["patentsview"],
            client=client,
        )
    assert output["provider_failures"] == {}
    assert output["records"][0]["publication_number"] == "1234567"
    assert output["records"][0]["assignees"] == ["Example Materials"]
    assert output["records"][0]["cpc_codes"] == ["B01D71/00"]
