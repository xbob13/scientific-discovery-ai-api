from types import SimpleNamespace

import httpx
from pydantic import SecretStr

from research_lab.patents import search_patents


async def test_uspto_odp_results_are_normalized_and_persisted(session, monkeypatch):
    settings = SimpleNamespace(
        uspto_odp_api_key=SecretStr("api-key"),
        uspto_odp_search_url="https://api.uspto.example/api/v1/patent/applications/search",
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
                "patentFileWrapperDataBag": [
                    {
                        "applicationNumberText": "18123456",
                        "applicationMetaData": {
                            "earliestPublicationNumber": "US20250001234A1",
                            "inventionTitle": "Antifouling membrane coating",
                            "abstractText": "A durable hydrophilic surface treatment.",
                            "publicationDate": "2025-01-02",
                            "applicantBag": [{"applicantNameText": "Example Materials"}],
                            "cpcClassificationBag": [{"classificationSymbolText": "B01D71/00"}],
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await search_patents(
            session,
            "antifouling membrane",
            ["uspto_odp"],
            client=client,
        )
    assert output["provider_failures"] == {}
    assert output["records"][0]["publication_number"] == "US20250001234A1"
    assert output["records"][0]["assignees"] == ["Example Materials"]
    assert output["records"][0]["cpc_codes"] == ["B01D71/00"]
    assert output["landscape"]["evidence_passages"][0]["locator"] == "abstract"


async def test_epo_search_uses_ops_range_and_retrieves_claims(session, monkeypatch):
    settings = SimpleNamespace(
        uspto_odp_api_key=None,
        patentsview_api_key=None,
        epo_ops_consumer_key=SecretStr("consumer"),
        epo_ops_consumer_secret=SecretStr("secret"),
        epo_ops_auth_url="https://ops.example/auth/accesstoken",
        epo_ops_search_url="https://ops.example/rest-services/published-data/search",
        epo_ops_published_data_url="https://ops.example/rest-services/published-data",
        epo_ops_claims_per_search=1,
    )
    monkeypatch.setattr("research_lab.patents.get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("accesstoken"):
            return httpx.Response(200, json={"access_token": "token"})
        if request.url.path.endswith("search"):
            assert request.headers["x-ops-range"] == "1-10"
            return httpx.Response(
                200,
                text=(
                    '<ops:world-patent-data xmlns:ops="http://ops.epo.org">'
                    '<exchange-document country="EP" doc-number="1234567" kind="A1">'
                    "<invention-title>Durable membrane coating</invention-title>"
                    "</exchange-document></ops:world-patent-data>"
                ),
            )
        assert request.url.path.endswith("/publication/epodoc/EP1234567A1/claims")
        return httpx.Response(
            200,
            text=(
                "<claims><claim><claim-text>A membrane comprising a crosslinked coating."
                "</claim-text></claim></claims>"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await search_patents(session, "crosslinked membrane coating", ["epo_ops"], client=client)
    assert output["records"][0]["claims_excerpt"].startswith("A membrane comprising")
    assert output["landscape"]["claim_coverage"]["records_with_claims"] == 1
