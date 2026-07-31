from types import SimpleNamespace

import httpx
import pytest

from research_lab.models import OutreachMessage
from research_lab.outreach import (
    approve_outreach,
    create_prospect,
    deliver_outreach,
    discover_and_draft_prospects,
    draft_outreach,
)
from research_lab.schemas import OutreachDraftCreate, ProspectCreate


def prospect_request() -> ProspectCreate:
    return ProspectCreate(
        name="Example Filtration",
        domain="example.com",
        evidence_signals=[
            {
                "summary": "The company publicly requested advanced membrane materials.",
                "source_url": "https://example.com/open-innovation",
                "weight": 35,
            }
        ],
        contact_name="Innovation Team",
        contact_email="innovation@example.com",
        contact_basis="Public innovation-program contact",
    )


async def test_outreach_requires_human_approval_and_enabled_delivery(session, monkeypatch):
    prospect = create_prospect(session, prospect_request())
    message = draft_outreach(
        session,
        OutreachDraftCreate(
            prospect_id=prospect.id,
            program_slug="advanced-separations",
            value_proposition=(
                "A tailored monitor could surface evidence-backed candidates and patent context "
                "against your published filtration priorities."
            ),
        ),
    )
    disabled = SimpleNamespace(
        outreach_send_enabled=False,
        outreach_delivery_webhook_url=None,
        outreach_delivery_token=None,
    )
    monkeypatch.setattr("research_lab.outreach.get_settings", lambda: disabled)
    with pytest.raises(PermissionError):
        await deliver_outreach(session, message)

    approve_outreach(session, message, "Lab Director")
    enabled = SimpleNamespace(
        outreach_send_enabled=True,
        outreach_delivery_webhook_url="https://mailer.example/send",
        outreach_delivery_token=None,
    )
    monkeypatch.setattr("research_lab.outreach.get_settings", lambda: enabled)

    def handler(request: httpx.Request) -> httpx.Response:
        assert b"innovation@example.com" in request.content
        return httpx.Response(202, json={"id": "provider-123"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sent = await deliver_outreach(session, message, client=client)
    assert sent.status == "sent"
    assert sent.provider_message_id == "provider-123"
    assert sent.approved_by == "Lab Director"


async def test_discovery_feed_creates_drafts_but_never_approves_them(session, monkeypatch):
    settings = SimpleNamespace(
        prospect_discovery_enabled=True,
        prospect_discovery_feed_url="https://discovery.example/prospects",
        prospect_discovery_token=None,
    )
    monkeypatch.setattr("research_lab.outreach.get_settings", lambda: settings)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "prospects": [
                    {
                        **prospect_request().model_dump(mode="json"),
                        "program_slug": "advanced-separations",
                        "value_proposition": (
                            "A governed evidence monitor can map membrane and patent signals "
                            "to the priorities in the public innovation request."
                        ),
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_and_draft_prospects(session, client)
    assert len(result["created"]) == 1
    assert len(result["drafted"]) == 1
    assert session.get(OutreachMessage, result["drafted"][0]).status == "draft"
