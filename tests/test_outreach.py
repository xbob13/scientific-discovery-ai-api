from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from research_lab.models import OutreachMessage
from research_lab.outreach import (
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
        source_url="https://example.com/open-innovation",
    )


def enabled_settings(**overrides):
    values = {
        "outreach_autonomous_enabled": True,
        "outreach_send_enabled": True,
        "outreach_delivery_webhook_url": "https://mailer.example/send",
        "outreach_delivery_token": None,
        "outreach_min_relevance_score": 25,
        "outreach_daily_send_limit": 20,
        "outreach_sender_name": "Patterson Research Labs",
        "outreach_reply_to": "research@example.com",
        "outreach_postal_address": "100 Research Way, Phoenix, AZ 85001",
        "outreach_unsubscribe_base_url": "https://api.example.com",
        "outreach_unsubscribe_secret": SecretStr("unsubscribe-test-secret-with-32-characters"),
        "prospect_discovery_enabled": True,
        "prospect_discovery_feed_url": "https://discovery.example/prospects",
        "prospect_discovery_token": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def test_outreach_is_policy_queued_and_delivered_without_human_approval(session, monkeypatch):
    monkeypatch.setattr("research_lab.outreach.get_settings", lambda: enabled_settings())
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
    assert message.status == "queued"
    assert message.approved_by is None

    def handler(request: httpx.Request) -> httpx.Response:
        assert b"innovation@example.com" in request.content
        assert b"Opt out: https://api.example.com/v1/outreach/unsubscribe" in request.content
        return httpx.Response(202, json={"id": "provider-123"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sent = await deliver_outreach(session, message, client=client)
    assert sent.status == "sent"
    assert sent.provider_message_id == "provider-123"
    assert sent.approved_by is None


async def test_discovery_feed_policy_queues_eligible_drafts(session, monkeypatch):
    settings = enabled_settings()
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
    assert session.get(OutreachMessage, result["drafted"][0]).status == "queued"


async def test_sender_configuration_fails_closed_without_human_queue(session, monkeypatch):
    monkeypatch.setattr(
        "research_lab.outreach.get_settings",
        lambda: enabled_settings(outreach_unsubscribe_secret=None),
    )
    prospect = create_prospect(session, prospect_request())
    message = draft_outreach(
        session,
        OutreachDraftCreate(
            prospect_id=prospect.id,
            program_slug="advanced-separations",
            value_proposition="A tailored monitor maps public evidence to the stated priorities.",
        ),
    )
    assert message.status == "held"
    with pytest.raises(PermissionError):
        await deliver_outreach(session, message)
