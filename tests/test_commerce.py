from types import SimpleNamespace

from pydantic import SecretStr
from sqlalchemy import select

from research_lab.commerce import activate_subscription
from research_lab.models import ResearchMandate, WorkspaceEntitlement
from research_lab.schemas import SubscriptionActivationRequest


def test_verified_activation_is_idempotent_and_provisions_a_mandate(session, monkeypatch):
    monkeypatch.setattr(
        "research_lab.commerce.get_settings",
        lambda: SimpleNamespace(
            portal_token_signing_secret=SecretStr("test-signing-secret-with-at-least-32-chars")
        ),
    )
    request = SubscriptionActivationRequest(
        external_session_id="cs_test_12345678",
        plan="continuous-monitor",
        contact_email="lead@example.com",
        organization="Example Materials",
        program_slug="advanced-separations",
        research_question=(
            "Identify durable membrane surface treatments that reduce irreversible fouling "
            "without sacrificing permeability."
        ),
    )
    first = activate_subscription(session, request)
    second = activate_subscription(session, request)

    assert first["workspace"]["slug"] == second["workspace"]["slug"]
    assert first["portal_token"] == second["portal_token"]
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    entitlement = session.scalar(select(WorkspaceEntitlement))
    mandate = session.scalar(select(ResearchMandate))
    assert entitlement.plan == "continuous-monitor"
    assert entitlement.monthly_request_limit == 30
    assert mandate.status == "queued"
    assert "patentsview" in mandate.sources
