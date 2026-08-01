from types import SimpleNamespace

from pydantic import SecretStr
from sqlalchemy import select

from research_lab.commerce import activate_subscription, apply_commerce_event
from research_lab.models import ClientWorkspace, ResearchMandate, WorkspaceEntitlement
from research_lab.schemas import CommerceLifecycleEventRequest, SubscriptionActivationRequest


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
    assert "uspto_odp" in mandate.sources


def test_signed_lifecycle_event_automatically_changes_entitlement(session, monkeypatch):
    monkeypatch.setattr(
        "research_lab.commerce.get_settings",
        lambda: SimpleNamespace(
            portal_token_signing_secret=SecretStr("test-signing-secret-with-at-least-32-chars")
        ),
    )
    activation = activate_subscription(
        session,
        SubscriptionActivationRequest(
            external_session_id="cs_test_lifecycle_12345678",
            external_customer_id="cus_test_12345678",
            external_subscription_id="sub_test_12345678",
            plan="continuous-monitor",
            contact_email="lead@example.com",
            organization="Lifecycle Materials",
            research_question=(
                "Continuously map coating evidence and patent claims for durable industrial membranes."
            ),
        ),
    )
    failed = CommerceLifecycleEventRequest(
        external_event_id="evt_payment_failed_12345678",
        event_type="invoice.payment_failed",
        external_customer_id="cus_test_12345678",
        external_subscription_id="sub_test_12345678",
    )
    first = apply_commerce_event(session, failed)
    repeated = apply_commerce_event(session, failed)
    workspace = session.get(ClientWorkspace, activation["workspace"]["id"])
    entitlement = session.scalar(select(WorkspaceEntitlement))
    assert first["subscription_status"] == "past_due"
    assert repeated["idempotent"] is True
    assert workspace.active is False
    assert entitlement.active is False

    restored = apply_commerce_event(
        session,
        CommerceLifecycleEventRequest(
            external_event_id="evt_invoice_paid_12345678",
            event_type="invoice.paid",
            external_customer_id="cus_test_12345678",
            external_subscription_id="sub_test_12345678",
        ),
    )
    session.refresh(workspace)
    session.refresh(entitlement)
    assert restored["entitlement_active"] is True
    assert workspace.active is True
    assert entitlement.active is True
