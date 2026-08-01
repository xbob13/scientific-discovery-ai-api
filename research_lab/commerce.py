from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .clients import hash_portal_token
from .config import get_settings
from .models import (
    ClientTopic,
    ClientWorkspace,
    CommerceEvent,
    ResearchMandate,
    SubscriptionActivation,
    WorkspaceEntitlement,
)
from .schemas import CommerceLifecycleEventRequest, SubscriptionActivationRequest


@dataclass(frozen=True)
class PlanPolicy:
    monthly_request_limit: int
    allowed_sources: tuple[str, ...]
    export_formats: tuple[str, ...] = ("json", "markdown")


PLAN_POLICIES = {
    "commissioned-brief": PlanPolicy(
        monthly_request_limit=3,
        allowed_sources=("openalex", "crossref", "datacite", "uspto_odp"),
    ),
    "continuous-monitor": PlanPolicy(
        monthly_request_limit=30,
        allowed_sources=("openalex", "crossref", "datacite", "uspto_odp", "epo_ops"),
    ),
    "enterprise": PlanPolicy(
        monthly_request_limit=250,
        allowed_sources=("openalex", "crossref", "datacite", "uspto_odp", "epo_ops"),
        export_formats=("json", "markdown", "csv"),
    ),
}


def _portal_token(external_session_id: str) -> str:
    raw_secret = get_settings().portal_token_signing_secret.get_secret_value()
    if raw_secret == "development-portal-signing-secret" or len(raw_secret) < 32:
        raise RuntimeError("secure portal token signing is not configured")
    secret = raw_secret.encode()
    digest = hmac.new(secret, external_session_id.encode(), hashlib.sha256).hexdigest()
    return f"prl_{digest}"


def _workspace_slug(organization: str, external_session_id: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", organization.lower()).strip("-")[:90] or "client"
    suffix = hashlib.sha256(external_session_id.encode()).hexdigest()[:10]
    return f"{stem}-{suffix}"


def activate_subscription(session: Session, request: SubscriptionActivationRequest) -> dict:
    """Create an isolated workspace only after a trusted commerce layer verifies payment."""
    policy = PLAN_POLICIES.get(request.plan)
    if policy is None:
        raise ValueError(f"unsupported plan: {request.plan}")

    existing = session.scalar(
        select(SubscriptionActivation).where(
            SubscriptionActivation.provider == request.provider,
            SubscriptionActivation.external_session_id == request.external_session_id,
        )
    )
    token = _portal_token(f"{request.provider}:{request.external_session_id}")
    if existing:
        workspace = session.get(ClientWorkspace, existing.workspace_id)
        if workspace is None:
            raise RuntimeError("subscription activation references a missing workspace")
        if not hmac.compare_digest(workspace.portal_token_sha256, hash_portal_token(token)):
            raise RuntimeError("portal signing key changed; administrator recovery is required")
        existing.external_customer_id = request.external_customer_id or existing.external_customer_id
        existing.external_subscription_id = (
            request.external_subscription_id or existing.external_subscription_id
        )
        existing.status = request.subscription_status or existing.status
        existing.current_period_end = request.current_period_end or existing.current_period_end
        entitlement = session.scalar(
            select(WorkspaceEntitlement).where(WorkspaceEntitlement.workspace_id == workspace.id)
        )
        is_active = _is_active_status(existing.status)
        workspace.active = is_active
        if entitlement:
            entitlement.active = is_active
        session.commit()
        return {
            "workspace": {"id": workspace.id, "name": workspace.name, "slug": workspace.slug},
            "portal_token": token,
            "plan": existing.plan,
            "subscription_status": existing.status,
            "idempotent": True,
        }

    slug = _workspace_slug(request.organization, request.external_session_id)
    workspace = ClientWorkspace(
        name=request.organization,
        slug=slug,
        portal_token_sha256=hash_portal_token(token),
        active=_is_active_status(request.subscription_status),
    )
    session.add(workspace)
    session.flush()

    entitlement = WorkspaceEntitlement(
        workspace_id=workspace.id,
        plan=request.plan,
        allowed_sources=list(policy.allowed_sources),
        monthly_request_limit=policy.monthly_request_limit,
        export_formats=list(policy.export_formats),
        active=_is_active_status(request.subscription_status),
    )
    topic = ClientTopic(
        workspace_id=workspace.id,
        name=request.program_slug.replace("-", " ").title(),
        research_question=request.research_question,
        keywords=_keywords(request.research_question),
    )
    session.add_all([entitlement, topic])
    session.flush()

    mandate = ResearchMandate(
        workspace_id=workspace.id,
        topic_id=topic.id,
        external_reference=f"activation:{request.provider}:{request.external_session_id}",
        title=f"Initial {topic.name} mandate",
        question=request.research_question,
        sources=list(policy.allowed_sources[:4]),
        requested_outputs=["executive_brief", "evidence_ledger", "prior_art"],
        constraints={"origin": "verified_commerce_activation"},
    )
    activation = SubscriptionActivation(
        provider=request.provider,
        external_session_id=request.external_session_id,
        external_customer_id=request.external_customer_id,
        external_subscription_id=request.external_subscription_id,
        workspace_id=workspace.id,
        plan=request.plan,
        contact_email=request.contact_email.lower(),
        organization=request.organization,
        status=request.subscription_status,
        current_period_end=request.current_period_end,
    )
    session.add_all([mandate, activation])
    session.commit()
    return {
        "workspace": {"id": workspace.id, "name": workspace.name, "slug": workspace.slug},
        "portal_token": token,
        "plan": request.plan,
        "subscription_status": request.subscription_status,
        "mandate_id": mandate.id,
        "idempotent": False,
    }


def apply_commerce_event(session: Session, request: CommerceLifecycleEventRequest) -> dict:
    """Apply a previously signature-verified provider event exactly once."""
    existing_event = session.scalar(
        select(CommerceEvent).where(
            CommerceEvent.provider == request.provider,
            CommerceEvent.external_event_id == request.external_event_id,
        )
    )
    if existing_event:
        return {
            "event_id": existing_event.external_event_id,
            "status": existing_event.status,
            "idempotent": True,
        }

    event_fingerprint = request.model_dump(mode="json", exclude_none=True)
    event = CommerceEvent(
        provider=request.provider,
        external_event_id=request.external_event_id,
        event_type=request.event_type,
        external_customer_id=request.external_customer_id,
        external_subscription_id=request.external_subscription_id,
        external_session_id=request.external_session_id,
        payload_sha256=hashlib.sha256(json.dumps(event_fingerprint, sort_keys=True).encode()).hexdigest(),
    )
    session.add(event)

    activation = _activation_for_event(session, request)
    if activation is None:
        event.status = "unmatched"
        session.commit()
        return {"event_id": event.external_event_id, "status": "unmatched", "idempotent": False}

    effective_status = _event_status(request)
    if request.external_customer_id:
        activation.external_customer_id = request.external_customer_id
    if request.external_subscription_id:
        activation.external_subscription_id = request.external_subscription_id
    activation.status = effective_status
    activation.last_event_at = request.event_created_at or datetime.now(UTC)

    workspace = session.get(ClientWorkspace, activation.workspace_id)
    entitlement = session.scalar(
        select(WorkspaceEntitlement).where(WorkspaceEntitlement.workspace_id == activation.workspace_id)
    )
    is_active = _is_active_status(effective_status)
    if workspace:
        workspace.active = is_active
    if entitlement:
        entitlement.active = is_active
    event.status = "processed"
    session.commit()
    return {
        "event_id": event.external_event_id,
        "status": event.status,
        "subscription_status": effective_status,
        "workspace_id": activation.workspace_id,
        "entitlement_active": is_active,
        "idempotent": False,
    }


def _activation_for_event(
    session: Session,
    request: CommerceLifecycleEventRequest,
) -> SubscriptionActivation | None:
    filters = []
    if request.external_subscription_id:
        filters.append(SubscriptionActivation.external_subscription_id == request.external_subscription_id)
    if request.external_session_id:
        filters.append(SubscriptionActivation.external_session_id == request.external_session_id)
    if request.external_customer_id:
        filters.append(SubscriptionActivation.external_customer_id == request.external_customer_id)
    for condition in filters:
        activation = session.scalar(
            select(SubscriptionActivation)
            .where(SubscriptionActivation.provider == request.provider, condition)
            .order_by(SubscriptionActivation.activated_at.desc())
        )
        if activation:
            return activation
    return None


def _event_status(request: CommerceLifecycleEventRequest) -> str:
    if request.status:
        return request.status.lower()
    return {
        "checkout.session.completed": "active",
        "invoice.paid": "active",
        "invoice.payment_failed": "past_due",
        "customer.subscription.deleted": "canceled",
        "charge.refunded": "refunded",
    }.get(request.event_type, "active")


def _is_active_status(value: str) -> bool:
    return value.lower() in {"active", "paid", "trialing", "complete", "no_payment_required"}


def _keywords(question: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "could",
        "from",
        "have",
        "into",
        "should",
        "their",
        "these",
        "this",
        "those",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    words = re.findall(r"[a-z0-9][a-z0-9+-]{3,}", question.lower())
    return sorted({word for word in words if word not in stopwords})[:30] or ["research"]
