from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .clients import hash_portal_token
from .config import get_settings
from .models import (
    ClientTopic,
    ClientWorkspace,
    ResearchMandate,
    SubscriptionActivation,
    WorkspaceEntitlement,
)
from .schemas import SubscriptionActivationRequest


@dataclass(frozen=True)
class PlanPolicy:
    monthly_request_limit: int
    allowed_sources: tuple[str, ...]
    export_formats: tuple[str, ...] = ("json", "markdown")


PLAN_POLICIES = {
    "commissioned-brief": PlanPolicy(
        monthly_request_limit=3,
        allowed_sources=("openalex", "crossref", "datacite", "patentsview"),
    ),
    "continuous-monitor": PlanPolicy(
        monthly_request_limit=30,
        allowed_sources=("openalex", "crossref", "datacite", "patentsview", "epo_ops"),
    ),
    "enterprise": PlanPolicy(
        monthly_request_limit=250,
        allowed_sources=("openalex", "crossref", "datacite", "patentsview", "epo_ops"),
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
        return {
            "workspace": {"id": workspace.id, "name": workspace.name, "slug": workspace.slug},
            "portal_token": token,
            "plan": existing.plan,
            "idempotent": True,
        }

    slug = _workspace_slug(request.organization, request.external_session_id)
    workspace = ClientWorkspace(
        name=request.organization,
        slug=slug,
        portal_token_sha256=hash_portal_token(token),
    )
    session.add(workspace)
    session.flush()

    entitlement = WorkspaceEntitlement(
        workspace_id=workspace.id,
        plan=request.plan,
        allowed_sources=list(policy.allowed_sources),
        monthly_request_limit=policy.monthly_request_limit,
        export_formats=list(policy.export_formats),
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
        workspace_id=workspace.id,
        plan=request.plan,
        contact_email=request.contact_email.lower(),
        organization=request.organization,
    )
    session.add_all([mandate, activation])
    session.commit()
    return {
        "workspace": {"id": workspace.id, "name": workspace.name, "slug": workspace.slug},
        "portal_token": token,
        "plan": request.plan,
        "mandate_id": mandate.id,
        "idempotent": False,
    }


def _keywords(question: str) -> list[str]:
    stopwords = {
        "about", "after", "could", "from", "have", "into", "should", "their", "these",
        "this", "those", "using", "what", "when", "where", "which", "with", "would",
    }
    words = re.findall(r"[a-z0-9][a-z0-9+-]{3,}", question.lower())
    return sorted({word for word in words if word not in stopwords})[:30] or ["research"]
