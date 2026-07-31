from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import OutreachMessage, ProspectAccount
from .schemas import OutreachDraftCreate, ProspectCreate


def create_prospect(session: Session, request: ProspectCreate) -> ProspectAccount:
    signals = [_normalize_signal(item) for item in request.evidence_signals]
    if request.contact_email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", request.contact_email):
        raise ValueError("contact email is invalid")
    score = min(100.0, round(sum(item["weight"] for item in signals), 2))
    prospect = ProspectAccount(
        name=request.name,
        domain=request.domain.lower() if request.domain else None,
        target_type=request.target_type,
        relevance_score=score,
        evidence_signals=signals,
        contact_name=request.contact_name,
        contact_email=request.contact_email.lower() if request.contact_email else None,
        contact_basis=request.contact_basis,
        source_url=str(request.source_url) if request.source_url else None,
    )
    session.add(prospect)
    session.commit()
    return prospect


def draft_outreach(
    session: Session,
    request: OutreachDraftCreate,
) -> OutreachMessage:
    prospect = session.get(ProspectAccount, request.prospect_id)
    if prospect is None:
        raise LookupError("prospect not found")
    if prospect.suppressed:
        raise PermissionError("prospect is suppressed")
    if not prospect.contact_email or not prospect.contact_basis:
        raise PermissionError("verified business contact and contact basis are required")
    evidence = prospect.evidence_signals[:5]
    signal_summary = evidence[0]["summary"] if evidence else "your public R&D priorities"
    subject = f"Evidence intelligence for {prospect.name}'s technical priorities"
    greeting = f"Hello {prospect.contact_name}," if prospect.contact_name else "Hello,"
    body = (
        f"{greeting}\n\n"
        f"I am reaching out from Patterson Research Labs because {signal_summary}. "
        f"We operate a governed {request.program_slug.replace('-', ' ')} intelligence workflow "
        "that continuously maps literature, datasets, and patent signals to a defined industrial "
        "research decision.\n\n"
        f"{request.value_proposition.strip()}\n\n"
        "If this is relevant, I can share a concise example of the evidence package and its "
        "validation gates. If it is not, reply and we will not contact you again.\n\n"
        "Patterson Research Labs"
    )
    message = OutreachMessage(
        prospect_id=prospect.id,
        subject=subject,
        body=body,
        evidence=evidence,
    )
    session.add(message)
    session.commit()
    return message


def approve_outreach(
    session: Session,
    message: OutreachMessage,
    approved_by: str,
) -> OutreachMessage:
    if message.status not in {"draft", "approved"}:
        raise PermissionError(f"message cannot be approved from status {message.status}")
    prospect = session.get(ProspectAccount, message.prospect_id)
    if prospect is None or prospect.suppressed:
        raise PermissionError("prospect is missing or suppressed")
    message.status = "approved"
    message.approved_by = approved_by
    message.approved_at = datetime.now(UTC)
    session.commit()
    return message


async def deliver_outreach(
    session: Session,
    message: OutreachMessage,
    client: httpx.AsyncClient | None = None,
) -> OutreachMessage:
    settings = get_settings()
    if not settings.outreach_send_enabled:
        raise PermissionError("outreach delivery is disabled")
    if not settings.outreach_delivery_webhook_url:
        raise RuntimeError("outreach delivery webhook is not configured")
    if message.status != "approved" or not message.approved_at:
        raise PermissionError("human approval is required before delivery")
    prospect = session.get(ProspectAccount, message.prospect_id)
    if prospect is None or prospect.suppressed or not prospect.contact_email:
        raise PermissionError("prospect cannot be contacted")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20, trust_env=False)
    assert client is not None
    headers = {"Content-Type": "application/json"}
    if settings.outreach_delivery_token:
        headers["Authorization"] = (
            f"Bearer {settings.outreach_delivery_token.get_secret_value()}"
        )
    try:
        response = await client.post(
            settings.outreach_delivery_webhook_url,
            headers=headers,
            json={
                "to": prospect.contact_email,
                "subject": message.subject,
                "text": message.body,
                "metadata": {
                    "message_id": message.id,
                    "prospect_id": prospect.id,
                    "approved_by": message.approved_by,
                    "approved_at": message.approved_at.isoformat(),
                },
            },
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        message.status = "sent"
        message.sent_at = datetime.now(UTC)
        message.provider_message_id = str(payload.get("id") or payload.get("message_id") or "")[:240] or None
        message.error = None
        prospect.status = "contacted"
        session.commit()
        return message
    except Exception as exc:
        message.status = "delivery_failed"
        message.error = f"{type(exc).__name__}: {str(exc)[:500]}"
        session.commit()
        raise
    finally:
        if own_client:
            await client.aclose()


def suppress_prospect(session: Session, prospect: ProspectAccount) -> ProspectAccount:
    prospect.suppressed = True
    prospect.status = "suppressed"
    for message in session.scalars(
        select(OutreachMessage).where(
            OutreachMessage.prospect_id == prospect.id,
            OutreachMessage.status.in_(["draft", "approved"]),
        )
    ):
        message.status = "suppressed"
    session.commit()
    return prospect


def prospect_view(prospect: ProspectAccount) -> dict:
    return {
        "id": prospect.id,
        "name": prospect.name,
        "domain": prospect.domain,
        "target_type": prospect.target_type,
        "relevance_score": prospect.relevance_score,
        "evidence_signals": prospect.evidence_signals,
        "contact_name": prospect.contact_name,
        "contact_email": prospect.contact_email,
        "contact_basis": prospect.contact_basis,
        "source_url": prospect.source_url,
        "status": prospect.status,
        "suppressed": prospect.suppressed,
        "created_at": prospect.created_at,
    }


def message_view(message: OutreachMessage) -> dict:
    return {
        "id": message.id,
        "prospect_id": message.prospect_id,
        "subject": message.subject,
        "body": message.body,
        "evidence": message.evidence,
        "status": message.status,
        "approved_by": message.approved_by,
        "approved_at": message.approved_at,
        "sent_at": message.sent_at,
        "provider_message_id": message.provider_message_id,
        "error": message.error,
        "created_at": message.created_at,
    }


async def discover_and_draft_prospects(
    session: Session,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Ingest an approved evidence feed; never crawl sites or approve messages."""
    settings = get_settings()
    if not settings.prospect_discovery_enabled:
        raise PermissionError("prospect discovery is disabled")
    if not settings.prospect_discovery_feed_url:
        raise RuntimeError("prospect discovery feed is not configured")
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30, trust_env=False)
    assert client is not None
    headers = {"Accept": "application/json"}
    if settings.prospect_discovery_token:
        headers["Authorization"] = f"Bearer {settings.prospect_discovery_token.get_secret_value()}"
    created: list[str] = []
    drafted: list[str] = []
    skipped: list[str] = []
    try:
        response = await client.get(settings.prospect_discovery_feed_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("prospects", []) if isinstance(payload, dict) else []
        for item in items[:250]:
            try:
                candidate = ProspectCreate.model_validate(item)
                existing = None
                if candidate.domain:
                    existing = session.scalar(
                        select(ProspectAccount).where(
                            ProspectAccount.domain == candidate.domain.lower()
                        )
                    )
                if existing:
                    skipped.append(existing.id)
                    continue
                prospect = create_prospect(session, candidate)
                created.append(prospect.id)
                if prospect.contact_email and prospect.contact_basis:
                    draft = draft_outreach(
                        session,
                        OutreachDraftCreate(
                            prospect_id=prospect.id,
                            program_slug=str(item.get("program_slug") or "custom")[:120],
                            value_proposition=str(
                                item.get("value_proposition")
                                or "A tailored evidence monitor can map current research and patent "
                                "signals to the technical priorities in your public program."
                            )[:1200],
                        ),
                    )
                    drafted.append(draft.id)
            except Exception:
                session.rollback()
                label = item if not isinstance(item, dict) else item.get("domain") or item.get("name")
                skipped.append(str(label or "invalid")[:300])
        return {"created": created, "drafted": drafted, "skipped": skipped}
    finally:
        if own_client:
            await client.aclose()


async def deliver_approved_messages(session: Session, limit: int = 25) -> dict:
    """Deliver only messages that already crossed the explicit human-approval gate."""
    messages = session.scalars(
        select(OutreachMessage)
        .where(OutreachMessage.status == "approved", OutreachMessage.approved_at.is_not(None))
        .order_by(OutreachMessage.approved_at)
        .limit(max(1, min(limit, 100)))
    ).all()
    sent: list[str] = []
    failures: dict[str, str] = {}
    for message in messages:
        try:
            await deliver_outreach(session, message)
            sent.append(message.id)
        except Exception as exc:
            failures[message.id] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"sent": sent, "failures": failures}


def _normalize_signal(value: dict) -> dict:
    summary = str(value.get("summary") or value.get("title") or "").strip()[:600]
    source_url = str(value.get("source_url") or value.get("url") or "").strip()[:1000]
    if len(summary) < 10 or not source_url.startswith(("https://", "http://")):
        raise ValueError("every evidence signal requires a summary and public source URL")
    try:
        weight = float(value.get("weight", 20))
    except (TypeError, ValueError):
        weight = 20
    return {"summary": summary, "source_url": source_url, "weight": min(40, max(1, weight))}
