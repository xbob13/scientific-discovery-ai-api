from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import OutreachMessage, ProspectAccount
from .schemas import OutreachDraftCreate, ProspectCreate

CONSUMER_EMAIL_DOMAINS = {
    "aol.com",
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "outlook.com",
    "proton.me",
    "protonmail.com",
    "yahoo.com",
}


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
    prior = session.scalar(
        select(OutreachMessage).where(
            OutreachMessage.prospect_id == prospect.id,
            OutreachMessage.status.not_in(["suppressed"]),
        )
    )
    if prior:
        raise PermissionError("a first-touch message already exists for this prospect")
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
        "If this is relevant, reply and the system will route a concise example of the evidence "
        "package and its validation gates. If it is not relevant, use the opt-out link below and "
        "the account will be suppressed automatically."
    )
    message = OutreachMessage(
        prospect_id=prospect.id,
        subject=subject,
        body=body,
        evidence=evidence,
        status="held",
    )
    session.add(message)
    session.flush()
    reason = _policy_hold_reason(session, prospect, message)
    if reason is None:
        message.status = "queued"
        message.error = None
        message.body = _delivery_body(message, prospect)
    else:
        message.error = reason
    session.commit()
    return message


async def deliver_outreach(
    session: Session,
    message: OutreachMessage,
    client: httpx.AsyncClient | None = None,
) -> OutreachMessage:
    settings = get_settings()
    readiness = autonomous_outreach_readiness()
    if not readiness["ready"]:
        raise PermissionError(f"autonomous outreach is not ready: {', '.join(readiness['missing'])}")
    if message.status != "queued":
        raise PermissionError(f"message cannot be delivered from status {message.status}")
    prospect = session.get(ProspectAccount, message.prospect_id)
    if prospect is None or prospect.suppressed or not prospect.contact_email:
        raise PermissionError("prospect cannot be contacted")
    reason = _policy_hold_reason(session, prospect, message)
    if reason:
        message.status = "held"
        message.error = reason
        session.commit()
        raise PermissionError(reason)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20, trust_env=False)
    assert client is not None
    headers = {"Content-Type": "application/json"}
    if settings.outreach_delivery_token:
        headers["Authorization"] = f"Bearer {settings.outreach_delivery_token.get_secret_value()}"
    try:
        response = await client.post(
            settings.outreach_delivery_webhook_url,
            headers=headers,
            json={
                "to": prospect.contact_email,
                "subject": message.subject,
                "text": _delivery_body(message, prospect),
                "from_name": settings.outreach_sender_name,
                "reply_to": settings.outreach_reply_to,
                "metadata": {
                    "message_id": message.id,
                    "prospect_id": prospect.id,
                    "policy": "autonomous-outreach-v1",
                    "contact_basis": prospect.contact_basis,
                    "evidence_urls": [item.get("source_url") for item in message.evidence],
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
            OutreachMessage.status.in_(["draft", "held", "queued", "delivery_failed"]),
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
        "policy_gate": "autonomous-outreach-v1",
        "sent_at": message.sent_at,
        "provider_message_id": message.provider_message_id,
        "error": message.error,
        "created_at": message.created_at,
    }


async def discover_and_draft_prospects(
    session: Session,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Ingest a structured evidence feed and policy-queue eligible first touches."""
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
                        select(ProspectAccount).where(ProspectAccount.domain == candidate.domain.lower())
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


async def deliver_queued_messages(session: Session, limit: int = 25) -> dict:
    """Deliver messages that pass the autonomous sender, evidence, and suppression policy."""
    queue_result = queue_eligible_messages(session)
    messages = session.scalars(
        select(OutreachMessage)
        .where(OutreachMessage.status == "queued")
        .order_by(OutreachMessage.created_at)
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
    return {"queue": queue_result, "sent": sent, "failures": failures}


def queue_eligible_messages(session: Session, limit: int = 250) -> dict:
    queued: list[str] = []
    held: dict[str, str] = {}
    messages = session.scalars(
        select(OutreachMessage)
        .where(OutreachMessage.status.in_(["draft", "held", "delivery_failed"]))
        .order_by(OutreachMessage.created_at)
        .limit(max(1, min(limit, 500)))
    ).all()
    for message in messages:
        prospect = session.get(ProspectAccount, message.prospect_id)
        reason = _policy_hold_reason(session, prospect, message) if prospect else "prospect missing"
        if reason:
            message.status = "held"
            message.error = reason
            held[message.id] = reason
            continue
        message.status = "queued"
        message.error = None
        message.body = _delivery_body(message, prospect)
        queued.append(message.id)
    session.commit()
    return {"queued": queued, "held": held}


def autonomous_outreach_readiness() -> dict:
    settings = get_settings()
    missing = []
    if not settings.outreach_autonomous_enabled:
        missing.append("OUTREACH_AUTONOMOUS_ENABLED")
    if not settings.outreach_send_enabled:
        missing.append("OUTREACH_SEND_ENABLED")
    required = {
        "OUTREACH_DELIVERY_WEBHOOK_URL": settings.outreach_delivery_webhook_url,
        "OUTREACH_SENDER_NAME": settings.outreach_sender_name,
        "OUTREACH_REPLY_TO": settings.outreach_reply_to,
        "OUTREACH_POSTAL_ADDRESS": settings.outreach_postal_address,
        "OUTREACH_UNSUBSCRIBE_BASE_URL": settings.outreach_unsubscribe_base_url,
        "OUTREACH_UNSUBSCRIBE_SECRET": settings.outreach_unsubscribe_secret,
    }
    missing.extend(name for name, value in required.items() if not value)
    if settings.outreach_unsubscribe_secret:
        secret = settings.outreach_unsubscribe_secret.get_secret_value()
        if len(secret) < 32:
            missing.append("OUTREACH_UNSUBSCRIBE_SECRET_LENGTH")
    if settings.outreach_reply_to and not re.fullmatch(
        r"[^\s@]+@[^\s@]+\.[^\s@]+", settings.outreach_reply_to
    ):
        missing.append("OUTREACH_REPLY_TO_VALID")
    if (
        settings.outreach_delivery_webhook_url
        and urlparse(settings.outreach_delivery_webhook_url).scheme != "https"
    ):
        missing.append("OUTREACH_DELIVERY_WEBHOOK_HTTPS")
    if (
        settings.outreach_unsubscribe_base_url
        and urlparse(settings.outreach_unsubscribe_base_url).scheme != "https"
    ):
        missing.append("OUTREACH_UNSUBSCRIBE_BASE_URL_HTTPS")
    if settings.outreach_postal_address and len(settings.outreach_postal_address.strip()) < 12:
        missing.append("OUTREACH_POSTAL_ADDRESS_COMPLETE")
    return {"ready": not missing, "missing": sorted(set(missing))}


def suppress_from_unsubscribe(
    session: Session,
    message_id: str,
    prospect_id: str,
    signature: str,
) -> ProspectAccount:
    settings = get_settings()
    if not settings.outreach_unsubscribe_secret:
        raise PermissionError("unsubscribe verification is unavailable")
    expected = _unsubscribe_signature(message_id, prospect_id)
    if not hmac.compare_digest(expected, signature):
        raise PermissionError("unsubscribe signature is invalid")
    message = session.get(OutreachMessage, message_id)
    prospect = session.get(ProspectAccount, prospect_id)
    if message is None or prospect is None or message.prospect_id != prospect.id:
        raise LookupError("outreach record not found")
    return suppress_prospect(session, prospect)


def _policy_hold_reason(
    session: Session,
    prospect: ProspectAccount,
    message: OutreachMessage,
) -> str | None:
    readiness = autonomous_outreach_readiness()
    if not readiness["ready"]:
        return f"sender configuration incomplete: {', '.join(readiness['missing'])}"
    settings = get_settings()
    if prospect.suppressed:
        return "prospect is suppressed"
    if prospect.relevance_score < settings.outreach_min_relevance_score:
        return "relevance score is below the autonomous threshold"
    if not prospect.contact_email or not prospect.contact_basis or not prospect.source_url:
        return "verified contact basis and evidence URL are required"
    email_domain = prospect.contact_email.rsplit("@", 1)[-1].lower()
    if email_domain in CONSUMER_EMAIL_DOMAINS:
        return "consumer email domains are not eligible"
    prospect_domain = (prospect.domain or "").lower().removeprefix("www.")
    if prospect_domain and not (
        email_domain == prospect_domain or email_domain.endswith(f".{prospect_domain}")
    ):
        return "contact email does not match the researched organization domain"
    if urlparse(prospect.source_url).scheme != "https":
        return "evidence source must use HTTPS"
    sent_since = datetime.now(UTC) - timedelta(hours=24)
    sent_count = len(
        session.scalars(select(OutreachMessage).where(OutreachMessage.sent_at >= sent_since)).all()
    )
    if sent_count >= max(1, settings.outreach_daily_send_limit):
        return "daily autonomous delivery cap reached"
    if message.sent_at:
        return "message has already been sent"
    return None


def _delivery_body(message: OutreachMessage, prospect: ProspectAccount) -> str:
    settings = get_settings()
    base_body = re.split(r"\n\n--\n", message.body, maxsplit=1)[0].rstrip()
    unsubscribe = _unsubscribe_url(message.id, prospect.id)
    return (
        f"{base_body}\n\n--\n"
        f"{settings.outreach_sender_name}\n"
        f"{settings.outreach_postal_address}\n"
        f"Reply: {settings.outreach_reply_to}\n"
        f"Opt out: {unsubscribe}"
    )


def _unsubscribe_url(message_id: str, prospect_id: str) -> str:
    settings = get_settings()
    base = str(settings.outreach_unsubscribe_base_url or "").rstrip("/")
    query = urlencode(
        {
            "message_id": message_id,
            "prospect_id": prospect_id,
            "signature": _unsubscribe_signature(message_id, prospect_id),
        }
    )
    return f"{base}/v1/outreach/unsubscribe?{query}"


def _unsubscribe_signature(message_id: str, prospect_id: str) -> str:
    secret = get_settings().outreach_unsubscribe_secret
    if not secret:
        return "unconfigured"
    return hmac.new(
        secret.get_secret_value().encode(),
        f"{message_id}:{prospect_id}".encode(),
        hashlib.sha256,
    ).hexdigest()


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
