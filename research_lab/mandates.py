from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .clients import verify_portal_token
from .commerce import PLAN_POLICIES
from .models import (
    ClientTopic,
    ClientWorkspace,
    IntelligenceBrief,
    ResearchMandate,
    WorkspaceEntitlement,
)
from .schemas import ClientMandateCreate


def authenticated_workspace(session: Session, slug: str, token: str) -> ClientWorkspace | None:
    workspace = session.scalar(
        select(ClientWorkspace).where(
            ClientWorkspace.slug == slug,
            ClientWorkspace.active.is_(True),
        )
    )
    if workspace is None or not verify_portal_token(workspace, token):
        return None
    return workspace


def entitlement_for(session: Session, workspace: ClientWorkspace) -> WorkspaceEntitlement:
    entitlement = session.scalar(
        select(WorkspaceEntitlement).where(WorkspaceEntitlement.workspace_id == workspace.id)
    )
    if entitlement:
        period_start = entitlement.period_started_at
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=UTC)
        if datetime.now(UTC) - period_start >= timedelta(days=30):
            entitlement.period_started_at = datetime.now(UTC)
            session.commit()
        return entitlement
    policy = PLAN_POLICIES["commissioned-brief"]
    entitlement = WorkspaceEntitlement(
        workspace_id=workspace.id,
        plan="commissioned-brief",
        allowed_sources=list(policy.allowed_sources),
        monthly_request_limit=policy.monthly_request_limit,
        export_formats=list(policy.export_formats),
    )
    session.add(entitlement)
    session.commit()
    return entitlement


def create_mandate(
    session: Session,
    workspace: ClientWorkspace,
    request: ClientMandateCreate,
) -> ResearchMandate:
    entitlement = entitlement_for(session, workspace)
    if not entitlement.active:
        raise PermissionError("workspace entitlement is inactive")
    requested_sources = sorted(set(request.sources))
    disallowed = set(requested_sources) - set(entitlement.allowed_sources)
    if disallowed:
        raise PermissionError(f"sources not included in plan: {sorted(disallowed)}")
    if request.external_reference:
        existing = session.scalar(
            select(ResearchMandate).where(
                ResearchMandate.workspace_id == workspace.id,
                ResearchMandate.external_reference == request.external_reference,
            )
        )
        if existing:
            return existing

    since = entitlement.period_started_at
    used = (
        session.scalar(
            select(func.count())
            .select_from(ResearchMandate)
            .where(
                ResearchMandate.workspace_id == workspace.id,
                ResearchMandate.created_at >= since,
            )
        )
        or 0
    )
    if used >= entitlement.monthly_request_limit:
        raise OverflowError("workspace request allowance has been reached")

    topic = session.scalar(
        select(ClientTopic)
        .where(ClientTopic.workspace_id == workspace.id, ClientTopic.active.is_(True))
        .order_by(ClientTopic.created_at)
    )
    mandate = ResearchMandate(
        workspace_id=workspace.id,
        topic_id=topic.id if topic else None,
        external_reference=request.external_reference,
        title=request.title,
        question=request.question,
        sources=requested_sources,
        requested_outputs=sorted(set(request.requested_outputs)),
        constraints=request.constraints,
    )
    session.add(mandate)
    session.commit()
    return mandate


def mandate_view(mandate: ResearchMandate) -> dict:
    return {
        "id": mandate.id,
        "title": mandate.title,
        "question": mandate.question,
        "sources": mandate.sources,
        "requested_outputs": mandate.requested_outputs,
        "constraints": mandate.constraints,
        "status": mandate.status,
        "result": mandate.result,
        "error": mandate.error,
        "created_at": mandate.created_at,
        "started_at": mandate.started_at,
        "completed_at": mandate.completed_at,
    }


async def process_mandate(session: Session, mandate: ResearchMandate) -> dict:
    from .orchestrator import ResearchCycle, run_cycle
    from .patents import search_patents, synthesize_patent_evidence

    if mandate.status not in {"queued", "failed"}:
        return mandate.result or {}
    claimed = session.execute(
        update(ResearchMandate)
        .where(
            ResearchMandate.id == mandate.id,
            ResearchMandate.status.in_(["queued", "failed"]),
        )
        .values(status="running", started_at=datetime.now(UTC), error=None)
    )
    session.commit()
    if claimed.rowcount != 1:
        session.refresh(mandate)
        return mandate.result or {}
    session.refresh(mandate)
    try:
        literature_sources = tuple(
            source for source in mandate.sources if source in {"openalex", "crossref", "datacite"}
        )
        research = {"question": mandate.question, "sources": {}, "candidates": []}
        if literature_sources:
            research = await run_cycle(
                session,
                ResearchCycle(
                    question=mandate.question,
                    sources=literature_sources,
                    cycle_window=f"mandate:{mandate.id}",
                ),
            )
        patent_providers = [
            source for source in mandate.sources if source in {"uspto_odp", "patentsview", "epo_ops"}
        ]
        patent_result = (
            await search_patents(
                session,
                mandate.question,
                patent_providers,
                limit_per_provider=10,
            )
            if patent_providers
            else {"records": [], "provider_failures": {}}
        )
        patent_result.setdefault("landscape", {}).update(
            synthesize_patent_evidence(
                patent_result.get("records", []),
                mandate.question,
                research.get("candidates", []),
            )
        )
        result = {
            **research,
            "patents": patent_result,
            "mandate_id": mandate.id,
            "requested_outputs": mandate.requested_outputs,
        }
        brief = _publish_mandate_brief(session, mandate, result)
        mandate.status = "succeeded"
        mandate.result = {**result, "brief_id": brief.id}
        mandate.completed_at = datetime.now(UTC)
        session.commit()
        return mandate.result
    except Exception as exc:
        session.rollback()
        mandate = session.get(ResearchMandate, mandate.id)
        if mandate:
            mandate.status = "failed"
            mandate.error = f"{type(exc).__name__}: {str(exc)[:500]}"
            mandate.completed_at = datetime.now(UTC)
            session.commit()
        raise


def _publish_mandate_brief(
    session: Session,
    mandate: ResearchMandate,
    result: dict,
) -> IntelligenceBrief:
    topic = session.get(ClientTopic, mandate.topic_id) if mandate.topic_id else None
    if topic is None:
        topic = ClientTopic(
            workspace_id=mandate.workspace_id,
            name=mandate.title,
            research_question=mandate.question,
            keywords=[],
        )
        session.add(topic)
        session.flush()
        mandate.topic_id = topic.id
    cycle_key = f"client-mandate:{mandate.id}"
    existing = session.scalar(
        select(IntelligenceBrief).where(
            IntelligenceBrief.topic_id == topic.id,
            IntelligenceBrief.cycle_key == cycle_key,
        )
    )
    if existing:
        return existing
    candidates = result.get("candidates", [])
    patents = result.get("patents", {}).get("records", [])
    patent_synthesis = result.get("patents", {}).get("landscape", {})
    brief = IntelligenceBrief(
        workspace_id=mandate.workspace_id,
        topic_id=topic.id,
        cycle_key=cycle_key,
        title=mandate.title,
        executive_summary=(
            f"Research run assembled {len(candidates)} evidence-backed literature candidates and "
            f"{len(patents)} patent records for review. Outputs are research intelligence, not "
            "technical certification or a legal opinion on patent status."
        ),
        payload={
            "question": mandate.question,
            "candidates": candidates,
            "patents": patents,
            "patent_synthesis": patent_synthesis,
            "source_summary": result.get("sources", {}),
            "limitations": result.get("limitations", []),
            "completed_at": result.get("completed_at"),
        },
    )
    session.add(brief)
    session.flush()
    return brief
