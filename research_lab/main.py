import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .adapters import ADAPTERS
from .auth import require_service_token
from .canonical import store_record
from .clients import hash_portal_token
from .commerce import activate_subscription, apply_commerce_event
from .compute import run_sheet_resistance
from .config import get_settings
from .datasets import discover_optimade_providers, seed_dataset_registry
from .db import Base, engine, get_session
from .institutions import discover_research_institutions, institution_view
from .mandates import (
    authenticated_workspace,
    create_mandate,
    entitlement_for,
    mandate_view,
    process_mandate,
)
from .models import (
    ClientTopic,
    ClientWorkspace,
    ComputeRun,
    ConnectionCandidate,
    DatasetDefinition,
    IntelligenceBrief,
    JobRun,
    OutreachMessage,
    PatentDocument,
    ProspectAccount,
    ResearchInstitution,
    ResearchMandate,
    Source,
    Work,
)
from .outreach import (
    autonomous_outreach_readiness,
    create_prospect,
    deliver_outreach,
    draft_outreach,
    message_view,
    prospect_view,
    suppress_from_unsubscribe,
    suppress_prospect,
)
from .patents import patent_view, search_patents
from .reports import candidate_report
from .schemas import (
    ClientMandateCreate,
    ClientTopicCreate,
    ClientWorkspaceCreate,
    CommerceLifecycleEventRequest,
    ComputeRequest,
    HarvestRequest,
    InstitutionDiscoveryRequest,
    OutreachDraftCreate,
    PatentSearchRequest,
    ProspectCreate,
    SubscriptionActivationRequest,
)

logger = logging.getLogger("research_lab")


async def process_mandate_by_id(mandate_id: str) -> None:
    """Run after a response while the scheduled durable worker remains the fallback."""
    from .db import SessionLocal

    with SessionLocal() as session:
        mandate = session.get(ResearchMandate, mandate_id)
        if mandate is None:
            return
        try:
            await process_mandate(session, mandate)
        except Exception:
            logger.exception("background client mandate failed", extra={"mandate_id": mandate_id})


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    from .db import SessionLocal

    with SessionLocal() as session:
        seed_dataset_registry(session)
    yield


app = FastAPI(
    title="Patterson Research Labs API",
    version="0.3.0",
    description=(
        "Governed research, patent, and client intelligence. Machine outputs remain auditable "
        "research leads until reviewed."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "product": "Patterson Research Labs",
        "status": "operational",
        "capabilities": [
            "research_intelligence",
            "patent_claim_synthesis",
            "client_workspaces",
            "institution_network",
            "autonomous_commercial_operations",
        ],
        "claims": "evidence-gated research intelligence",
    }


@app.get("/health")
def health(session: Session = Depends(get_session)):
    session.execute(select(1))
    settings = get_settings()
    return {
        "status": "healthy",
        "database": "connected",
        "kill_switch": settings.research_kill_switch,
        "timestamp": datetime.now(UTC),
    }


@app.get("/v1/sources", dependencies=[Depends(require_service_token)])
def sources(session: Session = Depends(get_session)):
    return [
        {
            "name": source.name,
            "enabled": source.enabled,
            "last_success_at": source.last_success_at,
            "last_error": source.last_error,
        }
        for source in session.scalars(select(Source).order_by(Source.name))
    ]


@app.post("/v1/harvest", dependencies=[Depends(require_service_token)])
async def harvest(request: HarvestRequest, session: Session = Depends(get_session)):
    settings = get_settings()
    if settings.research_kill_switch:
        raise HTTPException(status_code=503, detail="global research kill switch is active")
    unknown = set(request.sources) - ADAPTERS.keys()
    if unknown:
        raise HTTPException(status_code=422, detail=f"unsupported sources: {sorted(unknown)}")
    job_key = (
        f"harvest:{'|'.join(sorted(request.sources))}:{request.query.lower()}:{request.limit_per_source}"
    )
    prior = session.scalar(select(JobRun).where(JobRun.idempotency_key == job_key))
    if prior and prior.status == "succeeded":
        return prior.output
    job = prior or JobRun(job_type="harvest", idempotency_key=job_key, input=request.model_dump())
    session.add(job)
    job.status, job.started_at, job.attempts = "running", datetime.now(UTC), job.attempts + 1
    session.commit()

    summary = {"created_versions": 0, "canonical_works": [], "source_failures": {}}
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for source_name in request.sources:
            try:
                adapter = ADAPTERS[source_name](client)
                records = await adapter.harvest(request.query, request.limit_per_source)
                for record in records:
                    work, created = store_record(session, record)
                    summary["created_versions"] += int(created)
                    summary["canonical_works"].append(work.id)
                source = session.scalar(select(Source).where(Source.name == source_name))
                if source is None:
                    source = Source(name=source_name, base_url=adapter.base_url)
                    session.add(source)
                source.last_success_at, source.last_error = datetime.now(UTC), None
                session.commit()
            except Exception as exc:
                session.rollback()
                summary["source_failures"][source_name] = type(exc).__name__
                logger.exception("source harvest failed", extra={"source": source_name})
    summary["canonical_works"] = sorted(set(summary["canonical_works"]))
    job.status = "succeeded" if not summary["source_failures"] else "partial"
    job.output, job.finished_at = summary, datetime.now(UTC)
    session.commit()
    return summary


@app.post("/v1/compute/sheet-resistance", dependencies=[Depends(require_service_token)])
def sheet_resistance(request: ComputeRequest, session: Session = Depends(get_session)):
    if get_settings().research_kill_switch:
        raise HTTPException(status_code=503, detail="global research kill switch is active")
    output = run_sheet_resistance(request)
    replay = run_sheet_resistance(request)
    run = ComputeRun(
        analysis_type="sheet_resistance_monte_carlo",
        inputs=request.model_dump(),
        code_sha256=output["code_sha256"],
        seed=request.seed,
        result=output["result"],
        artifact_sha256=output["artifact_sha256"],
        reproducible=output["artifact_sha256"] == replay["artifact_sha256"],
        finished_at=output["finished_at"],
    )
    session.add(run)
    session.commit()
    return {
        "id": run.id,
        **output["result"],
        "artifact_sha256": run.artifact_sha256,
        "reproducible": run.reproducible,
    }


@app.get("/v1/dashboard", dependencies=[Depends(require_service_token)])
def dashboard(session: Session = Depends(get_session)):
    return {
        "canonical_works": session.scalar(select(func.count()).select_from(Work)),
        "connections_pending_review": session.scalar(
            select(func.count())
            .select_from(ConnectionCandidate)
            .where(ConnectionCandidate.status == "pending_review")
        ),
        "compute_runs": session.scalar(select(func.count()).select_from(ComputeRun)),
        "queued_client_mandates": session.scalar(
            select(func.count()).select_from(ResearchMandate).where(ResearchMandate.status == "queued")
        ),
        "patent_records": session.scalar(select(func.count()).select_from(PatentDocument)),
        "research_institutions": session.scalar(select(func.count()).select_from(ResearchInstitution)),
        "outreach_queued": session.scalar(
            select(func.count()).select_from(OutreachMessage).where(OutreachMessage.status == "queued")
        ),
        "outreach_held": session.scalar(
            select(func.count()).select_from(OutreachMessage).where(OutreachMessage.status == "held")
        ),
        "outreach_sent": session.scalar(
            select(func.count()).select_from(OutreachMessage).where(OutreachMessage.status == "sent")
        ),
        "autonomous_outreach": autonomous_outreach_readiness(),
        "kill_switch": get_settings().research_kill_switch,
        "medical_sandbox": "isolated; no clinical guidance endpoints enabled",
    }


@app.get("/v1/findings", dependencies=[Depends(require_service_token)])
def findings(limit: int = 20, session: Session = Depends(get_session)):
    """Return evidence-backed machine proposals in review-priority order."""
    return {
        "disclaimer": "Research leads only; priority scores are not truth probabilities.",
        "candidates": candidate_report(session, limit=max(1, min(limit, 100))),
    }


@app.get("/v1/data-sources", dependencies=[Depends(require_service_token)])
def data_sources(session: Session = Depends(get_session)):
    seed_dataset_registry(session)
    return [
        {
            "code": item.code,
            "name": item.name,
            "category": item.category,
            "base_url": item.base_url,
            "homepage_url": item.homepage_url,
            "adapter": item.adapter,
            "access_tier": item.access_tier,
            "license_summary": item.license_summary,
            "redistribution_policy": item.redistribution_policy,
            "capabilities": item.capabilities,
            "enabled": item.enabled,
            "last_discovered_at": item.last_discovered_at,
        }
        for item in session.scalars(
            select(DatasetDefinition).order_by(DatasetDefinition.category, DatasetDefinition.name)
        )
    ]


@app.post("/v1/data-sources/sync", dependencies=[Depends(require_service_token)])
async def sync_data_sources(session: Session = Depends(get_session)):
    core = seed_dataset_registry(session)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        providers = await discover_optimade_providers(session, client)
    return {"core_datasets": core, "optimade_providers": providers}


@app.post("/v1/client-workspaces", dependencies=[Depends(require_service_token)])
def create_client_workspace(
    request: ClientWorkspaceCreate,
    session: Session = Depends(get_session),
):
    if session.scalar(select(ClientWorkspace).where(ClientWorkspace.slug == request.slug)):
        raise HTTPException(status_code=409, detail="workspace slug already exists")
    workspace = ClientWorkspace(
        name=request.name,
        slug=request.slug,
        portal_token_sha256=hash_portal_token(request.portal_token),
    )
    session.add(workspace)
    session.commit()
    return {"id": workspace.id, "name": workspace.name, "slug": workspace.slug}


@app.post(
    "/v1/client-workspaces/{workspace_id}/topics",
    dependencies=[Depends(require_service_token)],
)
def create_client_topic(
    workspace_id: str,
    request: ClientTopicCreate,
    session: Session = Depends(get_session),
):
    workspace = session.get(ClientWorkspace, workspace_id)
    if workspace is None or not workspace.active:
        raise HTTPException(status_code=404, detail="active workspace not found")
    topic = ClientTopic(
        workspace_id=workspace.id,
        name=request.name,
        research_question=request.research_question,
        keywords=sorted({keyword.strip().lower() for keyword in request.keywords}),
    )
    session.add(topic)
    session.commit()
    return {
        "id": topic.id,
        "workspace_id": workspace.id,
        "name": topic.name,
        "research_question": topic.research_question,
        "keywords": topic.keywords,
    }


@app.post("/v1/subscriptions/activate", dependencies=[Depends(require_service_token)])
def activate_verified_subscription(
    request: SubscriptionActivationRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Provision access after the caller has independently verified checkout completion."""
    try:
        activation = activate_subscription(session, request)
        if activation.get("mandate_id"):
            background_tasks.add_task(process_mandate_by_id, activation["mandate_id"])
        return activation
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/subscriptions/events", dependencies=[Depends(require_service_token)])
def apply_verified_commerce_event(
    request: CommerceLifecycleEventRequest,
    session: Session = Depends(get_session),
):
    """Apply a payment lifecycle event after the edge function verifies its signature."""
    return apply_commerce_event(session, request)


@app.post("/v1/patents/search", dependencies=[Depends(require_service_token)])
async def patents_search(request: PatentSearchRequest, session: Session = Depends(get_session)):
    if get_settings().research_kill_switch:
        raise HTTPException(status_code=503, detail="global research kill switch is active")
    try:
        return await search_patents(
            session,
            request.query,
            request.providers,
            request.limit_per_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/patents", dependencies=[Depends(require_service_token)])
def patents(limit: int = 100, session: Session = Depends(get_session)):
    records = session.scalars(
        select(PatentDocument).order_by(PatentDocument.retrieved_at.desc()).limit(max(1, min(limit, 500)))
    )
    return {
        "records": [patent_view(record) for record in records],
        "disclaimer": "Research metadata only; not a legal-status or freedom-to-operate opinion.",
    }


@app.post("/v1/institutions/discover", dependencies=[Depends(require_service_token)])
async def institutions_discover(
    request: InstitutionDiscoveryRequest,
    session: Session = Depends(get_session),
):
    if get_settings().research_kill_switch:
        raise HTTPException(status_code=503, detail="global research kill switch is active")
    return await discover_research_institutions(
        session,
        request.query,
        request.limit,
        seed_verified_channels=request.seed_verified_channels,
    )


@app.get("/v1/institutions", dependencies=[Depends(require_service_token)])
def institutions(limit: int = 250, session: Session = Depends(get_session)):
    items = session.scalars(
        select(ResearchInstitution)
        .order_by(ResearchInstitution.relevance_score.desc(), ResearchInstitution.name)
        .limit(max(1, min(limit, 500)))
    )
    return {"institutions": [institution_view(item) for item in items]}


@app.post("/v1/prospects", dependencies=[Depends(require_service_token)])
def prospects_create(request: ProspectCreate, session: Session = Depends(get_session)):
    try:
        return prospect_view(create_prospect(session, request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/prospects", dependencies=[Depends(require_service_token)])
def prospects(session: Session = Depends(get_session)):
    return [
        prospect_view(prospect)
        for prospect in session.scalars(
            select(ProspectAccount).order_by(ProspectAccount.relevance_score.desc())
        )
    ]


@app.post("/v1/outreach/drafts", dependencies=[Depends(require_service_token)])
def outreach_draft(request: OutreachDraftCreate, session: Session = Depends(get_session)):
    try:
        return message_view(draft_outreach(session, request))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/outreach", dependencies=[Depends(require_service_token)])
def outreach_messages(session: Session = Depends(get_session)):
    return [
        message_view(message)
        for message in session.scalars(select(OutreachMessage).order_by(OutreachMessage.created_at.desc()))
    ]


@app.post("/v1/outreach/{message_id}/send", dependencies=[Depends(require_service_token)])
async def outreach_send(message_id: str, session: Session = Depends(get_session)):
    message = session.get(OutreachMessage, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="outreach message not found")
    try:
        return message_view(await deliver_outreach(session, message))
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/prospects/{prospect_id}/suppress", dependencies=[Depends(require_service_token)])
def prospect_suppress(prospect_id: str, session: Session = Depends(get_session)):
    prospect = session.get(ProspectAccount, prospect_id)
    if prospect is None:
        raise HTTPException(status_code=404, detail="prospect not found")
    return prospect_view(suppress_prospect(session, prospect))


@app.get("/v1/outreach/readiness", dependencies=[Depends(require_service_token)])
def outreach_readiness():
    return autonomous_outreach_readiness()


@app.get("/v1/outreach/unsubscribe")
def outreach_unsubscribe(
    message_id: str,
    prospect_id: str,
    signature: str,
    session: Session = Depends(get_session),
):
    try:
        suppress_from_unsubscribe(session, message_id, prospect_id, signature)
        return {"status": "suppressed", "message": "This address will not receive further outreach."}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v1/mandates/process-pending", dependencies=[Depends(require_service_token)])
async def process_pending_mandates(limit: int = 5, session: Session = Depends(get_session)):
    mandates = session.scalars(
        select(ResearchMandate)
        .where(ResearchMandate.status == "queued")
        .order_by(ResearchMandate.priority.desc(), ResearchMandate.created_at)
        .limit(max(1, min(limit, 25)))
    ).all()
    processed = []
    failures = {}
    for mandate in mandates:
        try:
            await process_mandate(session, mandate)
            processed.append(mandate.id)
        except Exception as exc:
            failures[mandate.id] = f"{type(exc).__name__}: {str(exc)[:300]}"
    remaining = (
        session.scalar(
            select(func.count()).select_from(ResearchMandate).where(ResearchMandate.status == "queued")
        )
        or 0
    )
    return {"processed": processed, "failures": failures, "remaining": remaining}


@app.get("/v1/client-portal/{slug}/briefs")
def client_portal_briefs(
    slug: str,
    x_client_token: str = Header(default=""),
    session: Session = Depends(get_session),
):
    workspace = authenticated_workspace(session, slug, x_client_token)
    if workspace is None:
        raise HTTPException(status_code=401, detail="invalid client portal credentials")
    entitlement = entitlement_for(session, workspace)
    briefs = session.scalars(
        select(IntelligenceBrief)
        .where(
            IntelligenceBrief.workspace_id == workspace.id,
            IntelligenceBrief.status == "published",
        )
        .order_by(IntelligenceBrief.created_at.desc())
        .limit(100)
    )
    return {
        "workspace": {"name": workspace.name, "slug": workspace.slug},
        "entitlement": {
            "plan": entitlement.plan,
            "allowed_sources": entitlement.allowed_sources,
            "monthly_request_limit": entitlement.monthly_request_limit,
            "export_formats": entitlement.export_formats,
            "active": entitlement.active,
        },
        "mandates": [
            mandate_view(mandate)
            for mandate in session.scalars(
                select(ResearchMandate)
                .where(ResearchMandate.workspace_id == workspace.id)
                .order_by(ResearchMandate.created_at.desc())
                .limit(100)
            )
        ],
        "briefs": [
            {
                "id": brief.id,
                "title": brief.title,
                "executive_summary": brief.executive_summary,
                "payload": brief.payload,
                "created_at": brief.created_at,
            }
            for brief in briefs
        ],
    }


@app.post("/v1/client-portal/{slug}/mandates", status_code=202)
def client_portal_create_mandate(
    slug: str,
    request: ClientMandateCreate,
    background_tasks: BackgroundTasks,
    x_client_token: str = Header(default=""),
    session: Session = Depends(get_session),
):
    workspace = authenticated_workspace(session, slug, x_client_token)
    if workspace is None:
        raise HTTPException(status_code=401, detail="invalid client portal credentials")
    if get_settings().research_kill_switch:
        raise HTTPException(status_code=503, detail="global research kill switch is active")
    try:
        mandate = create_mandate(session, workspace, request)
        background_tasks.add_task(process_mandate_by_id, mandate.id)
        return mandate_view(mandate)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
