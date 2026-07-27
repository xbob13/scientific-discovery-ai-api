import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .adapters import ADAPTERS
from .auth import require_service_token
from .canonical import store_record
from .compute import run_sheet_resistance
from .config import get_settings
from .db import Base, engine, get_session
from .models import ComputeRun, ConnectionCandidate, JobRun, Source, Work
from .schemas import ComputeRequest, HarvestRequest

logger = logging.getLogger("research_lab")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Patterson Research Labs API",
    version="0.1.0",
    description="Evidence-grounded research intelligence. Outputs are unvalidated research hypotheses.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"product": "Patterson Research Labs", "status": "operational", "claims": "unvalidated research"}


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
                records = await ADAPTERS[source_name](client).harvest(request.query, request.limit_per_source)
                for record in records:
                    work, created = store_record(session, record)
                    summary["created_versions"] += int(created)
                    summary["canonical_works"].append(work.id)
                source = session.scalar(select(Source).where(Source.name == source_name))
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
        "kill_switch": get_settings().research_kill_switch,
        "medical_sandbox": "isolated; no clinical guidance endpoints enabled",
    }
