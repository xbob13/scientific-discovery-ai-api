import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters import ADAPTERS
from .agenda import half_hour_window
from .agents import LiteratureCartographer, load_documents, persist_connections
from .canonical import store_record
from .config import get_settings
from .models import JobRun, Source
from .reports import candidate_report


@dataclass(frozen=True)
class ResearchCycle:
    question: str
    sources: tuple[str, ...] = ("openalex", "crossref")
    limit_per_source: int = 10
    connection_limit: int = 20
    cycle_window: str = field(default_factory=half_hour_window)

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return f"research-cycle:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


async def run_cycle(session: Session, cycle: ResearchCycle) -> dict:
    settings = get_settings()
    if settings.research_kill_switch:
        raise RuntimeError("global research kill switch is active")
    unknown = set(cycle.sources) - ADAPTERS.keys()
    if unknown:
        raise ValueError(f"unsupported sources: {sorted(unknown)}")

    previous = session.scalar(select(JobRun).where(JobRun.idempotency_key == cycle.key))
    if previous and previous.status == "succeeded":
        return previous.output or {}
    job = previous or JobRun(
        job_type="autonomous_research_cycle",
        idempotency_key=cycle.key,
        input=asdict(cycle),
    )
    session.add(job)
    job.status = "running"
    job.started_at = datetime.now(UTC)
    job.attempts = (job.attempts or 0) + 1
    session.commit()

    output = {
        "question": cycle.question,
        "sources": {},
        "new_versions": 0,
        "connections_created": 0,
        "limitations": [
            "Connections are machine-generated retrieval hypotheses, not scientific conclusions.",
            "Full-text support is not assumed when only title or abstract metadata is available.",
        ],
    }
    harvested_work_ids: set[str] = set()
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for source_name in cycle.sources:
            try:
                records = await ADAPTERS[source_name](client).harvest(
                    cycle.question, cycle.limit_per_source
                )
                source_new = 0
                for record in records:
                    work, created = store_record(session, record)
                    harvested_work_ids.add(work.id)
                    source_new += int(created)
                source = session.scalar(select(Source).where(Source.name == source_name))
                source.last_success_at = datetime.now(UTC)
                source.last_error = None
                session.commit()
                output["sources"][source_name] = {"records": len(records), "new_versions": source_new}
                output["new_versions"] += source_new
            except Exception as exc:
                session.rollback()
                output["sources"][source_name] = {"error": type(exc).__name__}

    documents = load_documents(session)
    proposals = LiteratureCartographer().propose(
        documents,
        cycle.connection_limit,
        focus_work_ids=harvested_work_ids,
    )
    assessed_ids = persist_connections(session, proposals)
    output["connections_created"] = len(assessed_ids)
    output["candidates"] = candidate_report(session, assessed_ids)
    output["completed_at"] = datetime.now(UTC).isoformat()
    complete = all("error" not in value for value in output["sources"].values())
    job.status = "succeeded" if complete else "partial"
    job.output = output
    job.finished_at = datetime.now(UTC)
    session.commit()
    return output


def run_cycle_sync(session: Session, cycle: ResearchCycle) -> dict:
    return asyncio.run(run_cycle(session, cycle))
