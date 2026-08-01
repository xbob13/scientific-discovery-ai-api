from __future__ import annotations

from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import OutreachMessage, ProspectAccount, ResearchInstitution
from .outreach import create_prospect, draft_outreach
from .schemas import OutreachDraftCreate, ProspectCreate

VERIFIED_COLLABORATION_CHANNELS = [
    {
        "name": "Northern Arizona University MPaCT Lab",
        "country_code": "US",
        "institution_type": "university_lab",
        "homepage_url": "https://nano.nau.edu/MPaCT.html",
        "collaboration_url": "https://nano.nau.edu/index.html",
        "contact_email": "mpct.nano@nau.edu",
        "contact_basis": "Public laboratory role account on the official NAU site",
        "signal": (
            "NAU's MPaCT Lab publicly invites academic and industrial collaboration in advanced "
            "materials and semiconductor processing."
        ),
    },
    {
        "name": "Northern Arizona University Innovations",
        "country_code": "US",
        "institution_type": "technology_transfer",
        "homepage_url": "https://in.nau.edu/research/innovations/",
        "collaboration_url": "https://in.nau.edu/research/innovations/agreement-templates/",
        "contact_email": "nauinnovations@nau.edu",
        "contact_basis": "Public technology-transfer role account on the official NAU site",
        "signal": (
            "NAU Innovations publishes collaboration-agreement pathways for sponsored research, "
            "material transfer, confidentiality, and licensing."
        ),
    },
    {
        "name": "University of Arizona Tech Launch Arizona",
        "country_code": "US",
        "institution_type": "technology_transfer",
        "homepage_url": "https://techlaunch.arizona.edu/",
        "collaboration_url": "https://techlaunch.arizona.edu/college-science-inventions",
        "contact_email": "info@tla.arizona.edu",
        "contact_basis": "Public technology-commercialization role account on the official university site",
        "signal": (
            "Tech Launch Arizona publicly supports university invention commercialization and "
            "industry engagement across science and engineering."
        ),
    },
    {
        "name": "Critical Materials Innovation Hub at Ames National Laboratory",
        "country_code": "US",
        "institution_type": "national_laboratory",
        "homepage_url": "https://www.ameslab.gov/cmi",
        "collaboration_url": "https://www.ameslab.gov/cmi/cmi-partners",
        "contact_email": "CMIaffiliates@ameslab.gov",
        "contact_basis": "Public affiliate-program role account on the official Ames Laboratory site",
        "signal": (
            "The Critical Materials Innovation Hub publishes an affiliate pathway for organizations "
            "working on critical-material supply, processing, reuse, and substitution."
        ),
    },
]


async def discover_research_institutions(
    session: Session,
    query: str,
    limit: int = 20,
    client: httpx.AsyncClient | None = None,
    seed_verified_channels: bool = True,
) -> dict:
    """Rank active organizations publishing in a technical area via OpenAlex."""
    settings = get_settings()
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30, follow_redirects=True, trust_env=False)
    assert client is not None
    discovered: list[str] = []
    failures: dict[str, str] = {}
    seeded = (
        seed_collaboration_channels(session) if seed_verified_channels else {"institutions": [], "drafts": []}
    )
    params: dict[str, object] = {
        "search": query,
        "group_by": "authorships.institutions.id",
        "per-page": min(200, max(25, limit * 3)),
    }
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key.get_secret_value()
    try:
        response = await client.get(f"{settings.openalex_base_url}/works", params=params)
        response.raise_for_status()
        groups = response.json().get("group_by", [])
        for group in groups[:limit]:
            openalex_id = str(group.get("key") or "").strip()
            if not openalex_id:
                continue
            try:
                institution_key = openalex_id.rsplit("/", 1)[-1]
                detail_params = {}
                if settings.openalex_api_key:
                    detail_params["api_key"] = settings.openalex_api_key.get_secret_value()
                detail = await client.get(
                    f"{settings.openalex_base_url}/institutions/{institution_key}",
                    params=detail_params,
                )
                detail.raise_for_status()
                record = detail.json()
                item = upsert_institution(
                    session,
                    {
                        "openalex_id": openalex_id,
                        "ror_id": (record.get("ids") or {}).get("ror"),
                        "name": (
                            record.get("display_name") or group.get("key_display_name") or institution_key
                        ),
                        "country_code": record.get("country_code"),
                        "institution_type": record.get("type"),
                        "homepage_url": record.get("homepage_url"),
                        "works_count": int(group.get("count") or 0),
                        "relevance_score": _relevance_score(int(group.get("count") or 0)),
                        "discovery_query": query,
                        "source_urls": [
                            openalex_id,
                            *(([record.get("homepage_url")]) if record.get("homepage_url") else []),
                        ],
                        "raw": {
                            "openalex_group": group,
                            "summary_stats": record.get("summary_stats") or {},
                            "topics": (record.get("topics") or [])[:10],
                        },
                    },
                )
                discovered.append(item.id)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                failures[openalex_id] = f"{type(exc).__name__}: {str(exc)[:240]}"
        session.commit()
    except Exception as exc:
        session.rollback()
        failures["openalex"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if own_client:
            await client.aclose()
    return {
        "query": query,
        "discovered": discovered,
        "verified_channels": seeded,
        "failures": failures,
        "method": (
            "Institutions are ranked from OpenAlex authorship groups. Contact channels are seeded "
            "only when a role account and collaboration basis are published on an official site."
        ),
    }


def seed_collaboration_channels(session: Session) -> dict:
    institutions: list[str] = []
    prospects: list[str] = []
    drafts: list[str] = []
    for channel in VERIFIED_COLLABORATION_CHANNELS:
        item = upsert_institution(
            session,
            {
                **{key: value for key, value in channel.items() if key != "signal"},
                "openalex_id": None,
                "ror_id": None,
                "works_count": 0,
                "relevance_score": 72,
                "discovery_query": "verified public collaboration channel",
                "source_urls": [channel["homepage_url"], channel["collaboration_url"]],
                "status": "verified_contact",
                "raw": {"evidence_signal": channel["signal"]},
            },
        )
        institutions.append(item.id)
        prospect = session.scalar(
            select(ProspectAccount).where(ProspectAccount.contact_email == channel["contact_email"].lower())
        )
        if prospect is None:
            domain = channel["contact_email"].split("@", 1)[1]
            prospect = create_prospect(
                session,
                ProspectCreate(
                    name=channel["name"],
                    domain=domain,
                    target_type="research_institution",
                    evidence_signals=[
                        {
                            "summary": channel["signal"],
                            "source_url": channel["collaboration_url"],
                            "weight": 36,
                        },
                        {
                            "summary": (
                                "The contact is a published institutional role account rather than "
                                "an inferred personal address."
                            ),
                            "source_url": channel["homepage_url"],
                            "weight": 28,
                        },
                    ],
                    contact_name="Partnerships team",
                    contact_email=channel["contact_email"],
                    contact_basis=channel["contact_basis"],
                    source_url=channel["collaboration_url"],
                ),
            )
        prospects.append(prospect.id)
        prior = session.scalar(select(OutreachMessage).where(OutreachMessage.prospect_id == prospect.id))
        if prior is None:
            message = draft_outreach(
                session,
                OutreachDraftCreate(
                    prospect_id=prospect.id,
                    program_slug="materials-intelligence-collaboration",
                    value_proposition=(
                        "We would like to explore a bounded collaboration in which our evidence "
                        "engine maps live literature, public patent claims, and materials datasets "
                        "to one of your published technical priorities. Any substantive research, "
                        "data-sharing, licensing, or sponsored-work terms would follow your "
                        "institution's normal agreement process."
                    ),
                ),
            )
            drafts.append(message.id)
    session.commit()
    return {"institutions": institutions, "prospects": prospects, "drafts": drafts}


def upsert_institution(session: Session, values: dict) -> ResearchInstitution:
    values = dict(values)
    if values.get("contact_email"):
        values["contact_email"] = values["contact_email"].lower()
    item = None
    if values.get("openalex_id"):
        item = session.scalar(
            select(ResearchInstitution).where(ResearchInstitution.openalex_id == values["openalex_id"])
        )
    if item is None and values.get("contact_email"):
        item = session.scalar(
            select(ResearchInstitution).where(
                ResearchInstitution.contact_email == values["contact_email"].lower()
            )
        )
    if item is None:
        item = ResearchInstitution(**values)
        session.add(item)
        session.flush()
    else:
        for key, value in values.items():
            if value is not None:
                setattr(item, key, value)
        item.last_discovered_at = datetime.now(UTC)
    return item


def institution_view(item: ResearchInstitution) -> dict:
    return {
        "id": item.id,
        "openalex_id": item.openalex_id,
        "ror_id": item.ror_id,
        "name": item.name,
        "country_code": item.country_code,
        "institution_type": item.institution_type,
        "homepage_url": item.homepage_url,
        "collaboration_url": item.collaboration_url,
        "contact_email": item.contact_email,
        "contact_basis": item.contact_basis,
        "works_count": item.works_count,
        "relevance_score": item.relevance_score,
        "discovery_query": item.discovery_query,
        "source_urls": item.source_urls,
        "status": item.status,
        "last_discovered_at": item.last_discovered_at,
    }


def _relevance_score(works_count: int) -> float:
    return round(min(100, 25 + 15 * max(0, len(str(max(1, works_count))) - 1)), 1)
