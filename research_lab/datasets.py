from __future__ import annotations

from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import DatasetDefinition

CORE_DATASETS = [
    {
        "code": "openalex",
        "name": "OpenAlex",
        "category": "scholarly_literature",
        "base_url": "https://api.openalex.org",
        "homepage_url": "https://openalex.org",
        "adapter": "openalex",
        "access_tier": "public",
        "license_summary": "Open scholarly metadata; referenced full-text licenses remain record-specific.",
        "redistribution_policy": "metadata_and_links",
        "capabilities": ["works", "authors", "institutions", "concepts", "citations"],
    },
    {
        "code": "crossref",
        "name": "Crossref",
        "category": "scholarly_literature",
        "base_url": "https://api.crossref.org",
        "homepage_url": "https://www.crossref.org",
        "adapter": "crossref",
        "access_tier": "public",
        "license_summary": "Public DOI metadata; abstracts and linked content may carry separate rights.",
        "redistribution_policy": "metadata_and_links",
        "capabilities": ["dois", "works", "funding", "relations", "corrections", "retractions"],
    },
    {
        "code": "optimade-providers",
        "name": "OPTIMADE provider federation",
        "category": "computed_materials",
        "base_url": "https://providers.optimade.org/v1",
        "homepage_url": "https://www.optimade.org",
        "adapter": "optimade_index",
        "access_tier": "public",
        "license_summary": "Provider-specific terms apply; inspect each provider before storing payloads.",
        "redistribution_policy": "metadata_only",
        "capabilities": ["provider_discovery", "structures", "composition", "cross_database_query"],
    },
    {
        "code": "nist-materials",
        "name": "NIST Materials Data Repository",
        "category": "experimental_materials",
        "base_url": "https://materialsdata.nist.gov",
        "homepage_url": "https://materialsdata.nist.gov",
        "adapter": "catalog",
        "access_tier": "public",
        "license_summary": "Public repository; individual depositors define record-level reuse terms.",
        "redistribution_policy": "record_terms",
        "capabilities": ["experimental_data", "materials_models", "measurements"],
    },
    {
        "code": "pubchem",
        "name": "PubChem",
        "category": "chemistry",
        "base_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        "homepage_url": "https://pubchem.ncbi.nlm.nih.gov",
        "adapter": "pug_rest",
        "access_tier": "public",
        "license_summary": "Public NCBI service; source annotations may carry their own terms.",
        "redistribution_policy": "provenance_required",
        "capabilities": ["structures", "identifiers", "properties", "hazards", "patents"],
    },
    {
        "code": "datacite",
        "name": "DataCite",
        "category": "research_datasets",
        "base_url": "https://api.datacite.org",
        "homepage_url": "https://datacite.org",
        "adapter": "datacite",
        "access_tier": "public",
        "license_summary": "Public DOI metadata API; underlying dataset licenses vary.",
        "redistribution_policy": "metadata_only",
        "capabilities": ["datasets", "dois", "citations", "versions", "funding"],
    },
    {
        "code": "doe-osti",
        "name": "DOE OSTI.GOV",
        "category": "technical_reports",
        "base_url": "https://www.osti.gov/api/v1",
        "homepage_url": "https://www.osti.gov",
        "adapter": "osti",
        "access_tier": "public",
        "license_summary": "Public DOE research metadata; full-text rights vary by record.",
        "redistribution_policy": "metadata_and_links",
        "capabilities": ["energy_research", "technical_reports", "software", "datasets"],
    },
    {
        "code": "uspto-patentsview",
        "name": "USPTO PatentsView",
        "category": "patents",
        "base_url": "https://search.patentsview.org",
        "homepage_url": "https://patentsview.org",
        "adapter": "patentsview",
        "access_tier": "public",
        "license_summary": "Research-grade USPTO-derived data; not the official patent record.",
        "redistribution_policy": "metadata_and_links",
        "capabilities": ["patents", "inventors", "assignees", "citations", "technology_landscape"],
    },
    {
        "code": "epo-ops",
        "name": "EPO Open Patent Services",
        "category": "patents",
        "base_url": "https://ops.epo.org/3.2/rest-services",
        "homepage_url": "https://www.epo.org/en/searching-for-patents/data/web-services/ops",
        "adapter": "epo_ops",
        "access_tier": "credentialed_public_service",
        "license_summary": "EPO OPS terms, quotas, and fair-use controls apply to programmatic patent data.",
        "redistribution_policy": "metadata_and_links",
        "capabilities": ["bibliography", "families", "legal_status", "full_text", "images"],
    },
]


def upsert_dataset(session: Session, definition: dict) -> DatasetDefinition:
    item = session.scalar(
        select(DatasetDefinition).where(DatasetDefinition.code == definition["code"])
    )
    if item is None:
        item = DatasetDefinition(**definition)
        session.add(item)
    else:
        for key, value in definition.items():
            setattr(item, key, value)
    return item


def seed_dataset_registry(session: Session) -> int:
    for definition in CORE_DATASETS:
        upsert_dataset(session, definition)
    session.commit()
    return len(CORE_DATASETS)


async def discover_optimade_providers(session: Session, client: httpx.AsyncClient) -> int:
    response = await client.get("https://providers.optimade.org/v1/links")
    response.raise_for_status()
    discovered = 0
    for record in response.json().get("data", []):
        attributes = record.get("attributes", {})
        base_url = attributes.get("base_url")
        if not base_url:
            continue
        upsert_dataset(
            session,
            {
                "code": f"optimade-{record['id']}",
                "name": attributes.get("name") or record["id"],
                "category": "computed_materials",
                "base_url": base_url,
                "homepage_url": attributes.get("homepage"),
                "adapter": "optimade",
                "access_tier": "provider_specific",
                "license_summary": "Discovered through OPTIMADE; provider-specific terms govern reuse.",
                "redistribution_policy": "metadata_only_until_reviewed",
                "capabilities": ["structures", "composition", "provider_extensions"],
                "last_discovered_at": datetime.now(UTC),
            },
        )
        discovered += 1
    session.commit()
    return discovered
