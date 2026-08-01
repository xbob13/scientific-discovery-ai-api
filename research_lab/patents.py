from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from xml.etree import ElementTree

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import PatentDocument


@dataclass
class PatentHit:
    provider: str
    publication_number: str
    title: str
    source_url: str
    application_number: str | None = None
    family_id: str | None = None
    jurisdiction: str | None = None
    abstract: str | None = None
    claims_excerpt: str | None = None
    inventors: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    cpc_codes: list[str] = field(default_factory=list)
    ipc_codes: list[str] = field(default_factory=list)
    priority_date: str | None = None
    filing_date: str | None = None
    publication_date: str | None = None
    legal_status: str | None = None
    raw: dict = field(default_factory=dict)


class PatentsViewClient:
    """Legacy adapter retained only for previously issued PatentsView credentials."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.settings = get_settings()

    @property
    def configured(self) -> bool:
        return self.settings.patentsview_api_key is not None

    async def search(self, query: str, limit: int) -> list[PatentHit]:
        if not self.configured:
            raise RuntimeError("PatentsView API key is not configured")
        expression = json.dumps({"_text_any": {"patent_title": query}})
        fields = [
            "patent_id",
            "patent_title",
            "patent_abstract",
            "patent_date",
            "patent_type",
            "inventor_name_first",
            "inventor_name_last",
            "assignee_organization",
            "cpc_group_id",
        ]
        response = await self.client.get(
            self.settings.patentsview_search_url,
            params={"q": expression, "f": json.dumps(fields), "o": json.dumps({"size": limit})},
            headers={
                "X-Api-Key": self.settings.patentsview_api_key.get_secret_value(),
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("patents") or payload.get("data") or payload.get("results") or []
        return [self._normalize(record) for record in records[:limit] if self._number(record)]

    @staticmethod
    def _number(record: dict) -> str:
        return str(
            record.get("patent_id") or record.get("patent_number") or record.get("publication_number") or ""
        ).strip()

    def _normalize(self, record: dict) -> PatentHit:
        number = self._number(record)
        inventors = _names(record.get("inventors")) or _joined_names(record, "inventor")
        assignees = _names(record.get("assignees")) or _values(record, "assignee_organization")
        cpc_codes = _values(record, "cpc_group_id") + _values(record, "cpc_subgroup_id")
        return PatentHit(
            provider="patentsview",
            publication_number=number,
            application_number=_text(record, "application_number", "app_number"),
            jurisdiction="US",
            title=_text(record, "patent_title", "title") or f"US patent {number}",
            abstract=_text(record, "patent_abstract", "abstract"),
            claims_excerpt=_truncate(_text(record, "claims", "claim_text", "patent_claims"), 4000),
            inventors=sorted(set(inventors)),
            assignees=sorted(set(assignees)),
            cpc_codes=sorted(set(cpc_codes)),
            priority_date=_text(record, "priority_date"),
            filing_date=_text(record, "filing_date", "application_date"),
            publication_date=_text(record, "patent_date", "publication_date"),
            legal_status=_text(record, "patent_type", "legal_status"),
            source_url=f"https://patents.google.com/patent/US{number}",
            raw=record,
        )


class UsptoOdpClient:
    """Live U.S. application metadata from USPTO's Patent File Wrapper API."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.settings = get_settings()

    @property
    def configured(self) -> bool:
        return self.settings.uspto_odp_api_key is not None

    async def search(self, query: str, limit: int) -> list[PatentHit]:
        if not self.configured:
            raise RuntimeError("USPTO Open Data Portal API key is not configured")
        response = await self.client.get(
            self.settings.uspto_odp_search_url,
            params={"q": query, "start": 0, "rows": min(limit, 25)},
            headers={
                "X-API-KEY": self.settings.uspto_odp_api_key.get_secret_value(),
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        records = (
            payload.get("patentFileWrapperDataBag") or payload.get("results") or payload.get("data") or []
        )
        return [self._normalize(record) for record in records[:limit] if isinstance(record, dict)]

    def _normalize(self, record: dict) -> PatentHit:
        metadata = record.get("applicationMetaData") or record.get("applicationMetadata") or {}
        application = _text(record, "applicationNumberText", "applicationNumber") or _text(
            metadata, "applicationNumberText", "applicationNumber"
        )
        publication = (
            _text(
                metadata,
                "earliestPublicationNumber",
                "publicationNumber",
                "patentNumber",
            )
            or _text(record, "publicationNumber", "patentNumber")
            or application
            or "unknown"
        )
        title = _text(metadata, "inventionTitle") or _text(record, "inventionTitle", "title")
        inventors = _party_names(metadata, "inventorBag", "inventor")
        applicants = _party_names(metadata, "applicantBag", "applicant")
        cpc_codes = _classification_codes(metadata, "cpcClassificationBag")
        source_key = re.sub(r"[^A-Za-z0-9]", "", application or publication)
        return PatentHit(
            provider="uspto_odp",
            publication_number=publication,
            application_number=application,
            jurisdiction="US",
            title=title or f"U.S. patent application {publication}",
            abstract=_text(metadata, "abstractText", "abstract") or _text(record, "abstractText"),
            inventors=inventors,
            assignees=applicants,
            cpc_codes=cpc_codes,
            filing_date=_text(metadata, "filingDate", "applicationFiledDate"),
            publication_date=_text(metadata, "publicationDate", "earliestPublicationDate"),
            legal_status=_text(
                metadata,
                "applicationStatusDescriptionText",
                "applicationStatusDescription",
            ),
            source_url=f"https://patentcenter.uspto.gov/applications/{source_key}",
            raw=record,
        )


class EpoOpsClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.settings = get_settings()

    @property
    def configured(self) -> bool:
        return bool(self.settings.epo_ops_consumer_key and self.settings.epo_ops_consumer_secret)

    async def search(self, query: str, limit: int) -> list[PatentHit]:
        if not self.configured:
            raise RuntimeError("EPO OPS credentials are not configured")
        key = self.settings.epo_ops_consumer_key.get_secret_value()
        secret = self.settings.epo_ops_consumer_secret.get_secret_value()
        basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
        token_response = await self.client.post(
            self.settings.epo_ops_auth_url,
            content="grant_type=client_credentials",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        token_response.raise_for_status()
        token = token_response.json()["access_token"]
        response = await self.client.get(
            self.settings.epo_ops_search_url,
            params={"q": f'ta all "{_ops_query(query)}"'},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/xml",
                "X-OPS-Range": f"1-{min(limit, 100)}",
            },
        )
        response.raise_for_status()
        hits = _parse_epo_results(response.text, limit)
        for hit in hits[: max(0, min(self.settings.epo_ops_claims_per_search, len(hits)))]:
            try:
                claims_response = await self.client.get(
                    (
                        f"{self.settings.epo_ops_published_data_url}/publication/epodoc/"
                        f"{hit.publication_number}/claims"
                    ),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/fulltext+xml",
                    },
                )
                if claims_response.status_code == 200:
                    hit.claims_excerpt = _truncate(_extract_claims(claims_response.text), 12_000)
                    hit.raw["claims_xml"] = claims_response.text[:50_000]
            except httpx.HTTPError:
                continue
        return hits


async def search_patents(
    session: Session,
    query: str,
    providers: list[str],
    limit_per_provider: int = 10,
    client: httpx.AsyncClient | None = None,
) -> dict:
    unknown = set(providers) - {"uspto_odp", "patentsview", "epo_ops"}
    if unknown:
        raise ValueError(f"unsupported patent providers: {sorted(unknown)}")
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30, follow_redirects=True, trust_env=False)
    assert client is not None
    records: list[dict] = []
    failures: dict[str, str] = {}
    clients = {
        "uspto_odp": UsptoOdpClient,
        "patentsview": PatentsViewClient,
        "epo_ops": EpoOpsClient,
    }
    try:
        for provider in providers:
            try:
                hits = await clients[provider](client).search(query, limit_per_provider)
                for hit in hits:
                    document = upsert_patent(session, hit)
                    records.append(patent_view(document))
                session.commit()
            except Exception as exc:
                session.rollback()
                failures[provider] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if own_client:
            await client.aclose()
    return {
        "query": query,
        "records": records,
        "landscape": patent_landscape(records, query),
        "provider_failures": failures,
        "disclaimer": (
            "Patent and claim excerpts support technical research triage. This is not a "
            "patentability, legal-status, or freedom-to-operate opinion."
        ),
    }


def upsert_patent(session: Session, hit: PatentHit) -> PatentDocument:
    existing = session.scalar(
        select(PatentDocument).where(
            PatentDocument.provider == hit.provider,
            PatentDocument.publication_number == hit.publication_number,
        )
    )
    checksum = hashlib.sha256(json.dumps(hit.raw, sort_keys=True, default=str).encode()).hexdigest()
    values = {
        key: value for key, value in hit.__dict__.items() if key not in {"provider", "publication_number"}
    }
    values["payload_sha256"] = checksum
    if existing:
        for key, value in values.items():
            setattr(existing, key, value)
        return existing
    document = PatentDocument(
        provider=hit.provider,
        publication_number=hit.publication_number,
        **values,
    )
    session.add(document)
    session.flush()
    return document


def patent_view(document: PatentDocument) -> dict:
    return {
        "id": document.id,
        "provider": document.provider,
        "publication_number": document.publication_number,
        "application_number": document.application_number,
        "family_id": document.family_id,
        "jurisdiction": document.jurisdiction,
        "title": document.title,
        "abstract": document.abstract,
        "claims_excerpt": document.claims_excerpt,
        "inventors": document.inventors,
        "assignees": document.assignees,
        "cpc_codes": document.cpc_codes,
        "ipc_codes": document.ipc_codes,
        "priority_date": document.priority_date,
        "filing_date": document.filing_date,
        "publication_date": document.publication_date,
        "legal_status": document.legal_status,
        "source_url": document.source_url,
        "retrieved_at": document.retrieved_at,
    }


def patent_landscape(records: list[dict], query: str) -> dict:
    """Build explainable portfolio facets from normalized patent metadata."""
    assignees = Counter(
        assignee for record in records for assignee in record.get("assignees", []) if assignee
    )
    classifications = Counter(
        code
        for record in records
        for code in [*record.get("cpc_codes", []), *record.get("ipc_codes", [])]
        if code
    )
    years = Counter(
        str(record.get("publication_date") or "")[:4]
        for record in records
        if re.match(r"^\d{4}", str(record.get("publication_date") or ""))
    )
    query_terms = {
        word
        for word in re.findall(r"[a-z][a-z0-9+-]{2,}", query.lower())
        if word not in {"and", "for", "the", "with", "from", "using"}
    }
    ranked = []
    for record in records:
        text = " ".join(
            str(record.get(field) or "") for field in ("title", "abstract", "claims_excerpt")
        ).lower()
        matched = sorted(term for term in query_terms if term in text)
        ranked.append(
            {
                "publication_number": record.get("publication_number"),
                "title": record.get("title"),
                "matched_terms": matched,
                "lexical_relevance": round(len(matched) / max(1, len(query_terms)), 3),
                "source_url": record.get("source_url"),
            }
        )
    ranked.sort(key=lambda item: (-item["lexical_relevance"], str(item["publication_number"])))
    synthesis = synthesize_patent_evidence(records, query)
    return {
        "record_count": len(records),
        "top_assignees": [{"name": name, "records": count} for name, count in assignees.most_common(10)],
        "top_classifications": [
            {"code": code, "records": count} for code, count in classifications.most_common(15)
        ],
        "publication_years": dict(sorted(years.items())),
        "ranked_records": ranked[:25],
        **synthesis,
        "method": (
            "Normalized metadata facets, exact source excerpts, and transparent lexical coverage. "
            "Prior-art pressure is a research-screening heuristic, never a novelty conclusion."
        ),
    }


def synthesize_patent_evidence(
    records: list[dict],
    query: str,
    literature_candidates: list[dict] | None = None,
) -> dict:
    """Turn source-linked patent text into bounded technical research signals."""
    query_terms = _meaningful_terms(query)
    passages: list[dict] = []
    signal_records: dict[tuple[str, str], set[str]] = {}
    vocabularies = {
        "material": {
            "polymer",
            "ceramic",
            "metal",
            "alloy",
            "oxide",
            "graphene",
            "membrane",
            "electrolyte",
            "catalyst",
            "composite",
            "hydrogel",
            "nanoparticle",
        },
        "process": {
            "coat",
            "coating",
            "deposit",
            "deposition",
            "anneal",
            "sinter",
            "etch",
            "crosslink",
            "functionalize",
            "fabricate",
            "extrude",
            "calcine",
        },
        "property": {
            "conductivity",
            "permeability",
            "selectivity",
            "durability",
            "fouling",
            "corrosion",
            "adhesion",
            "porosity",
            "thermal",
            "strength",
            "stability",
        },
    }
    with_claims = 0
    strong_records = 0
    for record in records:
        source_text = str(record.get("claims_excerpt") or record.get("abstract") or "").strip()
        locator = "claims" if record.get("claims_excerpt") else "abstract"
        if record.get("claims_excerpt"):
            with_claims += 1
        terms = _meaningful_terms(
            " ".join(str(record.get(field) or "") for field in ("title", "abstract", "claims_excerpt"))
        )
        matched = sorted(query_terms & terms)
        if len(matched) >= max(1, min(3, len(query_terms))):
            strong_records += 1
        excerpt = _best_excerpt(source_text, query_terms)
        if excerpt:
            passages.append(
                {
                    "provider": record.get("provider"),
                    "publication_number": record.get("publication_number"),
                    "locator": locator,
                    "text": excerpt,
                    "matched_terms": matched,
                    "source_url": record.get("source_url"),
                }
            )
        publication = str(record.get("publication_number") or "")
        for category, vocabulary in vocabularies.items():
            for signal in sorted(terms & vocabulary):
                signal_records.setdefault((category, signal), set()).add(publication)

    technical_signals = [
        {
            "category": category,
            "signal": signal,
            "record_count": len(publications),
            "publications": sorted(publications)[:10],
        }
        for (category, signal), publications in signal_records.items()
    ]
    technical_signals.sort(key=lambda item: (-item["record_count"], item["category"], item["signal"]))
    pressure_score = min(
        100,
        round(
            45 * strong_records / max(1, len(records))
            + 35 * with_claims / max(1, len(records))
            + min(20, len(records) * 2)
        ),
    )
    pressure_level = "high" if pressure_score >= 70 else "moderate" if pressure_score >= 40 else "low"
    return {
        "claim_coverage": {"records_with_claims": with_claims, "total_records": len(records)},
        "evidence_passages": passages[:25],
        "technical_signals": technical_signals[:30],
        "literature_bridges": _literature_bridges(records, literature_candidates or []),
        "prior_art_pressure": {
            "level": pressure_level,
            "score": pressure_score,
            "basis": (
                f"{strong_records} records matched multiple query terms; "
                f"{with_claims} included retrievable claim text."
            ),
            "caveat": (
                "Screening signal only. A qualified patent professional must determine novelty, "
                "claim scope, legal status, or freedom to operate."
            ),
        },
    }


def _literature_bridges(records: list[dict], candidates: list[dict]) -> list[dict]:
    bridges: list[dict] = []
    for candidate in candidates[:25]:
        candidate_text = " ".join(
            [
                str(candidate.get("title") or ""),
                str(candidate.get("bridge") or ""),
                " ".join(str(item) for item in candidate.get("concepts", [])),
            ]
        )
        candidate_terms = _meaningful_terms(candidate_text)
        for record in records[:50]:
            patent_terms = _meaningful_terms(
                " ".join(str(record.get(field) or "") for field in ("title", "abstract", "claims_excerpt"))
            )
            shared = sorted(candidate_terms & patent_terms)
            if len(shared) < 2:
                continue
            bridges.append(
                {
                    "candidate_id": candidate.get("id"),
                    "candidate_title": candidate.get("title"),
                    "publication_number": record.get("publication_number"),
                    "shared_concepts": shared[:12],
                    "source_url": record.get("source_url"),
                    "interpretation": (
                        "The literature proposal and patent record share technical concepts; "
                        "inspect the cited source passages before treating the relationship as prior art."
                    ),
                }
            )
    bridges.sort(key=lambda item: (-len(item["shared_concepts"]), str(item["publication_number"])))
    return bridges[:25]


def _meaningful_terms(value: str) -> set[str]:
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "into",
        "that",
        "the",
        "their",
        "this",
        "using",
        "with",
        "which",
        "where",
        "method",
        "system",
        "comprising",
        "including",
    }
    return {term for term in re.findall(r"[a-z][a-z0-9+-]{2,}", value.lower()) if term not in stopwords}


def _best_excerpt(value: str, query_terms: set[str], limit: int = 900) -> str | None:
    if not value:
        return None
    segments = [segment.strip() for segment in re.split(r"(?<=[.;])\s+|\n+", value) if segment.strip()]
    if not segments:
        return _truncate(value.strip(), limit)
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (-len(_meaningful_terms(item[1]) & query_terms), item[0]),
    )
    best_index = ranked[0][0]
    excerpt = " ".join(segments[best_index : best_index + 2])
    return _truncate(excerpt, limit)


def _parse_epo_results(xml: str, limit: int) -> list[PatentHit]:
    root = ElementTree.fromstring(xml)
    hits: list[PatentHit] = []
    for node in root.iter():
        if _local(node.tag) not in {"exchange-document", "publication-reference"}:
            continue
        number = node.attrib.get("doc-number") or _descendant_text(node, "doc-number")
        if not number or any(hit.publication_number == number for hit in hits):
            continue
        country = node.attrib.get("country") or _descendant_text(node, "country")
        kind = node.attrib.get("kind") or _descendant_text(node, "kind")
        publication = f"{country or ''}{number}{kind or ''}"
        title = _descendant_text(node, "invention-title") or f"Patent publication {publication}"
        hits.append(
            PatentHit(
                provider="epo_ops",
                publication_number=publication,
                jurisdiction=country,
                title=re.sub(r"\s+", " ", title).strip(),
                abstract=_descendant_text(node, "abstract"),
                claims_excerpt=_truncate(_descendant_text(node, "claims"), 4000),
                publication_date=_descendant_text(node, "date"),
                source_url=f"https://worldwide.espacenet.com/patent/search?q=pn%3D{publication}",
                raw={"xml": ElementTree.tostring(node, encoding="unicode")[:50_000]},
            )
        )
        if len(hits) >= limit:
            break
    return hits


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _descendant_text(node: ElementTree.Element, name: str) -> str | None:
    for child in node.iter():
        if _local(child.tag) == name:
            text = " ".join(part.strip() for part in child.itertext() if part.strip())
            if text:
                return text
    return None


def _extract_claims(xml: str) -> str | None:
    root = ElementTree.fromstring(xml)
    claim_parts = []
    for node in root.iter():
        if _local(node.tag) in {"claim", "claim-text"}:
            text = " ".join(part.strip() for part in node.itertext() if part.strip())
            if text and text not in claim_parts:
                claim_parts.append(text)
    return "\n".join(claim_parts) or _descendant_text(root, "claims")


def _ops_query(query: str) -> str:
    return re.sub(r"[^a-zA-Z0-9+\-/ ]", " ", query)[:400].strip()


def _text(record: dict, *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            value = value.get("value") or value.get("name")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _truncate(value: str | None, limit: int) -> str | None:
    return value[:limit] if value else None


def _values(record: dict, key: str) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get(key) or item.get("name") or item.get("value")
        if item is not None and str(item).strip():
            result.append(str(item).strip())
    return result


def _names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, dict):
            name = (
                item.get("name")
                or " ".join(str(item.get(key, "")).strip() for key in ("first_name", "last_name")).strip()
            )
            if name:
                names.append(name)
        elif item:
            names.append(str(item))
    return names


def _joined_names(record: dict, prefix: str) -> list[str]:
    first = _values(record, f"{prefix}_name_first")
    last = _values(record, f"{prefix}_name_last")
    return [" ".join(parts).strip() for parts in zip(first, last, strict=False) if any(parts)]


def _party_names(record: dict, bag_key: str, prefix: str) -> list[str]:
    value = record.get(bag_key) or []
    if isinstance(value, dict):
        value = value.get(f"{prefix}Bag") or value.get(prefix) or value.get("items") or [value]
    if not isinstance(value, list):
        value = [value]
    names = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _text(
            item,
            f"{prefix}NameText",
            "nameText",
            "organizationName",
            "personName",
        )
        if not name:
            name = " ".join(
                filter(
                    None,
                    [
                        _text(item, "givenName", "firstName"),
                        _text(item, "familyName", "lastName"),
                    ],
                )
            )
        if name:
            names.append(name)
    return sorted(set(names))


def _classification_codes(record: dict, key: str) -> list[str]:
    value = record.get(key) or []
    if isinstance(value, dict):
        value = value.get("cpcClassification") or value.get("items") or [value]
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        if isinstance(item, dict):
            code = _text(item, "classificationSymbolText", "symbol", "code")
        else:
            code = str(item).strip() if item else None
        if code:
            result.append(code)
    return sorted(set(result))
