import hashlib
import json
import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.common import normalize_doi
from .models import Identifier, RawSourceSnapshot, Source, Work, WorkVersion
from .schemas import SourceRecord

PARSER_VERSION = "2026-07-26.1"


def normalize_title(title: str) -> str:
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def store_record(session: Session, record: SourceRecord) -> tuple[Work, bool]:
    source = session.scalar(select(Source).where(Source.name == record.source_name))
    if not source:
        source = Source(name=record.source_name, base_url=record.endpoint.rsplit("/", 1)[0])
        session.add(source)
        session.flush()

    canonical_payload = json.dumps(record.raw, sort_keys=True, separators=(",", ":")).encode()
    checksum = hashlib.sha256(canonical_payload).hexdigest()
    snapshot = session.scalar(
        select(RawSourceSnapshot).where(
            RawSourceSnapshot.source_id == source.id,
            RawSourceSnapshot.external_id == record.external_id,
            RawSourceSnapshot.checksum_sha256 == checksum,
        )
    )
    if snapshot:
        version = session.scalar(select(WorkVersion).where(WorkVersion.source_snapshot_id == snapshot.id))
        return session.get(Work, version.work_id), False

    snapshot = RawSourceSnapshot(
        source_id=source.id,
        external_id=record.external_id,
        endpoint=record.endpoint,
        checksum_sha256=checksum,
        parser_version=PARSER_VERSION,
        payload=record.raw,
    )
    session.add(snapshot)
    session.flush()

    doi = normalize_doi(record.doi)
    work = None
    if doi:
        identifier = session.scalar(
            select(Identifier).where(Identifier.scheme == "doi", Identifier.normalized_value == doi)
        )
        work = identifier.work if identifier else None
    if not work:
        title_key = normalize_title(record.title)
        work = session.scalar(select(Work).where(Work.normalized_title == title_key))
    if not work:
        work = Work(title=record.title, normalized_title=normalize_title(record.title))
        session.add(work)
        session.flush()
        if doi:
            session.add(Identifier(work_id=work.id, scheme="doi", normalized_value=doi, raw_value=record.doi))

    session.add(
        WorkVersion(
            work_id=work.id,
            source_snapshot_id=snapshot.id,
            version_label=record.version,
            publication_state=record.publication_state,
            peer_reviewed=record.peer_reviewed,
            published_at=record.published_at,
            abstract=record.abstract,
            source_url=str(record.source_url),
            license=record.license,
        )
    )
    session.flush()
    return work, True
