import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class PublicationState(str, enum.Enum):
    PREPRINT = "preprint"
    PUBLISHED = "published"
    CORRECTED = "corrected"
    RETRACTED = "retracted"


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    base_url: Mapped[str] = mapped_column(String(500))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class SourceCursor(Base):
    __tablename__ = "source_cursors"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    scope: Mapped[str] = mapped_column(String(200), default="default")
    cursor: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("source_id", "scope"),)


class RawSourceSnapshot(Base):
    __tablename__ = "raw_source_snapshots"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    external_id: Mapped[str] = mapped_column(String(500))
    endpoint: Mapped[str] = mapped_column(String(1000))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    parser_version: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("source_id", "external_id", "checksum_sha256"),)


class Work(Base):
    __tablename__ = "works"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(Text)
    normalized_title: Mapped[str] = mapped_column(Text, index=True)
    work_type: Mapped[str] = mapped_column(String(80), default="article")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    versions: Mapped[list["WorkVersion"]] = relationship(back_populates="work")
    identifiers: Mapped[list["Identifier"]] = relationship(back_populates="work")


class WorkVersion(Base):
    __tablename__ = "work_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    work_id: Mapped[str] = mapped_column(ForeignKey("works.id"))
    source_snapshot_id: Mapped[str] = mapped_column(ForeignKey("raw_source_snapshots.id"))
    version_label: Mapped[str] = mapped_column(String(120), default="1")
    publication_state: Mapped[PublicationState] = mapped_column(Enum(PublicationState))
    peer_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    abstract: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(String(1000))
    license: Mapped[str | None] = mapped_column(String(200))
    work: Mapped[Work] = relationship(back_populates="versions")
    __table_args__ = (UniqueConstraint("source_snapshot_id"),)


class Identifier(Base):
    __tablename__ = "identifiers"
    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[str] = mapped_column(ForeignKey("works.id"))
    scheme: Mapped[str] = mapped_column(String(30))
    normalized_value: Mapped[str] = mapped_column(String(500))
    raw_value: Mapped[str] = mapped_column(String(500))
    work: Mapped[Work] = relationship(back_populates="identifiers")
    __table_args__ = (UniqueConstraint("scheme", "normalized_value"),)


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    work_version_id: Mapped[str] = mapped_column(ForeignKey("work_versions.id"))
    text: Mapped[str] = mapped_column(Text)
    locator: Mapped[str] = mapped_column(String(500))
    supporting_excerpt: Mapped[str | None] = mapped_column(Text)
    extraction_method: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str | None] = mapped_column(String(120))
    confidence: Mapped[float] = mapped_column(Float)
    human_review_status: Mapped[str] = mapped_column(String(40), default="pending")
    evidence_features: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_score: Mapped[float | None] = mapped_column(Float)


class ConnectionCandidate(Base):
    __tablename__ = "connection_candidates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(Text)
    source_work_ids: Mapped[list] = mapped_column(JSON)
    concepts: Mapped[list] = mapped_column(JSON)
    bridge: Mapped[str] = mapped_column(Text)
    assumptions: Mapped[list] = mapped_column(JSON)
    falsifiable_prediction: Mapped[str] = mapped_column(Text)
    validation_method: Mapped[str] = mapped_column(Text)
    reasons_to_reject: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(40), default="pending_review")


class CandidateAssessment(Base):
    """Versioned, inspectable assessment for a machine-proposed connection."""

    __tablename__ = "candidate_assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    connection_id: Mapped[str] = mapped_column(ForeignKey("connection_candidates.id"), index=True)
    assessment_version: Mapped[str] = mapped_column(String(40), default="evidence-v1")
    score: Mapped[float] = mapped_column(Float)
    score_components: Mapped[dict] = mapped_column(JSON)
    source_evidence: Mapped[list] = mapped_column(JSON)
    novelty_status: Mapped[str] = mapped_column(String(40), default="unassessed")
    novelty_queries: Mapped[list] = mapped_column(JSON, default=list)
    corroboration: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("connection_id", "assessment_version"),)


class ComputeRun(Base):
    __tablename__ = "compute_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_type: Mapped[str] = mapped_column(String(120))
    inputs: Mapped[dict] = mapped_column(JSON)
    code_sha256: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict] = mapped_column(JSON)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    reproducible: Mapped[bool] = mapped_column(Boolean)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class JobRun(Base):
    __tablename__ = "job_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type: Mapped[str] = mapped_column(String(120))
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    status: Mapped[str] = mapped_column(String(40), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    input: Mapped[dict] = mapped_column(JSON)
    output: Mapped[dict | None] = mapped_column(JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
