from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from .models import PublicationState


class SourceRecord(BaseModel):
    source_name: str
    endpoint: str
    external_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    published_at: datetime | None = None
    publication_state: PublicationState
    peer_reviewed: bool
    version: str = "1"
    source_url: HttpUrl
    license: str | None = None
    raw: dict


class HarvestRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    sources: list[str] = Field(default_factory=lambda: ["openalex", "crossref"])
    limit_per_source: int = Field(default=10, ge=1, le=100)


class ComputeRequest(BaseModel):
    conductivity_s_m: float = Field(gt=0)
    thickness_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    length_m: float = Field(gt=0)
    conductivity_relative_uncertainty: float = Field(default=0.05, ge=0, le=0.5)
    samples: int = Field(default=10_000, ge=100, le=100_000)
    seed: int = 42


class ClientWorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=240)
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", min_length=3, max_length=120)
    portal_token: str = Field(min_length=24, max_length=500)


class ClientTopicCreate(BaseModel):
    name: str = Field(min_length=2, max_length=240)
    research_question: str = Field(min_length=10, max_length=2000)
    keywords: list[str] = Field(min_length=1, max_length=50)


class SubscriptionActivationRequest(BaseModel):
    provider: str = Field(default="stripe", pattern=r"^[a-z0-9_-]+$", max_length=40)
    external_session_id: str = Field(min_length=8, max_length=255)
    external_customer_id: str | None = Field(default=None, max_length=255)
    external_subscription_id: str | None = Field(default=None, max_length=255)
    subscription_status: str = Field(default="active", max_length=40)
    current_period_end: datetime | None = None
    plan: str = Field(pattern=r"^[a-z0-9-]+$", max_length=80)
    contact_email: str = Field(min_length=5, max_length=320)
    organization: str = Field(min_length=2, max_length=240)
    program_slug: str = Field(default="custom", pattern=r"^[a-z0-9-]+$", max_length=120)
    research_question: str = Field(min_length=30, max_length=4000)


class ClientMandateCreate(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    question: str = Field(min_length=30, max_length=4000)
    sources: list[str] = Field(default_factory=lambda: ["openalex", "crossref", "datacite"])
    requested_outputs: list[str] = Field(default_factory=lambda: ["executive_brief", "evidence_ledger"])
    constraints: dict = Field(default_factory=dict)
    external_reference: str | None = Field(default=None, max_length=240)


class PatentSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    providers: list[str] = Field(default_factory=lambda: ["uspto_odp", "epo_ops"])
    limit_per_provider: int = Field(default=10, ge=1, le=100)


class ProspectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=240)
    domain: str | None = Field(default=None, max_length=300)
    target_type: str = Field(default="company", max_length=80)
    evidence_signals: list[dict] = Field(min_length=1, max_length=30)
    contact_name: str | None = Field(default=None, max_length=240)
    contact_email: str | None = Field(default=None, max_length=320)
    contact_basis: str | None = Field(default=None, max_length=240)
    source_url: HttpUrl | None = None


class OutreachDraftCreate(BaseModel):
    prospect_id: str = Field(min_length=8, max_length=64)
    program_slug: str = Field(default="custom", max_length=120)
    value_proposition: str = Field(min_length=20, max_length=1200)


class CommerceLifecycleEventRequest(BaseModel):
    provider: str = Field(default="stripe", pattern=r"^[a-z0-9_-]+$", max_length=40)
    external_event_id: str = Field(min_length=8, max_length=255)
    event_type: str = Field(min_length=3, max_length=120)
    external_customer_id: str | None = Field(default=None, max_length=255)
    external_subscription_id: str | None = Field(default=None, max_length=255)
    external_session_id: str | None = Field(default=None, max_length=255)
    event_created_at: datetime | None = None
    status: str | None = Field(default=None, max_length=40)


class InstitutionDiscoveryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=20, ge=1, le=50)
    seed_verified_channels: bool = True
