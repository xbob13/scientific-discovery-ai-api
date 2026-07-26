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
