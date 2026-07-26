import re

from ..config import get_settings
from ..models import PublicationState
from ..schemas import SourceRecord
from .base import SourceAdapter
from .common import normalize_doi, parse_date


class CrossrefAdapter(SourceAdapter):
    name = "crossref"
    base_url = "https://api.crossref.org"

    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        settings = get_settings()
        headers = {"User-Agent": f"PattersonResearchLabs/0.1 (mailto:{settings.crossref_mailto})"}
        response = await self.client.get(
            f"{self.base_url}/works", params={"query": query, "rows": limit}, headers=headers
        )
        response.raise_for_status()
        return [self._parse(item) for item in response.json()["message"].get("items", [])]

    def _parse(self, item: dict) -> SourceRecord:
        authors = [
            " ".join(part for part in (author.get("given"), author.get("family")) if part)
            for author in item.get("author", [])
        ]
        abstract = item.get("abstract")
        if abstract:
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
        published = item.get("published", {}).get("date-parts", [[]])[0]
        date = "-".join(str(x).zfill(2) for x in published) if published else None
        return SourceRecord(
            source_name=self.name,
            endpoint=f"{self.base_url}/works",
            external_id=item["DOI"],
            title=(item.get("title") or ["[untitled]"])[0],
            authors=authors,
            abstract=abstract,
            doi=normalize_doi(item.get("DOI")),
            published_at=parse_date(date),
            publication_state=PublicationState.PUBLISHED,
            peer_reviewed=item.get("subtype") != "preprint",
            source_url=item.get("URL") or f"https://doi.org/{item['DOI']}",
            license=(item.get("license") or [{}])[0].get("URL"),
            raw=item,
        )
