from ..config import get_settings
from ..models import PublicationState
from ..schemas import SourceRecord
from .base import SourceAdapter
from .common import abstract_from_inverted_index, normalize_doi, parse_date


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"
    base_url = "https://api.openalex.org"

    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        settings = get_settings()
        params = {"search": query, "per-page": limit, "mailto": settings.crossref_mailto}
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key.get_secret_value()
        payload = await self.get_json(f"{self.base_url}/works", params=params)
        return [self._parse(item) for item in payload.get("results", [])]

    def _parse(self, item: dict) -> SourceRecord:
        doi = normalize_doi(item.get("doi"))
        is_preprint = item.get("type") == "preprint"
        return SourceRecord(
            source_name=self.name,
            endpoint=f"{self.base_url}/works",
            external_id=item["id"],
            title=item.get("display_name") or "[untitled]",
            authors=[a["author"]["display_name"] for a in item.get("authorships", []) if a.get("author")],
            abstract=abstract_from_inverted_index(item.get("abstract_inverted_index")),
            doi=doi,
            published_at=parse_date(item.get("publication_date")),
            publication_state=PublicationState.PREPRINT if is_preprint else PublicationState.PUBLISHED,
            peer_reviewed=not is_preprint,
            source_url=item.get("doi") or item["id"],
            license=(item.get("best_oa_location") or {}).get("license"),
            raw=item,
        )
