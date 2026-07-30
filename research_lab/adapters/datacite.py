from ..config import get_settings
from ..models import PublicationState
from ..schemas import SourceRecord
from .base import SourceAdapter
from .common import normalize_doi, parse_date


class DataCiteAdapter(SourceAdapter):
    """Research-dataset discovery through DataCite's public metadata API."""

    name = "datacite"
    base_url = "https://api.datacite.org"

    @staticmethod
    def _text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        settings = get_settings()
        payload = await self.get_json(
            f"{self.base_url}/dois",
            params={
                "query": query,
                "page[size]": limit,
                "resource-type-id": "dataset",
            },
            headers={
                "User-Agent": (
                    f"PattersonResearchLabs/0.2 (mailto:{settings.crossref_mailto})"
                )
            },
        )
        return [self._parse(item) for item in payload.get("data", [])]

    def _parse(self, item: dict) -> SourceRecord:
        attributes = item.get("attributes") or {}
        titles = [
            value for value in (attributes.get("titles") or []) if isinstance(value, dict)
        ]
        descriptions = [
            value
            for value in (attributes.get("descriptions") or [])
            if isinstance(value, dict)
        ]
        creators = [
            value for value in (attributes.get("creators") or []) if isinstance(value, dict)
        ]
        rights = [
            value for value in (attributes.get("rightsList") or []) if isinstance(value, dict)
        ]
        doi = str(attributes.get("doi") or item["id"])
        published = attributes.get("published")
        source_url = self._text(attributes.get("url")) or f"https://doi.org/{doi}"
        return SourceRecord(
            source_name=self.name,
            endpoint=f"{self.base_url}/dois",
            external_id=item["id"],
            title=(
                self._text(titles[0].get("title")) if titles else None
            )
            or "[untitled dataset]",
            authors=[
                self._text(creator.get("name"))
                or " ".join(
                    part
                    for part in (
                        self._text(creator.get("givenName")),
                        self._text(creator.get("familyName")),
                    )
                    if part is not None
                )
                for creator in creators
                if creator.get("name") or creator.get("familyName")
            ],
            abstract=next(
                (
                    self._text(description.get("description"))
                    for description in descriptions
                    if self._text(description.get("description"))
                ),
                None,
            ),
            doi=normalize_doi(doi),
            published_at=parse_date(str(published) if published is not None else None),
            publication_state=PublicationState.PUBLISHED,
            peer_reviewed=False,
            source_url=source_url,
            license=self._text(rights[0].get("rightsUri")) if rights else None,
            raw=item,
        )
