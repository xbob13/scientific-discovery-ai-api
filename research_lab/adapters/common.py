import re
from datetime import UTC, datetime


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(".,;)") or None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def abstract_from_inverted_index(index: dict | None) -> str | None:
    if not index:
        return None
    positioned = sorted((position, word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in positioned)
