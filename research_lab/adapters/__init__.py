from .crossref import CrossrefAdapter
from .datacite import DataCiteAdapter
from .openalex import OpenAlexAdapter

ADAPTERS = {
    "openalex": OpenAlexAdapter,
    "crossref": CrossrefAdapter,
    "datacite": DataCiteAdapter,
}

__all__ = ["ADAPTERS", "CrossrefAdapter", "DataCiteAdapter", "OpenAlexAdapter"]
