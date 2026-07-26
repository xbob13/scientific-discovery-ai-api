from .crossref import CrossrefAdapter
from .openalex import OpenAlexAdapter

ADAPTERS = {"openalex": OpenAlexAdapter, "crossref": CrossrefAdapter}

__all__ = ["ADAPTERS", "CrossrefAdapter", "OpenAlexAdapter"]
