from abc import ABC, abstractmethod

import httpx

from ..schemas import SourceRecord


class SourceAdapter(ABC):
    name: str
    base_url: str

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abstractmethod
    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        raise NotImplementedError
