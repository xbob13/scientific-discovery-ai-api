import asyncio
from abc import ABC, abstractmethod

import httpx

from ..schemas import SourceRecord


class SourceAdapter(ABC):
    name: str
    base_url: str

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def get_json(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        max_attempts: int = 4,
    ) -> dict:
        for attempt in range(max_attempts):
            response = await self.client.get(url, params=params, headers=headers)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response.json()
            if attempt == max_attempts - 1:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 10.0) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    @abstractmethod
    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        raise NotImplementedError
