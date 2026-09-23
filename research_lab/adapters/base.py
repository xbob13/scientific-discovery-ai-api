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
        retryable_statuses = {429, 500, 502, 503, 504}
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(url, params=params, headers=headers)
            except httpx.RequestError:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(min(2**attempt, 10.0))
                continue
            if response.status_code not in retryable_statuses:
                response.raise_for_status()
                return response.json()
            if attempt == max_attempts - 1:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 10.0) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            await asyncio.sleep(min(delay, 10.0))
        raise RuntimeError("unreachable")

    @abstractmethod
    async def harvest(self, query: str, limit: int) -> list[SourceRecord]:
        raise NotImplementedError
