import httpx
import pytest

from research_lab.adapters.base import SourceAdapter


class StubAdapter(SourceAdapter):
    name = "test"
    base_url = "https://example.test"

    async def harvest(self, query: str, limit: int):
        return []


@pytest.mark.asyncio
async def test_retryable_rate_limit_recovers(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        payload = await StubAdapter(client).get_json("https://example.test/data")
    assert payload == {"ok": True}
    assert attempts == 2


@pytest.mark.asyncio
async def test_non_retryable_failure_is_immediate():
    attempts = 0

    def handler(request: httpx.Request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await StubAdapter(client).get_json("https://example.test/data")
    assert attempts == 1
