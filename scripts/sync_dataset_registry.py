import asyncio
import json

import httpx

from research_lab.datasets import discover_optimade_providers, seed_dataset_registry
from research_lab.db import Base, SessionLocal, engine


async def sync() -> dict:
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        core = seed_dataset_registry(session)
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                optimade = await discover_optimade_providers(session, client)
                optimade_error = None
            except Exception as exc:
                session.rollback()
                optimade = 0
                optimade_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {
        "core_datasets": core,
        "optimade_providers": optimade,
        "optimade_error": optimade_error,
    }


def main() -> None:
    print(json.dumps(asyncio.run(sync()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
