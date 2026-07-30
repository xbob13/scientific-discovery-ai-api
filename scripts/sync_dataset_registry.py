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
            optimade = await discover_optimade_providers(session, client)
    return {"core_datasets": core, "optimade_providers": optimade}


def main() -> None:
    print(json.dumps(asyncio.run(sync()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
