import argparse
import asyncio
import json

from sqlalchemy import select

from research_lab.db import Base, SessionLocal, engine
from research_lab.mandates import process_mandate
from research_lab.models import ResearchMandate


async def run(limit: int) -> dict:
    Base.metadata.create_all(engine)
    processed: list[str] = []
    failures: dict[str, str] = {}
    with SessionLocal() as session:
        mandates = session.scalars(
            select(ResearchMandate)
            .where(ResearchMandate.status == "queued")
            .order_by(ResearchMandate.priority.desc(), ResearchMandate.created_at)
            .limit(limit)
        ).all()
        for mandate in mandates:
            try:
                await process_mandate(session, mandate)
                processed.append(mandate.id)
            except Exception as exc:
                failures[mandate.id] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"processed": processed, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Process metered client research mandates.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(max(1, min(args.limit, 25)))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
