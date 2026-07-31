import argparse
import asyncio
import json

from research_lab.db import Base, SessionLocal, engine
from research_lab.outreach import deliver_approved_messages, discover_and_draft_prospects


async def run(discover: bool, deliver: bool, limit: int) -> dict:
    Base.metadata.create_all(engine)
    output: dict = {}
    with SessionLocal() as session:
        if discover:
            output["discovery"] = await discover_and_draft_prospects(session)
        if deliver:
            output["delivery"] = await deliver_approved_messages(session, limit)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest governed prospect evidence and deliver already-approved outreach."
    )
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--deliver-approved", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if not args.discover and not args.deliver_approved:
        parser.error("select --discover and/or --deliver-approved")
    result = asyncio.run(
        run(args.discover, args.deliver_approved, max(1, min(args.limit, 100)))
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
