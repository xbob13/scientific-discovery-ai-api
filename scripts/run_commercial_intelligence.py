import argparse
import asyncio
import json

from research_lab.config import get_settings
from research_lab.db import Base, SessionLocal, engine
from research_lab.institutions import discover_research_institutions
from research_lab.outreach import deliver_queued_messages, discover_and_draft_prospects


async def run(discover: bool, institutions: bool, deliver: bool, limit: int) -> dict:
    Base.metadata.create_all(engine)
    output: dict = {}
    with SessionLocal() as session:
        if discover:
            output["discovery"] = await discover_and_draft_prospects(session)
        if institutions:
            settings = get_settings()
            output["institutions"] = await discover_research_institutions(
                session,
                settings.institution_discovery_query,
                settings.institution_discovery_limit,
                seed_verified_channels=True,
            )
        if deliver:
            output["delivery"] = await deliver_queued_messages(session, limit)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover evidence-backed accounts and operate policy-gated autonomous outreach."
    )
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--discover-institutions", action="store_true")
    parser.add_argument("--deliver", action="store_true")
    parser.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    settings = get_settings()
    discover = args.discover or (args.all_enabled and settings.prospect_discovery_enabled)
    institutions = args.discover_institutions or (args.all_enabled and settings.institution_discovery_enabled)
    deliver = args.deliver or (args.all_enabled and settings.outreach_autonomous_enabled)
    if not discover and not institutions and not deliver:
        parser.error("no commercial intelligence operation is enabled")
    result = asyncio.run(run(discover, institutions, deliver, max(1, min(args.limit, 100))))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
