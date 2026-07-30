import argparse
import json
import os
from pathlib import Path

from sqlalchemy import select

from research_lab.agenda import scheduled_client_question
from research_lab.db import Base, SessionLocal, engine
from research_lab.models import ClientTopic
from research_lab.orchestrator import ResearchCycle, run_cycle_sync
from research_lab.reports import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded, auditable digital research cycle.")
    parser.add_argument(
        "--question",
        default=os.getenv("RESEARCH_QUESTION"),
        help="Research objective; omitted runs prioritize active client topics, then the lab agenda.",
    )
    parser.add_argument("--limit-per-source", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/latest-cycle.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/latest-cycle.md"))
    args = parser.parse_args()
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        question = args.question or scheduled_client_question(
            list(
                session.scalars(
                    select(ClientTopic.research_question)
                    .where(ClientTopic.active.is_(True))
                    .order_by(ClientTopic.id)
                )
            )
        )
        result = run_cycle_sync(
            session,
            ResearchCycle(question=question, limit_per_source=args.limit_per_source),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
