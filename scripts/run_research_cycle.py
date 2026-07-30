import argparse
import json
import os
from pathlib import Path

from research_lab.agenda import scheduled_question
from research_lab.db import Base, SessionLocal, engine
from research_lab.orchestrator import ResearchCycle, run_cycle_sync
from research_lab.reports import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded, auditable digital research cycle.")
    parser.add_argument(
        "--question",
        default=os.getenv("RESEARCH_QUESTION") or scheduled_question(),
        help="Research objective; omitted scheduled runs rotate through the lab agenda.",
    )
    parser.add_argument("--limit-per-source", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/latest-cycle.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/latest-cycle.md"))
    args = parser.parse_args()
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        result = run_cycle_sync(
            session,
            ResearchCycle(question=args.question, limit_per_source=args.limit_per_source),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
