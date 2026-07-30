import argparse
import os

from sqlalchemy import select

from research_lab.clients import hash_portal_token
from research_lab.db import Base, SessionLocal, engine
from research_lab.models import ClientTopic, ClientWorkspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision an isolated materials-intelligence client.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--topic-name", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--keywords", required=True, help="Comma-separated topic keywords.")
    args = parser.parse_args()
    token = os.getenv("CLIENT_PORTAL_TOKEN")
    if not token or len(token) < 24:
        raise SystemExit("CLIENT_PORTAL_TOKEN must contain at least 24 characters")

    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        workspace = session.scalar(
            select(ClientWorkspace).where(ClientWorkspace.slug == args.slug)
        )
        if workspace is None:
            workspace = ClientWorkspace(
                name=args.name,
                slug=args.slug,
                portal_token_sha256=hash_portal_token(token),
            )
            session.add(workspace)
            session.flush()
        topic = session.scalar(
            select(ClientTopic).where(
                ClientTopic.workspace_id == workspace.id,
                ClientTopic.name == args.topic_name,
            )
        )
        if topic is None:
            topic = ClientTopic(
                workspace_id=workspace.id,
                name=args.topic_name,
                research_question=args.question,
                keywords=sorted(
                    {
                        keyword.strip().lower()
                        for keyword in args.keywords.split(",")
                        if keyword.strip()
                    }
                ),
            )
            session.add(topic)
        session.commit()
        print(f"workspace={workspace.slug} topic={topic.name}")


if __name__ == "__main__":
    main()
