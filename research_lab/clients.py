from __future__ import annotations

import hashlib
import hmac
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ClientTopic, ClientWorkspace, IntelligenceBrief


def hash_portal_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify_portal_token(workspace: ClientWorkspace, token: str) -> bool:
    return hmac.compare_digest(workspace.portal_token_sha256, hash_portal_token(token))


def distribute_cycle(session: Session, cycle_key: str, result: dict) -> int:
    """Publish only relevant research into each isolated client workspace."""
    searchable = " ".join(
        [
            result.get("question", ""),
            *[
                " ".join(
                    [
                        candidate.get("title", ""),
                        " ".join(candidate.get("concepts", [])),
                        candidate.get("bridge", ""),
                    ]
                )
                for candidate in result.get("candidates", [])
            ],
        ]
    ).lower()
    published = 0
    topics = session.scalars(select(ClientTopic).where(ClientTopic.active.is_(True))).all()
    for topic in topics:
        keywords = {
            keyword.strip().lower()
            for keyword in topic.keywords
            if keyword.strip()
        }
        matched = sorted(
            keyword
            for keyword in keywords
            if re.search(rf"\b{re.escape(keyword)}\b", searchable)
        )
        if not matched:
            continue
        prior = session.scalar(
            select(IntelligenceBrief).where(
                IntelligenceBrief.topic_id == topic.id,
                IntelligenceBrief.cycle_key == cycle_key,
            )
        )
        if prior:
            continue
        candidates = [
            candidate
            for candidate in result.get("candidates", [])
            if any(
                keyword in (
                    candidate.get("title", "")
                    + " "
                    + " ".join(candidate.get("concepts", []))
                    + " "
                    + candidate.get("bridge", "")
                ).lower()
                for keyword in matched
            )
        ]
        session.add(
            IntelligenceBrief(
                workspace_id=topic.workspace_id,
                topic_id=topic.id,
                cycle_key=cycle_key,
                title=f"{topic.name}: materials intelligence update",
                executive_summary=(
                    f"Cycle matched {len(matched)} client terms and produced "
                    f"{len(candidates)} relevant evidence-backed candidates."
                ),
                payload={
                    "question": result.get("question"),
                    "matched_keywords": matched,
                    "candidates": candidates,
                    "source_summary": result.get("sources", {}),
                    "completed_at": result.get("completed_at"),
                },
            )
        )
        published += 1
    session.commit()
    return published
