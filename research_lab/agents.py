import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ConnectionCandidate, Work, WorkVersion

STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "based",
    "been",
    "between",
    "both",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "show",
    "such",
    "than",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "were",
    "which",
    "with",
}


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())
        if token not in STOP_WORDS
    }


@dataclass(frozen=True)
class ResearchDocument:
    work_id: str
    title: str
    abstract: str
    peer_reviewed: bool


@dataclass(frozen=True)
class ProposedConnection:
    left_id: str
    right_id: str
    bridge_terms: tuple[str, ...]
    score: float


class LiteratureCartographer:
    """Deterministic candidate generator; it proposes links but never asserts truth."""

    def propose(self, documents: list[ResearchDocument], limit: int = 20) -> list[ProposedConnection]:
        if len(documents) < 2:
            return []
        document_terms = {
            document.work_id: terms(f"{document.title} {document.abstract}")
            for document in documents
        }
        frequency = Counter(term for values in document_terms.values() for term in values)
        proposals: list[ProposedConnection] = []
        for index, left in enumerate(documents):
            for right in documents[index + 1 :]:
                shared = document_terms[left.work_id] & document_terms[right.work_id]
                if not shared:
                    continue
                ranked = sorted(
                    shared,
                    key=lambda term: (math.log((len(documents) + 1) / (frequency[term] + 1)), term),
                    reverse=True,
                )
                bridge = tuple(ranked[:5])
                score = sum(math.log((len(documents) + 1) / (frequency[t] + 1)) for t in bridge)
                proposals.append(ProposedConnection(left.work_id, right.work_id, bridge, round(score, 6)))
        return sorted(proposals, key=lambda item: (-item.score, item.left_id, item.right_id))[:limit]


def load_documents(session: Session, limit: int = 200) -> list[ResearchDocument]:
    rows = session.execute(
        select(Work, WorkVersion)
        .join(WorkVersion, WorkVersion.work_id == Work.id)
        .where(WorkVersion.abstract.is_not(None))
        .order_by(Work.created_at.desc())
        .limit(limit)
    )
    return [
        ResearchDocument(work.id, work.title, version.abstract or "", version.peer_reviewed)
        for work, version in rows
    ]


def persist_connections(session: Session, proposals: list[ProposedConnection]) -> int:
    created = 0
    for proposal in proposals:
        source_ids = sorted([proposal.left_id, proposal.right_id])
        existing = session.scalars(select(ConnectionCandidate)).all()
        if any(sorted(candidate.source_work_ids) == source_ids for candidate in existing):
            continue
        bridge = ", ".join(proposal.bridge_terms)
        session.add(
            ConnectionCandidate(
                title=f"Investigate a possible bridge through {bridge}",
                source_work_ids=source_ids,
                concepts=list(proposal.bridge_terms),
                bridge=(
                    "The source records share unusually informative terminology. "
                    "This is a retrieval signal, not evidence that the proposed relationship is true."
                ),
                assumptions=[
                    "The terms are used with compatible definitions.",
                    "The reported systems are comparable enough to justify follow-up.",
                ],
                falsifiable_prediction=(
                    f"If the bridge through {bridge} is meaningful, a targeted search should recover "
                    "independent work that explicitly relates both source domains."
                ),
                validation_method=(
                    "Run a targeted multi-source search, inspect primary-source passages, and require "
                    "independent corroboration before promotion."
                ),
                reasons_to_reject=[
                    "Shared vocabulary is coincidental or polysemous.",
                    "The source conditions or populations are incompatible.",
                    "No independent corroborating source can be found.",
                ],
                status="machine_proposed",
            )
        )
        created += 1
    session.commit()
    return created
