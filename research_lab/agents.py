import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CandidateAssessment, ConnectionCandidate, Work, WorkVersion

STOP_WORDS = {
    "about", "according", "after", "also", "among", "analysis", "approach", "based",
    "been", "between", "both", "conclusion", "considered", "current", "different",
    "discussed", "finally", "focus", "from", "have", "however", "including", "into",
    "investigated", "material", "materials", "method", "methods", "more", "most",
    "paper", "performance", "possible", "process", "progress", "properties", "proposed",
    "recent", "research", "results", "review", "show", "shown", "significant", "some",
    "state", "studies", "study", "such", "systematic", "than", "that", "their",
    "therefore", "these", "this", "through", "using", "were", "which", "with",
}

MECHANISM_TERMS = {
    "adsorption", "amperometric", "antibody", "battery", "binding", "biosensing",
    "catalysis", "catalyst", "conductivity", "corrosion", "desalination", "diffusion",
    "electrode", "electrochemical", "electrolysis", "energy", "enzyme", "field-effect",
    "filtration", "graphene", "harvesting", "impedance", "membrane", "metamaterial",
    "nanostructure", "oxide", "photovoltaic", "polymer", "porosity", "protein",
    "recycling", "sensor", "sensors", "signal", "sorption", "thermal", "transducer",
    "transducers", "wastewater", "water",
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
    source_url: str = ""
    published_at: str | None = None


@dataclass(frozen=True)
class ProposedConnection:
    left_id: str
    right_id: str
    bridge_terms: tuple[str, ...]
    score: float
    score_components: dict
    left_evidence: str
    right_evidence: str
    novelty_status: str
    corroborating_ids: tuple[str, ...]


def evidence_excerpt(text: str, bridge_terms: tuple[str, ...], limit: int = 360) -> str:
    """Return an exact, bounded source excerpt containing the strongest bridge signal."""
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    ranked = sorted(
        sentences,
        key=lambda sentence: sum(term in sentence.lower() for term in bridge_terms),
        reverse=True,
    )
    excerpt = next((sentence for sentence in ranked if sentence), normalized)
    return excerpt[:limit].rstrip()


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
                mechanism_shared = shared & MECHANISM_TERMS
                if not mechanism_shared:
                    continue
                ranked = sorted(
                    shared,
                    key=lambda term: (
                        math.log((len(documents) + 1) / (frequency[term] + 1)),
                        term,
                    ),
                    reverse=True,
                )
                bridge = tuple(sorted(mechanism_shared, key=lambda term: ranked.index(term))[:3])
                specificity = sum(
                    math.log((len(documents) + 1) / (frequency[term] + 1))
                    for term in bridge
                )
                review_quality = (float(left.peer_reviewed) + float(right.peer_reviewed)) / 2
                title_terms_left = terms(left.title)
                title_terms_right = terms(right.title)
                title_union = title_terms_left | title_terms_right
                domain_divergence = 1 - (
                    len(title_terms_left & title_terms_right) / len(title_union)
                    if title_union
                    else 0
                )
                corroboration_threshold = min(2, len(bridge))
                corroborating_ids = tuple(
                    document.work_id
                    for document in documents
                    if document.work_id not in {left.work_id, right.work_id}
                    and len(set(bridge) & document_terms[document.work_id])
                    >= corroboration_threshold
                )
                novelty_status = (
                    "unresolved_in_collected_corpus"
                    if not corroborating_ids
                    else "partially_corroborated"
                    if len(corroborating_ids) == 1
                    else "likely_established"
                )
                novelty_signal = 1.0 if not corroborating_ids else 0.4 if len(corroborating_ids) == 1 else 0.0
                evidence_coverage = min(1.0, len(mechanism_shared) / 3)
                score = 100 * (
                    0.30 * min(1.0, specificity / 5)
                    + 0.20 * evidence_coverage
                    + 0.15 * review_quality
                    + 0.20 * domain_divergence
                    + 0.15 * novelty_signal
                )
                proposals.append(
                    ProposedConnection(
                        left.work_id,
                        right.work_id,
                        bridge,
                        round(score, 2),
                        {
                            "lexical_specificity": round(min(1.0, specificity / 5), 4),
                            "evidence_coverage": round(evidence_coverage, 4),
                            "peer_review_quality": round(review_quality, 4),
                            "domain_divergence": round(domain_divergence, 4),
                            "novelty_signal": round(novelty_signal, 4),
                        },
                        evidence_excerpt(f"{left.title}. {left.abstract}", bridge),
                        evidence_excerpt(f"{right.title}. {right.abstract}", bridge),
                        novelty_status,
                        corroborating_ids[:5],
                    )
                )
        return sorted(
            proposals,
            key=lambda item: (-item.score, item.left_id, item.right_id),
        )[:limit]


def load_documents(session: Session, limit: int = 200) -> list[ResearchDocument]:
    rows = session.execute(
        select(Work, WorkVersion)
        .join(WorkVersion, WorkVersion.work_id == Work.id)
        .where(WorkVersion.abstract.is_not(None))
        .order_by(Work.created_at.desc())
        .limit(limit)
    )
    return [
        ResearchDocument(
            work.id,
            work.title,
            version.abstract or "",
            version.peer_reviewed,
            version.source_url,
            version.published_at.isoformat() if version.published_at else None,
        )
        for work, version in rows
    ]


def persist_connections(session: Session, proposals: list[ProposedConnection]) -> list[str]:
    """Persist candidates and a separately versioned assessment; return newly assessed IDs."""
    assessed: list[str] = []
    documents = {document.work_id: document for document in load_documents(session)}
    existing = session.scalars(select(ConnectionCandidate)).all()
    for proposal in proposals:
        source_ids = sorted([proposal.left_id, proposal.right_id])
        candidate = next(
            (item for item in existing if sorted(item.source_work_ids) == source_ids),
            None,
        )
        left, right = documents[proposal.left_id], documents[proposal.right_id]
        bridge = ", ".join(proposal.bridge_terms)
        if candidate is None:
            candidate = ConnectionCandidate(
                title=f"Test whether {bridge} links “{left.title}” and “{right.title}”",
                source_work_ids=source_ids,
                concepts=list(proposal.bridge_terms),
                bridge=(
                    f"Both source records independently discuss {bridge}. The research task is to "
                    "determine whether the mechanisms are compatible rather than merely lexical."
                ),
                assumptions=[
                    "The shared terms have compatible technical meanings in both sources.",
                    "The operating conditions and material systems can be compared.",
                ],
                falsifiable_prediction=(
                    f"A targeted search combining {bridge} will recover independent primary research "
                    "that explicitly transfers or contrasts the mechanism between the two systems."
                ),
                validation_method=(
                    "Inspect full text, identify measured mechanisms and boundary conditions, seek "
                    "independent corroboration, then design a quantitative reproduction."
                ),
                reasons_to_reject=[
                    "The shared terminology has different meanings in the two domains.",
                    "Boundary conditions make the mechanisms physically incompatible.",
                    "Targeted prior-art search shows the relationship is already established.",
                    "No independent evidence supports the proposed bridge.",
                ],
                status="machine_proposed",
            )
            session.add(candidate)
            session.flush()
            existing.append(candidate)
        prior = session.scalar(
            select(CandidateAssessment).where(
                CandidateAssessment.connection_id == candidate.id,
                CandidateAssessment.assessment_version == "evidence-v1",
            )
        )
        if prior:
            continue
        evidence = [
            {
                "work_id": document.work_id,
                "title": document.title,
                "url": document.source_url,
                "published_at": document.published_at,
                "peer_reviewed": document.peer_reviewed,
                "excerpt": excerpt,
            }
            for document, excerpt in (
                (left, proposal.left_evidence),
                (right, proposal.right_evidence),
            )
        ]
        corroboration = [
            {
                "work_id": documents[work_id].work_id,
                "title": documents[work_id].title,
                "url": documents[work_id].source_url,
            }
            for work_id in proposal.corroborating_ids
            if work_id in documents
        ]
        session.add(
            CandidateAssessment(
                connection_id=candidate.id,
                score=proposal.score,
                score_components=proposal.score_components,
                source_evidence=evidence,
                novelty_status=proposal.novelty_status,
                novelty_queries=[
                    " ".join(f'"{term}"' for term in proposal.bridge_terms[:3])
                ],
                corroboration=corroboration,
                limitations=[
                    "Assessment uses bibliographic metadata and available abstracts, not full text.",
                    "Score ranks retrieval priority and is not a probability that the claim is true.",
                    "Novelty remains unverified until targeted prior-art search is completed.",
                ],
            )
        )
        assessed.append(candidate.id)
    session.commit()
    return assessed
