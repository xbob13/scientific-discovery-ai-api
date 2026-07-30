from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CandidateAssessment, ConnectionCandidate


def candidate_report(
    session: Session,
    candidate_ids: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    query = (
        select(ConnectionCandidate, CandidateAssessment)
        .join(CandidateAssessment, CandidateAssessment.connection_id == ConnectionCandidate.id)
        .order_by(CandidateAssessment.score.desc(), CandidateAssessment.created_at.desc())
        .limit(limit)
    )
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        query = query.where(ConnectionCandidate.id.in_(candidate_ids))
    return [
        {
            "id": candidate.id,
            "title": candidate.title,
            "status": candidate.status,
            "priority_score": assessment.score,
            "score_components": assessment.score_components,
            "bridge": candidate.bridge,
            "concepts": candidate.concepts,
            "source_evidence": assessment.source_evidence,
            "falsifiable_prediction": candidate.falsifiable_prediction,
            "validation_method": candidate.validation_method,
            "assumptions": candidate.assumptions,
            "reasons_to_reject": candidate.reasons_to_reject,
            "novelty": {
                "status": assessment.novelty_status,
                "queries": assessment.novelty_queries,
                "corroboration": assessment.corroboration,
            },
            "limitations": assessment.limitations,
        }
        for candidate, assessment in session.execute(query)
    ]


def render_markdown(result: dict) -> str:
    lines = [
        "# Autonomous research cycle",
        "",
        f"**Question:** {result['question']}",
        f"**Completed:** {result['completed_at']}",
        f"**New source versions:** {result['new_versions']}",
        f"**New evidence-backed candidates:** {result['connections_created']}",
        "",
        "> Machine-generated research leads. Scores rank review priority; they are not truth probabilities.",
        "",
    ]
    candidates = result.get("candidates", [])
    if not candidates:
        lines += [
            "## Result",
            "",
            "No new candidate passed the evidence gate in this cycle.",
            "",
        ]
    for index, candidate in enumerate(candidates, 1):
        lines += [
            f"## {index}. {candidate['title']}",
            "",
            f"**Priority score:** {candidate['priority_score']}/100",
            "",
            candidate["bridge"],
            "",
            "### Source evidence",
            "",
        ]
        for source in candidate["source_evidence"]:
            lines += [
                f"- [{source['title']}]({source['url']})"
                f" — peer reviewed: `{str(source['peer_reviewed']).lower()}`",
                f"  - Excerpt: “{source['excerpt']}”",
            ]
        lines += [
            "",
            f"**Falsifiable prediction:** {candidate['falsifiable_prediction']}",
            "",
            f"**Validation:** {candidate['validation_method']}",
            "",
            f"**Novelty status:** `{candidate['novelty']['status']}`",
            "",
            "**Reject this lead if:**",
            "",
            *[f"- {reason}" for reason in candidate["reasons_to_reject"]],
            "",
        ]
    return "\n".join(lines)
