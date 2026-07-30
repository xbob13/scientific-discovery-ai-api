from research_lab.agents import (
    LiteratureCartographer,
    ResearchDocument,
    evidence_excerpt,
    persist_connections,
)
from research_lab.models import JobRun, PublicationState, Work, WorkVersion
from research_lab.reports import candidate_report, render_markdown


def test_cartographer_proposes_explainable_deterministic_connection():
    documents = [
        ResearchDocument("a", "Graphene sensor", "A graphene surface detects protein binding.", True),
        ResearchDocument("b", "Protein adsorption", "Protein binding changes graphene conductivity.", True),
        ResearchDocument("c", "Unrelated astronomy", "A distant galaxy contains old stars.", True),
    ]
    first = LiteratureCartographer().propose(documents)
    second = LiteratureCartographer().propose(documents)
    assert first == second
    assert len(first) == 1
    assert {first[0].left_id, first[0].right_id} == {"a", "b"}
    assert "graphene" in first[0].bridge_terms


def test_cartographer_does_not_invent_connection_without_shared_signal():
    documents = [
        ResearchDocument("a", "Graphene sensor", "Conductivity measurement.", True),
        ResearchDocument("b", "Protein assay", "Fluorescent antibody.", True),
    ]
    assert LiteratureCartographer().propose(documents) == []


def test_cartographer_rejects_generic_academic_prose():
    documents = [
        ResearchDocument("a", "First review", "However, studies show significant progress.", True),
        ResearchDocument("b", "Second review", "Studies therefore show significant progress.", True),
    ]
    assert LiteratureCartographer().propose(documents) == []


def test_new_job_attempt_counter_can_be_incremented_before_flush():
    job = JobRun(job_type="cycle", idempotency_key="test", input={})
    job.attempts = (job.attempts or 0) + 1
    assert job.attempts == 1


def test_evidence_excerpt_prefers_sentence_with_bridge():
    text = "Background sentence. Graphene conductivity changes after protein binding. Closing."
    assert evidence_excerpt(text, ("graphene", "binding")).startswith("Graphene conductivity")


def test_persisted_candidate_has_citations_score_and_rejection_criteria(session):
    left = Work(title="Graphene sensor", normalized_title="graphene sensor")
    right = Work(title="Protein adsorption", normalized_title="protein adsorption")
    session.add_all([left, right])
    session.flush()
    session.add_all(
        [
            WorkVersion(
                work_id=left.id,
                source_snapshot_id="snapshot-left",
                publication_state=PublicationState.PUBLISHED,
                peer_reviewed=True,
                abstract="A graphene surface detects protein binding through conductivity.",
                source_url="https://example.org/left",
            ),
            WorkVersion(
                work_id=right.id,
                source_snapshot_id="snapshot-right",
                publication_state=PublicationState.PUBLISHED,
                peer_reviewed=True,
                abstract="Protein binding changes graphene conductivity.",
                source_url="https://example.org/right",
            ),
        ]
    )
    session.commit()
    proposals = LiteratureCartographer().propose(
        [
            ResearchDocument(
                left.id,
                left.title,
                "A graphene surface detects protein binding through conductivity.",
                True,
                "https://example.org/left",
            ),
            ResearchDocument(
                right.id,
                right.title,
                "Protein binding changes graphene conductivity.",
                True,
                "https://example.org/right",
            ),
        ]
    )
    ids = persist_connections(session, proposals)
    report = candidate_report(session, ids)
    assert len(report) == 1
    assert report[0]["priority_score"] > 0
    assert len(report[0]["source_evidence"]) == 2
    assert all(source["url"] for source in report[0]["source_evidence"])
    assert report[0]["reasons_to_reject"]
    assert "Priority score" in render_markdown(
        {
            "question": "test",
            "completed_at": "now",
            "new_versions": 2,
            "connections_created": 1,
            "candidates": report,
        }
    )
