from research_lab.clients import (
    distribute_cycle,
    hash_portal_token,
    verify_portal_token,
)
from research_lab.models import ClientTopic, ClientWorkspace, IntelligenceBrief


def test_portal_tokens_are_hashed_and_constant_time_verified():
    workspace = ClientWorkspace(
        name="Example",
        slug="example",
        portal_token_sha256=hash_portal_token("a-secure-client-token-1234"),
    )
    assert verify_portal_token(workspace, "a-secure-client-token-1234")
    assert not verify_portal_token(workspace, "wrong-token")
    assert "a-secure-client-token" not in workspace.portal_token_sha256


def test_cycle_is_distributed_only_to_relevant_client_topics(session):
    relevant = ClientWorkspace(
        name="Membrane Client",
        slug="membrane-client",
        portal_token_sha256=hash_portal_token("membrane-client-secret-token"),
    )
    unrelated = ClientWorkspace(
        name="Battery Client",
        slug="battery-client",
        portal_token_sha256=hash_portal_token("battery-client-secret-token"),
    )
    session.add_all([relevant, unrelated])
    session.flush()
    session.add_all(
        [
            ClientTopic(
                workspace_id=relevant.id,
                name="Fouling control",
                research_question="Find scalable membrane fouling treatments.",
                keywords=["membrane", "fouling"],
            ),
            ClientTopic(
                workspace_id=unrelated.id,
                name="Solid state batteries",
                research_question="Find solid electrolyte advances.",
                keywords=["battery", "electrolyte"],
            ),
        ]
    )
    session.commit()
    result = {
        "question": "membrane fouling control",
        "completed_at": "2026-07-30T00:00:00Z",
        "sources": {"openalex": {"records": 10}},
        "candidates": [
            {
                "title": "Antifouling membrane coating",
                "concepts": ["membrane", "fouling"],
                "bridge": "A surface treatment may reduce irreversible fouling.",
            }
        ],
    }
    assert distribute_cycle(session, "cycle-1", result) == 1
    briefs = session.query(IntelligenceBrief).all()
    assert len(briefs) == 1
    assert briefs[0].workspace_id == relevant.id
    assert briefs[0].payload["matched_keywords"] == ["fouling", "membrane"]
    assert distribute_cycle(session, "cycle-1", result) == 0
