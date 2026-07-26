from research_lab.canonical import store_record
from research_lab.models import PublicationState
from research_lab.schemas import SourceRecord


def record(source: str, external_id: str):
    return SourceRecord(
        source_name=source,
        endpoint=f"https://{source}.example/works",
        external_id=external_id,
        title="A measured materials result",
        doi="https://doi.org/10.1234/ABC.1",
        publication_state=PublicationState.PUBLISHED,
        peer_reviewed=True,
        source_url="https://doi.org/10.1234/abc.1",
        raw={"id": external_id, "source": source},
    )


def test_cross_source_doi_dedup_preserves_versions(session):
    first, first_created = store_record(session, record("openalex", "W1"))
    second, second_created = store_record(session, record("crossref", "10.1234/abc.1"))
    session.commit()
    assert first_created and second_created
    assert first.id == second.id
    assert len(first.versions) == 2


def test_snapshot_replay_is_idempotent(session):
    first, created = store_record(session, record("openalex", "W1"))
    replay, replay_created = store_record(session, record("openalex", "W1"))
    assert created
    assert not replay_created
    assert first.id == replay.id
