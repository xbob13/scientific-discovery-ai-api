import pytest

from research_lab.clients import hash_portal_token
from research_lab.mandates import create_mandate
from research_lab.models import ClientWorkspace, WorkspaceEntitlement
from research_lab.schemas import ClientMandateCreate


def request(reference: str) -> ClientMandateCreate:
    return ClientMandateCreate(
        title="Membrane prior-art landscape",
        question=(
            "Which coating chemistries reduce irreversible hollow-fiber membrane fouling "
            "under industrial water treatment conditions?"
        ),
        sources=["openalex", "patentsview"],
        external_reference=reference,
    )


def test_client_mandates_are_idempotent_and_metered(session):
    workspace = ClientWorkspace(
        name="Membrane Co",
        slug="membrane-co",
        portal_token_sha256=hash_portal_token("membrane-client-access-token"),
    )
    session.add(workspace)
    session.flush()
    session.add(
        WorkspaceEntitlement(
            workspace_id=workspace.id,
            plan="commissioned-brief",
            allowed_sources=["openalex", "patentsview"],
            monthly_request_limit=1,
        )
    )
    session.commit()

    first = create_mandate(session, workspace, request("client-job-1"))
    repeated = create_mandate(session, workspace, request("client-job-1"))
    assert first.id == repeated.id
    with pytest.raises(OverflowError):
        create_mandate(session, workspace, request("client-job-2"))


def test_client_cannot_request_sources_outside_entitlement(session):
    workspace = ClientWorkspace(
        name="Limited Co",
        slug="limited-co",
        portal_token_sha256=hash_portal_token("limited-client-access-token"),
    )
    session.add(workspace)
    session.flush()
    session.add(
        WorkspaceEntitlement(
            workspace_id=workspace.id,
            allowed_sources=["openalex"],
            monthly_request_limit=3,
        )
    )
    session.commit()
    payload = request("restricted-source")
    with pytest.raises(PermissionError):
        create_mandate(session, workspace, payload)
