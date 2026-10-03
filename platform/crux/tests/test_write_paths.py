"""Separately authorized operations: propose ≠ approve ≠ schema ≠ permission."""

from crux.access import (
    AccessDenied,
    Enrollment,
    EnrollmentStatus,
    Identity,
    Operation,
    Session,
    authorize,
)
import pytest


def _enroll(ops):
    identity = Identity("owner", "agent-b", "cred-b")
    return Enrollment(
        identity=identity,
        status=EnrollmentStatus.ACTIVE,
        granted_kb_ids=frozenset({"kb-staging"}),
        operations=frozenset(ops),
        utopia_user_id="u2",
        utopia_token_id="t2",
    ), Session("s1", identity)


def test_propose_capability_does_not_imply_approve_schema_or_permission():
    enrollment, session = _enroll({Operation.READ, Operation.PROPOSE})
    authorize(enrollment, session, kb_id="kb-staging", operation=Operation.PROPOSE)
    # These operations are not in the enum on purpose: they are Utopia UI only
    # until pin-verify exposes mutation APIs. Agents must not gain them here.
    assert not hasattr(Operation, "APPROVE")
    assert not hasattr(Operation, "EDIT_ONTOLOGY")
    assert not hasattr(Operation, "CHANGE_PERMISSION")


def test_read_only_agent_cannot_hit_propose_path():
    enrollment, session = _enroll({Operation.READ})
    with pytest.raises(AccessDenied):
        authorize(enrollment, session, kb_id="kb-staging", operation=Operation.PROPOSE)
