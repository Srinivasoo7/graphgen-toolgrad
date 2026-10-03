"""RED then GREEN: enrollment survives process restart (SQLite)."""

import pytest

from crux.access import EnrollmentStatus, Identity, Operation
from crux.registry import EnrollmentRegistry, UnknownIdentity


def test_sqlite_activate_survives_reopen(tmp_path):
    path = tmp_path / "enroll.db"
    identity = Identity("owner-1", "agent-a", "cred-a")
    first = EnrollmentRegistry(path=str(path))
    first.activate(
        identity,
        utopia_user_id="utp-user-1",
        utopia_token_id="utp-tok-1",
        granted_kb_ids=frozenset({"kb-production"}),
        operations=frozenset({Operation.READ}),
        utopia_pat="utp-pat-secret",
    )

    second = EnrollmentRegistry(path=str(path))
    record = second.lookup(identity)
    assert record.status is EnrollmentStatus.ACTIVE
    assert record.granted_kb_ids == frozenset({"kb-production"})
    assert record.utopia_user_id == "utp-user-1"
    assert second.token_for(identity) == "utp-pat-secret"


def test_sqlite_unknown_stays_pending_and_has_no_token(tmp_path):
    registry = EnrollmentRegistry(path=str(tmp_path / "enroll.db"))
    claimed = Identity("maybe-owner", "maybe-agent", "maybe-cred")
    record = registry.lookup(claimed)
    assert record.status is EnrollmentStatus.PENDING
    with pytest.raises(UnknownIdentity):
        registry.token_for(claimed)


def test_sqlite_revoke_survives_reopen(tmp_path):
    path = tmp_path / "enroll.db"
    identity = Identity("owner-1", "agent-a", "cred-a")
    first = EnrollmentRegistry(path=str(path))
    first.activate(
        identity,
        utopia_user_id="u",
        utopia_token_id="t",
        granted_kb_ids=frozenset({"kb-staging"}),
        operations=frozenset({Operation.READ, Operation.PROPOSE}),
        utopia_pat="pat",
    )
    first.revoke(identity)

    second = EnrollmentRegistry(path=str(path))
    assert second.lookup(identity).status is EnrollmentStatus.REVOKED
    with pytest.raises(UnknownIdentity):
        second.token_for(identity)
