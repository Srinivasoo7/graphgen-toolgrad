"""RED then GREEN: enrollment registry (ECC tdd-workflow)."""

import pytest

from crux.access import EnrollmentStatus, Identity, Operation
from crux.registry import EnrollmentRegistry, UnknownIdentity


def test_unknown_id_is_pending_and_has_no_grants():
    registry = EnrollmentRegistry()
    claimed = Identity("maybe-owner", "maybe-agent", "maybe-cred")
    record = registry.lookup(claimed)
    assert record.status is EnrollmentStatus.PENDING
    assert record.granted_kb_ids == frozenset()
    assert record.utopia_user_id is None


def test_admin_activate_binds_utopia_ids_and_bases():
    registry = EnrollmentRegistry()
    identity = Identity("owner-1", "agent-a", "cred-a")
    registry.activate(
        identity,
        utopia_user_id="utp-user-1",
        utopia_token_id="utp-tok-1",
        granted_kb_ids=frozenset({"kb-production"}),
        operations=frozenset({Operation.READ}),
    )
    record = registry.lookup(identity)
    assert record.status is EnrollmentStatus.ACTIVE
    assert record.granted_kb_ids == frozenset({"kb-production"})
    assert record.utopia_user_id == "utp-user-1"


def test_activate_without_utopia_binding_rejected():
    registry = EnrollmentRegistry()
    identity = Identity("owner-1", "agent-a", "cred-a")
    with pytest.raises(UnknownIdentity):
        registry.activate(
            identity,
            utopia_user_id=None,
            utopia_token_id=None,
            granted_kb_ids=frozenset({"kb-production"}),
            operations=frozenset({Operation.READ}),
        )


def test_revoke_blocks_later_lookup_as_revoked():
    registry = EnrollmentRegistry()
    identity = Identity("owner-1", "agent-a", "cred-a")
    registry.activate(
        identity,
        utopia_user_id="u",
        utopia_token_id="t",
        granted_kb_ids=frozenset({"kb-staging"}),
        operations=frozenset({Operation.READ, Operation.PROPOSE}),
    )
    registry.revoke(identity)
    assert registry.lookup(identity).status is EnrollmentStatus.REVOKED


def test_public_records_never_include_pats():
    registry = EnrollmentRegistry()
    identity = Identity("owner-1", "agent-a", "cred-a")
    registry.activate(
        identity,
        utopia_user_id="u",
        utopia_token_id="t",
        granted_kb_ids=frozenset({"kb-production"}),
        operations=frozenset({Operation.READ}),
        utopia_pat="super-secret-pat",
    )
    rows = registry.public_records()
    blob = str(rows)
    assert "super-secret-pat" not in blob
    assert rows[0]["agent_id"] == "agent-a"
    assert rows[0]["granted_kb_ids"] == ["kb-production"]
    assert "utopia_pat" not in rows[0]

