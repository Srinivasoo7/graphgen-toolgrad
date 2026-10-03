from crux.access import (
    AccessDenied,
    Enrollment,
    EnrollmentStatus,
    Identity,
    Operation,
    Session,
    authorize,
    effective_kb_ids,
)
import pytest


PROD = "kb-production"
STAGING = "kb-staging"


def _active(**kwargs) -> Enrollment:
    identity = Identity(owner_id="owner-1", agent_id="agent-a", credential_id="cred-a")
    defaults = dict(
        identity=identity,
        status=EnrollmentStatus.ACTIVE,
        granted_kb_ids=frozenset({PROD}),
        operations=frozenset({Operation.READ}),
        utopia_user_id="utp-user-1",
        utopia_token_id="utp-tok-1",
    )
    defaults.update(kwargs)
    return Enrollment(**defaults)


def test_session_cannot_grant_extra_base():
    enr = _active()
    sess = Session(session_id="s1", identity=enr.identity, kb_ids=frozenset({PROD, STAGING}))
    with pytest.raises(AccessDenied, match="cannot grant"):
        effective_kb_ids(enr, sess)


def test_session_may_restrict_to_subset():
    enr = _active(granted_kb_ids=frozenset({PROD, STAGING}))
    sess = Session(session_id="s1", identity=enr.identity, kb_ids=frozenset({PROD}))
    assert effective_kb_ids(enr, sess) == frozenset({PROD})


def test_pending_enrollment_has_no_private_knowledge():
    enr = _active(status=EnrollmentStatus.PENDING, utopia_user_id=None, utopia_token_id=None)
    sess = Session(session_id="s1", identity=enr.identity, kb_ids=frozenset())
    with pytest.raises(AccessDenied, match="pending"):
        authorize(enr, sess, kb_id=PROD, operation=Operation.READ)


def test_untrusted_claimed_id_without_utopia_binding():
    enr = _active(utopia_user_id=None, utopia_token_id=None)
    sess = Session(session_id="s1", identity=enr.identity)
    with pytest.raises(AccessDenied, match="claimed"):
        authorize(enr, sess, kb_id=PROD, operation=Operation.READ)


def test_read_only_cannot_propose():
    enr = _active()
    sess = Session(session_id="s1", identity=enr.identity)
    with pytest.raises(AccessDenied, match="propose"):
        authorize(enr, sess, kb_id=PROD, operation=Operation.PROPOSE)


def test_revoked_denied():
    enr = _active(status=EnrollmentStatus.REVOKED)
    sess = Session(session_id="s1", identity=enr.identity)
    with pytest.raises(AccessDenied, match="revoked"):
        authorize(enr, sess, kb_id=PROD, operation=Operation.READ)
