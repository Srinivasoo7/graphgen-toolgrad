import pytest

from crux.access import AccessDenied
from tests.test_app import _app


def _enroll(app, kbs, ops=None):
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": kbs,
            "operations": ops or ["read"],
        },
        admin_token="admin-secret",
    )


def test_start_session_may_only_restrict():
    app = _app()
    _enroll(app, ["kb-production", "kb-staging"])
    opened = app.start_session(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "kb_ids": ["kb-production"],
        }
    )
    assert opened["kb_ids"] == ["kb-production"]
    assert opened["session_id"]


def test_start_session_rejects_ungranted_base():
    app = _app()
    _enroll(app, ["kb-production"])
    with pytest.raises(AccessDenied, match="cannot grant"):
        app.start_session(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "kb_ids": ["kb-staging"],
            }
        )


def test_start_session_pending_is_denied():
    app = _app()
    with pytest.raises(AccessDenied, match="pending"):
        app.start_session(
            {
                "owner_id": "stranger",
                "agent_id": "nope",
                "credential_id": "nope",
            }
        )
