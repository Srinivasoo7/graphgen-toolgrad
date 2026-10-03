import pytest

from crux.access import AccessDenied
from tests.test_app import _app


def test_list_enrollments_requires_admin_and_hides_pats():
    app = _app()
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "hidden-pat",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    with pytest.raises(AccessDenied):
        app.list_enrollments(admin_token="wrong")
    rows = app.list_enrollments(admin_token="admin-secret")
    assert rows[0]["agent_id"] == "agent-a"
    assert "hidden-pat" not in str(rows)
    assert "utopia_pat" not in rows[0]
