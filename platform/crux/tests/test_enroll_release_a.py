import pytest

from crux.access import AccessDenied
from tests.test_app import _app, _settings


def _restricted_app(tmp_path):
    (tmp_path / "bootstrap.json").write_text(
        '{"production_kb_id":"kb-p","staging_kb_id":"kb-s"}',
        encoding="utf-8",
    )
    return _app(settings=_settings(data_dir=str(tmp_path)))


def test_enroll_rejects_kb_outside_bootstrapped_restricted_bases(tmp_path):
    app = _restricted_app(tmp_path)
    with pytest.raises(AccessDenied, match="Production or Staging"):
        app.enroll(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-x",
                "credential_id": "cred-x",
                "utopia_user_id": "u",
                "utopia_token_id": "t",
                "utopia_pat": "p",
                "granted_kb_ids": ["kb-general-open"],
                "operations": ["read"],
            },
            admin_token="admin-secret",
        )


def test_enroll_rejects_mixed_kb_set(tmp_path):
    app = _restricted_app(tmp_path)
    with pytest.raises(AccessDenied, match="Production or Staging"):
        app.enroll(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-x",
                "credential_id": "cred-x",
                "utopia_user_id": "u",
                "utopia_token_id": "t",
                "utopia_pat": "p",
                "granted_kb_ids": ["kb-p", "kb-general-open"],
                "operations": ["read"],
            },
            admin_token="admin-secret",
        )


def test_enroll_allows_bootstrapped_restricted_bases(tmp_path):
    app = _restricted_app(tmp_path)
    result = app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-p",
            "credential_id": "cred-p",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-p"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    assert result["status"] == "active"
    assert result["granted_kb_ids"] == ["kb-p"]
