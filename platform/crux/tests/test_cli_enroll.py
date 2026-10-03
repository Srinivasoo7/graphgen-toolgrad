import json

from crux.__main__ import main
from crux.access import Identity, Operation
from crux.registry import EnrollmentRegistry


def test_enroll_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "hr")
    monkeypatch.setenv("CRUX_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("CRUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UTOPIA_URL", "http://utopia.invalid")
    monkeypatch.setenv("HEADROOM_URL", "http://headroom.invalid")

    payload = tmp_path / "enroll.json"
    payload.write_text(
        json.dumps(
            {
                "owner_id": "owner-1",
                "agent_id": "incident-bot",
                "credential_id": "cred-inc",
                "utopia_user_id": "u",
                "utopia_token_id": "t",
                "utopia_pat": "pat",
                "granted_kb_ids": ["kb-production"],
                "operations": ["read"],
            }
        ),
        encoding="utf-8",
    )
    assert main(["enroll", str(payload)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "active"
    registry = EnrollmentRegistry(path=str(tmp_path / "enroll.db"))
    record = registry.lookup(Identity("owner-1", "incident-bot", "cred-inc"))
    assert Operation.READ in record.operations
