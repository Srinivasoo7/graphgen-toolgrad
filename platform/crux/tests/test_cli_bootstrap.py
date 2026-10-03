import json

from crux.__main__ import main
from crux.access import Identity
from crux.bootstrap import BootstrapResult


def test_bootstrap_command_writes_state(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "hr")
    monkeypatch.setenv("CRUX_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("CRUX_ADMIN_PASSWORD", "password12")
    monkeypatch.setenv("CRUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UTOPIA_URL", "http://utopia.invalid")

    def fake_bootstrap(admin, registry, *, email, password, owner_id):
        registry.activate(
            Identity(owner_id, "agent-prod", "cred-prod"),
            utopia_user_id="u",
            utopia_token_id="t",
            granted_kb_ids=frozenset({"kb-production"}),
            operations=frozenset(),
            utopia_pat="pat",
        )
        return BootstrapResult(
            production_kb_id="kb-production",
            staging_kb_id="kb-staging",
            workspace_id="ws-1",
            user_id="u",
        )

    monkeypatch.setattr("crux.__main__.bootstrap_release_a", fake_bootstrap)
    assert main(["bootstrap"]) == 0
    state = json.loads((tmp_path / "bootstrap.json").read_text(encoding="utf-8"))
    assert state["production_kb_id"] == "kb-production"
    printed = json.loads(capsys.readouterr().out)
    assert printed["staging_kb_id"] == "kb-staging"
