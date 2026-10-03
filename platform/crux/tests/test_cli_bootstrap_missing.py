import pytest

from crux.__main__ import main


def test_propose_without_bootstrap_explains_how_to_fix(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CRUX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CRUX_KB_ID", raising=False)
    monkeypatch.setenv("CRUX_AGENT_ID", "agent-staging")
    monkeypatch.setenv("CRUX_CREDENTIAL_ID", "cred-staging")
    with pytest.raises(SystemExit) as raised:
        main(["propose", "hello"])
    assert raised.value.code == 2
    assert "bootstrap.json" in capsys.readouterr().err
