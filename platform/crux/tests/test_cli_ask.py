from crux.__main__ import main
from crux.client import CruxClient


def test_ask_command(monkeypatch, capsys):
    monkeypatch.setenv("CRUX_URL", "http://crux.example")
    monkeypatch.setenv("CRUX_OWNER_ID", "owner-1")
    monkeypatch.setenv("CRUX_AGENT_ID", "agent-prod")
    monkeypatch.setenv("CRUX_CREDENTIAL_ID", "cred-prod")
    monkeypatch.setenv("CRUX_KB_ID", "kb-production")

    def fake_init(self, base_url, **kwargs):
        self.base_url = base_url

    def fake_start(self, kb_ids=None, session_id=None):
        return {"session_id": "s1", "kb_ids": list(kb_ids or [])}

    def fake_ask(self, **kwargs):
        return {
            "utopia": {"result": {"content": [{"text": "hit"}]}},
            "compressed": {"tokens_saved": 2},
        }

    monkeypatch.setattr(CruxClient, "__init__", fake_init)
    monkeypatch.setattr(CruxClient, "start_session", fake_start)
    monkeypatch.setattr(CruxClient, "ask", fake_ask)
    assert main(["ask", "write", "path"]) == 0
    out = capsys.readouterr().out
    assert "hit" in out
