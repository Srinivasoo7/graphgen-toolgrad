from crux.__main__ import main
from crux.client import CruxClient


def test_propose_command(monkeypatch, capsys):
    monkeypatch.setenv("CRUX_URL", "http://crux.example")
    monkeypatch.setenv("CRUX_OWNER_ID", "owner-1")
    monkeypatch.setenv("CRUX_AGENT_ID", "agent-staging")
    monkeypatch.setenv("CRUX_CREDENTIAL_ID", "cred-staging")
    monkeypatch.setenv("CRUX_KB_ID", "kb-staging")

    def fake_init(self, base_url, **kwargs):
        self.base_url = base_url

    monkeypatch.setattr(CruxClient, "__init__", fake_init)
    monkeypatch.setattr(
        CruxClient,
        "start_session",
        lambda self, kb_ids=None, session_id=None: {
            "session_id": "s1",
            "kb_ids": list(kb_ids or []),
        },
    )
    monkeypatch.setattr(
        CruxClient,
        "remember",
        lambda self, **kwargs: {"text": "proposed", "utopia": {"ok": True}},
    )
    assert main(["propose", "owner", "Alice", "runs", "staging"]) == 0
    captured = capsys.readouterr()
    assert "proposed" in captured.out
    assert "pending Review" in captured.err
    assert "not a live graph edge" in captured.err


def test_propose_prints_compressed_text_when_envelope_lacks_text(monkeypatch, capsys):
    monkeypatch.setenv("CRUX_URL", "http://crux.example")
    monkeypatch.setenv("CRUX_OWNER_ID", "owner-1")
    monkeypatch.setenv("CRUX_AGENT_ID", "agent-staging")
    monkeypatch.setenv("CRUX_CREDENTIAL_ID", "cred-staging")
    monkeypatch.setenv("CRUX_KB_ID", "kb-staging")

    def fake_init(self, base_url, **kwargs):
        self.base_url = base_url

    monkeypatch.setattr(CruxClient, "__init__", fake_init)
    monkeypatch.setattr(
        CruxClient,
        "start_session",
        lambda self, kb_ids=None, session_id=None: {
            "session_id": "s1",
            "kb_ids": list(kb_ids or []),
        },
    )
    monkeypatch.setattr(
        CruxClient,
        "remember",
        lambda self, **kwargs: {
            "compressed": {"messages": [{"content": "Recorded the sentence"}]},
        },
    )
    assert main(["propose", "hello"]) == 0
    captured = capsys.readouterr()
    assert "Recorded the sentence" in captured.out
    assert "pending Review" in captured.err


def test_audit_command(monkeypatch, capsys):
    monkeypatch.setenv("CRUX_URL", "http://crux.example")
    monkeypatch.setenv("CRUX_ADMIN_TOKEN", "admin")

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"events": [{"tool": "search_chunks", "arg_keys": ["as_of"]}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers=None):
            assert url.endswith("/v1/audit")
            assert headers["Authorization"] == "Bearer admin"
            return FakeResp()

    monkeypatch.setattr("httpx.Client", FakeClient)
    assert main(["audit"]) == 0
    assert "search_chunks" in capsys.readouterr().out
