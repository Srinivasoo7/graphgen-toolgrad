from crux.__main__ import main
from crux.client import CruxClient


def _patch_client(monkeypatch, **methods):
    def fake_init(self, base_url, **kwargs):
        self.base_url = base_url

    monkeypatch.setenv("CRUX_URL", "http://crux.example")
    monkeypatch.setenv("CRUX_OWNER_ID", "owner-1")
    monkeypatch.setenv("CRUX_AGENT_ID", "agent-prod")
    monkeypatch.setenv("CRUX_CREDENTIAL_ID", "cred-prod")
    monkeypatch.setenv("CRUX_KB_ID", "kb-production")
    monkeypatch.setattr(CruxClient, "__init__", fake_init)
    monkeypatch.setattr(
        CruxClient,
        "start_session",
        lambda self, kb_ids=None, session_id=None: {
            "session_id": "s1",
            "kb_ids": list(kb_ids or []),
        },
    )
    for name, impl in methods.items():
        monkeypatch.setattr(CruxClient, name, impl)


def test_entities_command(monkeypatch, capsys):
    def fake_find(self, **kwargs):
        assert kwargs["query"] == "ontology"
        return {"text": "01a07cbf-668c-73b1-a559-1e5002f5710f | Azure Purview"}

    _patch_client(monkeypatch, find_entities=fake_find)
    assert main(["entities", "ontology"]) == 0
    assert "Azure Purview" in capsys.readouterr().out


def test_facts_command(monkeypatch, capsys):
    eid = "01a07cbf-668c-73b1-a559-1e5002f5710f"

    def fake_facts(self, **kwargs):
        assert kwargs["entity_id"] == eid
        return {"text": "publishes lineage"}

    _patch_client(monkeypatch, entity_facts=fake_facts)
    assert main(["facts", eid]) == 0
    assert "publishes lineage" in capsys.readouterr().out


def test_facts_requires_entity_id(capsys):
    assert main(["facts"]) == 2
    assert "usage" in capsys.readouterr().err
