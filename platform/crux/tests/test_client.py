"""Workflow agents talk to Crux through CruxClient, not Utopia or Headroom."""

from crux.client import CruxClient
from crux.server import serve_in_thread
from tests.test_app import _app


def test_client_tools_catalog_ask_and_session():
    app = _app()
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "incident-bot",
            "credential_id": "cred-inc",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production", "kb-staging"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    server, host, port = serve_in_thread(app)
    try:
        client = CruxClient(
            f"http://{host}:{port}",
            owner_id="owner-1",
            agent_id="incident-bot",
            credential_id="cred-inc",
        )
        tools = client.tools()
        assert "search_chunks" in tools["read"]
        catalog = client.catalog()
        assert catalog["granted_kb_ids"] == ["kb-production", "kb-staging"]
        session = client.start_session(kb_ids=["kb-production"])
        assert session["kb_ids"] == ["kb-production"]
        asked = client.ask(
            kb_id="kb-production",
            tool="search_chunks",
            arguments={"query": "outage"},
            session_id=session["session_id"],
            session_kb_ids=["kb-production"],
        )
        assert asked["compressed"]["tokens_saved"] == 7
        found = client.find_entities(
            kb_id="kb-production",
            query="outage",
            session_id=session["session_id"],
            session_kb_ids=["kb-production"],
        )
        assert found["text"] == "compressed"
        facts = client.entity_facts(
            kb_id="kb-production",
            entity_id="01a07cbf-668c-73b1-a559-1e5002f5710f",
            session_id=session["session_id"],
            session_kb_ids=["kb-production"],
        )
        assert facts["text"] == "compressed"
        ready = client.readiness()
        assert ready["product"] == "crux"
        assert "blockers" in ready
    finally:
        server.shutdown()


def test_client_remember_is_propose_and_read_only_is_denied():
    app = _app()
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "incident-bot",
            "credential_id": "cred-inc",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    server, host, port = serve_in_thread(app)
    try:
        client = CruxClient(
            f"http://{host}:{port}",
            owner_id="owner-1",
            agent_id="incident-bot",
            credential_id="cred-inc",
        )
        session = client.start_session(kb_ids=["kb-production"])
        try:
            client.remember(
                kb_id="kb-production",
                text="ship incident INC-1",
                session_id=session["session_id"],
                session_kb_ids=session["kb_ids"],
            )
            assert False, "read-only remember should fail"
        except Exception as exc:
            assert "403" in str(exc) or "propose" in str(exc).lower()
    finally:
        server.shutdown()

