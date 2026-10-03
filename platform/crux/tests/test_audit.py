from crux.audit import FileAuditLog, build_event, event_to_public_dict


def test_no_plain_argument_values_or_hashes():
    event = build_event(
        owner_id="owner",
        agent_id="agent",
        credential_id="cred",
        session_id="sess",
        tool="search_chunks",
        kb_id="kb-production",
        status="ok",
        args={"query": "secret incident", "as_of": "2024-01-01"},
    )
    public = event_to_public_dict(event)
    blob = str(public)
    assert "secret incident" not in blob
    assert "as_of" in event.arg_keys
    assert "query" not in event.arg_keys


def test_keyed_correlation_is_hmac_not_plain_hash_of_args():
    event = build_event(
        owner_id="owner",
        agent_id="agent",
        credential_id="cred",
        session_id="sess",
        tool="remember",
        kb_id=None,
        status="ok",
        args={"text": "password123"},
        correlation_secret=b"test-secret",
    )
    assert event.correlation
    assert "password123" not in event.correlation
    assert "text" not in event.arg_keys


def test_file_audit_log_survives_reopen(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = FileAuditLog(str(path))
    first.append(
        build_event(
            owner_id="owner",
            agent_id="agent",
            credential_id="cred",
            session_id="sess",
            tool="search_chunks",
            kb_id="kb-production",
            status="ok",
            args={"query": "secret incident", "as_of": "2024-01-01"},
        )
    )
    second = FileAuditLog(str(path))
    blob = second.dump()
    assert "secret incident" not in blob
    assert second.events[0].tool == "search_chunks"
    assert "as_of" in second.events[0].arg_keys
