"""Tests for bridge/redaction.py and bridge/secret_scan.py (keyless)."""

import os

from bridge import redaction, secret_scan


def test_email_redacted_and_counted():
    text, hits = redaction.redact_text("contact bob@example.com now")
    assert "bob@example.com" not in text
    assert "[REDACTED_EMAIL]" in text
    assert hits == {"email": 1}


def test_phone_and_ssn():
    text, hits = redaction.redact_text("call 555-123-4567, ssn 123-45-6789")
    assert "[REDACTED_PHONE]" in text and "[REDACTED_SSN]" in text
    assert hits["phone"] == 1 and hits["ssn"] == 1


def test_api_key_and_bearer():
    text, hits = redaction.redact_text(
        "key sk-abcDEF1234567890abcdef and Bearer tokXYZ9876543210tok")
    assert "sk-abcDEF" not in text
    assert hits.get("api_key") == 1 and hits.get("bearer_token") == 1


def test_credential_assignment():
    text, hits = redaction.redact_text('api_key = "supersecret12345"')
    assert "supersecret" not in text
    assert hits.get("api_key_assignment") == 1


def test_non_secret_text_untouched():
    text, hits = redaction.redact_text("List the directory tree for the project.")
    assert text == "List the directory tree for the project."
    assert hits == {}


def test_redact_qa_scrubs_fields_and_previews():
    qa = {
        "question": "Email alice@example.com the list",
        "answer": "called list_directory",
        "chain": [{
            "tool": "list_directory",
            "tool_input": {"path": "."},
            "result_preview": "sent to bob@example.com ok",
        }],
        "chain_valid": True,
    }
    clean, hits = redaction.redact_qa(qa)
    assert "alice@example.com" not in clean["question"]
    assert "bob@example.com" not in clean["chain"][0]["result_preview"]
    assert hits["email"] == 2
    assert clean["chain_valid"] is True
    # original untouched
    assert "alice@example.com" in qa["question"]


def test_redact_qa_counts_provenance():
    qa = {
        "question": "q",
        "answer": "a",
        "chain": [],
        "redactions": {"email": 3},
    }
    clean, hits = redaction.redact_qa(qa)
    assert clean["redactions"]["email"] == 3
    assert hits == {}


def test_redact_qa_non_string_fields_kept():
    qa = {"question": "q", "answer": "a", "chain": [{"tool": "t",
           "tool_input": {"n": 5}, "result_preview": 123}]}
    clean, _ = redaction.redact_qa(qa)
    assert clean["chain"][0]["tool_input"] == {"n": 5}
    assert clean["chain"][0]["result_preview"] == 123


def test_secret_scan_clean_text():
    assert secret_scan.scan_text("hello world\nno secrets here\n") == []


def test_secret_scan_finds_private_key():
    findings = secret_scan.scan_text("x\n-----BEGIN RSA PRIVATE KEY-----\nabc\n")
    assert len(findings) == 1
    assert findings[0].kind == "private_key"
    # the finding never echoes the secret
    assert "BEGIN RSA" not in findings[0].excerpt


def test_secret_scan_finds_aws_key():
    findings = secret_scan.scan_text('key = "AKIAIOSFODNN7EXAMPLE"\n')
    assert any(f.kind == "aws_access_key" for f in findings)


def test_secret_scan_finds_openrouter_key():
    findings = secret_scan.scan_text('k="sk-or-v1-abcdef1234567890abcdef1234567890ab"\n')
    assert any(f.kind == "openrouter_key" for f in findings)


def test_secret_scan_finds_bearer():
    findings = secret_scan.scan_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789\n")
    assert any(f.kind == "bearer_token" for f in findings)


def test_secret_scan_finds_password_assignment():
    findings = secret_scan.scan_text('password = "hunter22-hunter"\n')
    assert any(f.kind == "password_assignment" for f in findings)


def test_secret_scan_line_numbers():
    findings = secret_scan.scan_text("ok\nAKIAIOSFODNN7EXAMPLE\n")
    assert findings[0].line_no == 2


def test_secret_scan_directory():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "clean.txt")
        with open(clean, "w") as fh:
            fh.write("nothing here")
        dirty = os.path.join(tmp, "dirty.txt")
        with open(dirty, "w") as fh:
            fh.write('aws = "AKIAIOSFODNN7EXAMPLE"\n')
        findings = secret_scan.scan_directory(tmp)
        assert [f.path for f in findings] == [dirty]
        assert all(f.kind != "dirty" for f in findings)


def test_secret_scan_empty_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert secret_scan.scan_directory(tmp) == []
