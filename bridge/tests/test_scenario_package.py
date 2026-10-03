"""Tests for bridge/models/scenario_package.py + bridge/exporters (keyless)."""

import json
import os
import tempfile
from contextlib import contextmanager

from bridge import chain_verifier
from bridge.chain_verifier import MockToolExecutor
from bridge.exporters import copilot_eval
from bridge.models.scenario_package import (
    ApprovalRecord,
    ScenarioPackage,
)


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _qa(category="normal", failure_mode=None):
    return {
        "question": "Create a ticket for the VPN outage",
        "answer_draft": 'Created: {"id": "INC-1042", "status": "New"}',
        "required_tools": ["create_ticket"],
        "chain": [{
            "tool": "create_ticket",
            "tool_input": {"title": "VPN outage"},
            "result_preview": '{"id": "INC-1042", "status": "New"}',
        }],
        "entity_refs": ["Ticket"],
        "business_rules": ["Tickets must have an owner before closing"],
        "expected_outcomes": [
            {"description": "ticket created with status New",
             "assertion": "New",
             "state_after": "create_ticket applied: status=New"},
        ],
        "category": category,
        "failure_mode": failure_mode,
        "provenance": {"source": "toolgrad_trace", "chain_id": "chain_0"},
    }


def _report():
    from bridge.trace_to_qa import derive_expected_outcomes

    ex = MockToolExecutor()
    ex.register(
        "create_ticket",
        {"type": "object",
         "properties": {"title": {"type": "string"}},
         "required": ["title"]},
        lambda ti: {"id": "INC-1042", "status": "New"},
    )
    qa = _qa()
    qa["expected_outcomes"] = derive_expected_outcomes(qa["chain"])
    # strip keys the verifier does not know; keep the contract intact
    qa = {k: qa[k] for k in (
        "question", "answer_draft", "required_tools", "chain",
        "entity_refs", "expected_outcomes", "provenance")}
    return chain_verifier.verify_qa(qa, ex)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def test_from_qa_carries_intent_chain_outcomes_and_verification():
    pkg = ScenarioPackage.from_qa(_qa(), _report())
    assert pkg.scenario_id == "chain_0"
    assert pkg.intent == "Create a ticket for the VPN outage"
    assert pkg.intent_type == "create"
    assert pkg.category == "normal"
    assert pkg.entities == ["Ticket"]
    assert pkg.business_rules == ["Tickets must have an owner before closing"]
    assert pkg.tool_chain[0]["tool"] == "create_ticket"
    assert pkg.expected_outcomes[0].assertion == "New"
    assert pkg.expected_answer.startswith("Created:")
    assert pkg.verification.safe_to_review is True
    assert pkg.verification.intent_aligned is True
    assert pkg.approvals == []  # filled by the Crux adapter later


def test_from_qa_with_no_report_stays_unverified():
    pkg = ScenarioPackage.from_qa(_qa(), None)
    assert pkg.verification.safe_to_review is False
    assert pkg.scenario_id == "chain_0"


def test_category_must_be_known():
    with _raises(ValueError):
        ScenarioPackage(scenario_id="x", intent="y", category="chaos")


def test_dict_round_trip():
    pkg = ScenarioPackage.from_qa(_qa(category="security", failure_mode="authorization_denied"), _report())
    pkg.approvals.append(ApprovalRecord(
        approver="sme@acme", role="business", decision="approved",
        evidence_id="crux:ev-123"))
    restored = ScenarioPackage.from_dict(pkg.as_dict())
    assert restored.category == "security"
    assert restored.failure_mode == "authorization_denied"
    assert restored.expected_outcomes[0].state_after == "create_ticket applied: status=New"
    assert restored.verification.safe_to_review is True
    assert restored.approvals[0].evidence_id == "crux:ev-123"


# ---------------------------------------------------------------------------
# Copilot eval exporter
# ---------------------------------------------------------------------------


def test_eval_row_schema():
    row = copilot_eval.eval_row(ScenarioPackage.from_qa(_qa(), _report()))
    assert row["scenario_id"] == "chain_0"
    assert row["prompt"] == "Create a ticket for the VPN outage"
    assert row["expected_answer"].startswith("Created:")
    assert row["expected_tool_calls"] == [
        {"tool": "create_ticket", "arguments": {"title": "VPN outage"}}
    ]
    assert row["expected_outcomes"] == ["New"]
    assert row["business_rules"] == ["Tickets must have an owner before closing"]
    gt = row["ground_truth"]
    assert gt["intent_type"] == "create"
    assert gt["verified"] is True
    assert gt["checks"]["intent_aligned"] is True
    assert gt["checks"]["answer_faithful"] is True


def test_eval_row_carries_failure_category():
    pkg = ScenarioPackage.from_qa(
        _qa(category="security", failure_mode="authorization_denied"), _report())
    row = copilot_eval.eval_row(pkg)
    assert row["category"] == "security"
    assert row["failure_mode"] == "authorization_denied"


def test_to_jsonl_is_parseable_and_one_row_per_line():
    pkgs = [
        ScenarioPackage.from_qa(_qa(), _report()),
        ScenarioPackage.from_qa(_qa(category="recovery", failure_mode="tool_error"), _report()),
    ]
    out = copilot_eval.to_jsonl(pkgs)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2
    rows = [json.loads(ln) for ln in lines]
    assert rows[0]["scenario_id"] == "chain_0"
    assert rows[1]["category"] == "recovery"
    assert copilot_eval.to_jsonl([]) == ""


def test_write_jsonl_writes_file():
    pkg = ScenarioPackage.from_qa(_qa(), _report())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "eval.jsonl")
        assert copilot_eval.write_jsonl([pkg], path) == path
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    assert len(rows) == 1 and rows[0]["prompt"]
