"""Live two-base isolation. Skipped unless CRUX_LIVE=1 and the stack is up."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from crux.leak import assert_no_leak, bundle_from_mcp_text
from crux.measure import first_entity_id

LIVE = os.environ.get("CRUX_LIVE") == "1"
STATE = Path("data/crux/bootstrap.json")

_STAGING_ONLY = (
    "AI_Engineering.pptx",
    "AI Engineering (EducationEvent)",
    "Claude offerings",
)
_PROD_ONLY = ("Azure Purview", "Ontology SDK", "TRACE Eval")


def test_empty_mcp_text_is_not_a_leak():
    assert_no_leak(bundle_from_mcp_text("No results."))
    assert_no_leak(bundle_from_mcp_text("No matching manual sections."))
    assert_no_leak(bundle_from_mcp_text("No matching entities."))
    assert_no_leak(bundle_from_mcp_text("Entity not found."))


def test_hit_text_records_source_document():
    text = (
        '[1] "AI_Engineering.pptx" section 1 (document_id: abc):\nSESSION AGENDA'
    )
    bundle = bundle_from_mcp_text(text)
    assert "AI_Engineering.pptx" in bundle.source_documents
    assert bundle.search_hit_counts >= 1


@pytest.mark.skipif(not LIVE or not STATE.exists(), reason="live stack not requested")
def test_live_restricted_bases_do_not_leak_each_other():
    state = json.loads(STATE.read_text(encoding="utf-8"))
    prod_kb = state["production_kb_id"]
    staging_kb = state["staging_kb_id"]

    prod_search = _ask("agent-prod", "cred-prod", prod_kb, "AI Engineering slides")
    staging_search = _ask(
        "agent-staging",
        "cred-staging",
        staging_kb,
        "ontology bidirectional write path",
    )
    general = _ask(
        "agent-prod",
        "cred-prod",
        prod_kb,
        "TRACE Eval intervention calibrated",
    )
    prod_bundle = bundle_from_mcp_text(prod_search)
    staging_bundle = bundle_from_mcp_text(staging_search)
    assert "AI_Engineering.pptx" not in prod_bundle.source_documents
    assert "Ontology" not in " ".join(staging_bundle.source_documents)
    assert_no_leak(bundle_from_mcp_text(general))

    prod_entities = _ask(
        "agent-prod",
        "cred-prod",
        prod_kb,
        tool="find_entities",
        arguments={"query": "AI Engineering slides"},
    )
    staging_entities = _ask(
        "agent-staging",
        "cred-staging",
        staging_kb,
        tool="find_entities",
        arguments={"query": "Azure Purview ontology"},
    )
    _assert_markers_absent(prod_entities, _STAGING_ONLY)
    _assert_markers_absent(staging_entities, _PROD_ONLY)

    prod_docs = _ask(
        "agent-prod",
        "cred-prod",
        prod_kb,
        tool="search_docs",
        arguments={"query": "AI Engineering slides"},
    )
    staging_docs = _ask(
        "agent-staging",
        "cred-staging",
        staging_kb,
        tool="search_docs",
        arguments={"query": "ontology bidirectional write path"},
    )
    _assert_markers_absent(prod_docs, _STAGING_ONLY)
    _assert_markers_absent(staging_docs, _PROD_ONLY)

    prod_eid = first_entity_id(prod_entities)
    staging_eid = first_entity_id(staging_entities)
    if prod_eid:
        prod_facts = _ask(
            "agent-prod",
            "cred-prod",
            prod_kb,
            tool="entity_facts",
            arguments={"entity_id": prod_eid},
        )
        _assert_markers_absent(prod_facts, _STAGING_ONLY)
        facts_bundle = bundle_from_mcp_text(prod_facts)
        assert facts_bundle.graph_relationships or facts_bundle.derived_facts
    if staging_eid:
        staging_facts = _ask(
            "agent-staging",
            "cred-staging",
            staging_kb,
            tool="entity_facts",
            arguments={"entity_id": staging_eid},
        )
        _assert_markers_absent(staging_facts, _PROD_ONLY)

    invalid_facts = _ask(
        "agent-prod",
        "cred-prod",
        prod_kb,
        tool="entity_facts",
        arguments={"entity_id": "00000000-0000-0000-0000-000000000001"},
    )
    assert_no_leak(bundle_from_mcp_text(invalid_facts))

    with pytest.raises(urllib.error.HTTPError) as denied:
        _ask(
            "agent-prod",
            "cred-prod",
            staging_kb,
            "anything",
            expect_error=True,
        )
    assert denied.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as denied_facts:
        _ask(
            "agent-prod",
            "cred-prod",
            staging_kb,
            tool="entity_facts",
            arguments={"entity_id": "00000000-0000-0000-0000-000000000001"},
            expect_error=True,
        )
    assert denied_facts.value.code == 403


def _assert_markers_absent(text: str, markers: tuple[str, ...]) -> None:
    lowered = text.lower()
    for marker in markers:
        assert marker.lower() not in lowered


def _ask(
    agent: str,
    cred: str,
    kb_id: str,
    query: str | None = None,
    *,
    tool: str = "search_chunks",
    arguments: dict[str, Any] | None = None,
    expect_error: bool = False,
) -> str:
    body = {
        "owner_id": "owner-1",
        "agent_id": agent,
        "credential_id": cred,
        "session_id": "live-leak",
        "kb_id": kb_id,
        "tool": tool,
        "arguments": arguments if arguments is not None else {"query": query},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8788/v1/ask",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if expect_error:
        with urllib.request.urlopen(req, timeout=30):
            return ""
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["utopia"]["result"]["content"][0]["text"]
