"""Hold corpus and questions constant. Compression-off cells are not a caller flag."""

from __future__ import annotations

import re
from typing import Any


_ENTITY_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_NO_GRAPH = "no_graph_coverage"
_METHOD = (
    "A and C are tokens_before from the same authorized compress-on ask. "
    "Not a second HTTP call and not a caller bypass flag."
)


def first_entity_id(text: str) -> str | None:
    """First find_entities row id (pipe column 0). None if the list is empty."""
    if not text or "no matching" in text.lower():
        return None
    for line in text.splitlines():
        candidate = line.split("|", 1)[0].strip()
        match = _ENTITY_ID.fullmatch(candidate)
        if match:
            return match.group(0)
    return None


def _ask_text(payload: dict[str, Any]) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text:
        return text
    try:
        return payload["utopia"]["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _tokens(payload: dict[str, Any]) -> dict[str, Any]:
    compressed = payload.get("compressed") or {}
    return {
        "tokens_before": compressed.get("tokens_before"),
        "tokens_after": compressed.get("tokens_after"),
        "tokens_saved": compressed.get("tokens_saved"),
        "skipped": compressed.get("skipped"),
    }


def _off_cell(on_cell: dict[str, Any] | None) -> dict[str, Any] | None:
    if not on_cell:
        return None
    return {
        "tokens": on_cell.get("tokens_before"),
        "derived_from": "tokens_before",
        "no_caller_bypass": True,
    }


def measure_cells(client: Any, *, kb_id: str, query: str) -> dict[str, Any]:
    session = client.start_session(kb_ids=[kb_id])
    session_id = session["session_id"]
    session_kbs = session["kb_ids"]
    document = client.ask(
        kb_id=kb_id,
        tool="search_chunks",
        arguments={"query": query},
        session_id=session_id,
        session_kb_ids=session_kbs,
    )
    entities = client.ask(
        kb_id=kb_id,
        tool="find_entities",
        arguments={"query": query},
        session_id=session_id,
        session_kb_ids=session_kbs,
    )
    entity_id = first_entity_id(_ask_text(entities))
    skipped: dict[str, str] = {}
    on_document = _tokens(document)
    cells: dict[str, Any] = {
        "B_document_compress_on": on_document,
        "A_document_compress_off": _off_cell(on_document),
    }
    if entity_id is None:
        cells["D_graph_compress_on"] = None
        cells["C_graph_compress_off"] = _off_cell(None)
        skipped["D_graph_compress_on"] = _NO_GRAPH
        skipped["C_graph_compress_off"] = _NO_GRAPH
        graph = {"entity_id": None, "fallback": "document"}
    else:
        facts = client.ask(
            kb_id=kb_id,
            tool="entity_facts",
            arguments={"entity_id": entity_id},
            session_id=session_id,
            session_kb_ids=session_kbs,
        )
        on_graph = _tokens(facts)
        cells["D_graph_compress_on"] = on_graph
        cells["C_graph_compress_off"] = _off_cell(on_graph)
        graph = {"entity_id": entity_id}
    return {
        "hypothesis": True,
        "query": query,
        "kb_id": kb_id,
        "graph": graph,
        "cells": cells,
        "skipped": skipped,
        "method": _METHOD,
    }
