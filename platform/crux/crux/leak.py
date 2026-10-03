"""Acceptance: unauthorized facts must not leak across Utopia bases.

Release A isolation is two restricted bases. Same-base object filters are later.
This module encodes the leak surfaces that must stay empty for a denied kb_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMPTY_MCP = frozenset(
    {
        "No results.",
        "No matching manual sections.",
        "No matching entities.",
        "Invalid entity_id (expected the uuid returned by find_entities).",
        "Entity not found.",
    }
)
_DOC_NAME = re.compile(
    r'"([^"]+\.(?:pdf|pptx|ppt|md|txt|docx))"', re.IGNORECASE
)
_SEARCH_HIT = re.compile(r"\[\d+\]")


LEAK_SURFACES = (
    "graph_relationships",
    "search_hit_counts",
    "summaries",
    "citations",
    "source_documents",
    "derived_facts",
)


@dataclass
class RetrievalBundle:
    graph_relationships: list[object] = field(default_factory=list)
    search_hit_counts: int = 0
    summaries: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    derived_facts: list[object] = field(default_factory=list)


class InMemoryStore:
    """Stand-in for Utopia per-base isolation in unit tests."""

    def __init__(self) -> None:
        self._by_kb: dict[str, RetrievalBundle] = {}

    def put(self, kb_id: str, bundle: RetrievalBundle) -> None:
        self._by_kb[kb_id] = bundle

    def retrieve(self, allowed_kb_ids: frozenset[str], kb_id: str) -> RetrievalBundle:
        if kb_id not in allowed_kb_ids:
            return RetrievalBundle()
        return self._by_kb.get(kb_id, RetrievalBundle())


def assert_no_leak(bundle: RetrievalBundle) -> None:
    assert bundle.graph_relationships == []
    assert bundle.search_hit_counts == 0
    assert bundle.summaries == []
    assert bundle.citations == []
    assert bundle.source_documents == []
    assert bundle.derived_facts == []


def _is_empty_mcp(text: str) -> bool:
    if not text or text in _EMPTY_MCP:
        return True
    lowered = text.lower()
    if "no matching" in lowered:
        return True
    if lowered.startswith("invalid entity"):
        return True
    if "entity not found" in lowered:
        return True
    return False


def bundle_from_mcp_text(text: str) -> RetrievalBundle:
    """Turn instrumented MCP text into leak-surface fields."""
    stripped = (text or "").strip()
    if _is_empty_mcp(stripped):
        return RetrievalBundle()
    documents = _DOC_NAME.findall(stripped)
    hits = len(_SEARCH_HIT.findall(stripped))
    if hits == 0 and documents:
        hits = 1
    relationships: list[object] = []
    derived: list[object] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if "→" in line or "->" in line:
            relationships.append(line)
            derived.append(line)
        elif " | " in line and not documents:
            derived.append(line)
    has_content = bool(documents or relationships or derived or hits)
    return RetrievalBundle(
        graph_relationships=relationships,
        search_hit_counts=hits,
        source_documents=list(documents),
        # Release A MCP text cites the same filenames; keep both surfaces.
        citations=list(documents),
        summaries=[stripped[:240]] if has_content else [],
        derived_facts=derived,
    )
