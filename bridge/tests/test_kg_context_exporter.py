"""Unit tests for bridge.kg_context_exporter (no API keys, no Ray)."""

import json

import networkx as nx

from bridge.kg_context_exporter import (
    ExportConfig,
    estimate_tokens,
    export_kg_context,
    export_kg_context_json,
    render_kg_context,
)


def _tiny_kg() -> nx.Graph:
    """Hand-built KG in GraphGen's light_rag attribute schema."""
    g = nx.Graph()
    g.add_node(
        "Tesla",
        entity_name="Tesla",
        entity_type="ORGANIZATION",
        description="American electric vehicle and clean energy company founded by Elon Musk.",
        source_id="doc1",
    )
    g.add_node(
        "Model 3",
        entity_name="Model 3",
        entity_type="PRODUCT",
        description="Tesla's mass-market electric sedan.",
        source_id="doc1",
    )
    g.add_node(
        "Elon Musk",
        entity_name="Elon Musk",
        entity_type="PERSON",
        description="CEO of Tesla and SpaceX.",
        source_id="doc2",
    )
    g.add_node(
        "Gigafactory",
        entity_name="Gigafactory",
        entity_type="FACILITY",
        description="Tesla's large-scale battery and vehicle factories.",
        source_id="doc2",
    )
    g.add_node(
        "Lithium",
        entity_name="Lithium",
        entity_type="MATERIAL",
        description="Key raw material for EV batteries.",
        source_id="doc3",
    )
    # Edge WITH an explicit relation label.
    g.add_edge(
        "Tesla",
        "Model 3",
        relation="manufactures",
        description="Tesla manufactures the Model 3 sedan.",
        source_id="doc1",
    )
    # Edges WITHOUT a relation label (GraphGen's light_rag builder emits
    # none) — relation must fall back to the description.
    g.add_edge(
        "Elon Musk",
        "Tesla",
        description="Elon Musk is the CEO of Tesla.",
        source_id="doc2",
    )
    g.add_edge(
        "Tesla",
        "Gigafactory",
        description="Tesla operates Gigafactories.",
        source_id="doc2",
    )
    g.add_edge(
        "Lithium",
        "Gigafactory",
        description="Gigafactories consume lithium for batteries.",
        source_id="doc3",
    )
    return g


def test_export_shape():
    ctx = export_kg_context(_tiny_kg())
    assert set(ctx) == {"entities", "triples", "communities", "meta"}, set(ctx)
    assert ctx["meta"]["node_count"] == 5
    assert ctx["meta"]["edge_count"] == 4
    for e in ctx["entities"]:
        assert set(e) >= {"name", "entity_type", "description", "degree"}, set(e)
    for t in ctx["triples"]:
        assert set(t) >= {"head", "relation", "tail", "description"}, set(t)
    for c in ctx["communities"]:
        assert set(c) >= {"id", "size", "members", "summary"}, set(c)
    # Most-connected node ranks first (Tesla has degree 3).
    assert ctx["entities"][0]["name"] == "Tesla"
    assert ctx["entities"][0]["degree"] == 3


def test_budget_caps():
    cfg = ExportConfig(max_entities=2, max_triples=1, max_communities=1,
                       max_tokens=10_000)
    ctx = export_kg_context(_tiny_kg(), cfg)
    assert len(ctx["entities"]) == 2
    assert len(ctx["triples"]) == 1
    assert len(ctx["communities"]) == 1
    assert ctx["meta"]["exported_entities"] == 2


def test_token_budget_enforced():
    cfg = ExportConfig(max_tokens=120)
    ctx = export_kg_context(_tiny_kg(), cfg)
    text = render_kg_context(ctx)
    assert estimate_tokens(text) <= cfg.max_tokens, text
    assert ctx["meta"]["estimated_tokens"] <= cfg.max_tokens


def test_relation_fallback():
    ctx = export_kg_context(_tiny_kg(), ExportConfig(max_tokens=10_000))
    by_pair = {(t["head"], t["tail"]): t for t in ctx["triples"]}
    # Explicit label wins.
    assert by_pair[("Tesla", "Model 3")]["relation"] == "manufactures"
    # Missing label falls back to description text.
    musk_tesla = by_pair.get(("Elon Musk", "Tesla")) or by_pair.get(("Tesla", "Elon Musk"))
    assert musk_tesla is not None
    assert "CEO" in musk_tesla["relation"]


def test_render_contains_sections():
    ctx = export_kg_context(_tiny_kg())
    text = render_kg_context(ctx)
    assert "Tesla" in text
    assert "manufactures" in text
    assert "Domain entities:" in text
    assert "Key relationships:" in text
    assert "Topic clusters:" in text


def test_json_serializable_and_sorted():
    blob = export_kg_context_json(_tiny_kg())
    parsed = json.loads(blob)  # must not raise
    assert parsed["meta"]["node_count"] == 5


def test_empty_graph():
    ctx = export_kg_context(nx.Graph())
    assert ctx["entities"] == []
    assert ctx["triples"] == []
    assert ctx["communities"] == []
    assert render_kg_context(ctx).strip() != ""  # headers still render


def test_missing_attrs_degrade_gracefully():
    g = nx.Graph()
    g.add_node("bare_node")  # no attributes at all
    g.add_node("other")
    g.add_edge("bare_node", "other")  # no attributes at all
    ctx = export_kg_context(g)
    assert ctx["entities"][0]["name"] == "bare_node"
    assert ctx["triples"][0]["relation"] == "related_to"
