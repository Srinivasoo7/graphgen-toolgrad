"""Tests for bridge/kg_builder.py (keyless — fake descriptors/llm)."""

from bridge import kg_builder
from bridge.config import ConfigError
from bridge.mcp_client import ToolDescriptor


def _descriptors():
    return [
        ToolDescriptor(
            name="list_orders",
            description="List customer orders filtered by status",
            input_schema={"properties": {
                "status": {"type": "string"},
                "customer_id": {"type": "string"},
            }},
            server="crm",
        ),
        ToolDescriptor(
            name="get_order",
            description="Get one order by its id",
            input_schema={"properties": {
                "order_id": {"type": "string"},
                "customer_id": {"type": "string"},
            }},
            server="crm",
        ),
    ]


def test_build_kg_from_tools():
    g = kg_builder.build_kg_from_tools(_descriptors())
    node_ids = set(g.nodes)
    assert "list_orders" in node_ids
    assert "get_order" in node_ids
    # shared "customer_id" parameter links the two tools
    assert g.has_edge("list_orders", "get_order") or \
        g.has_edge("get_order", "list_orders")
    assert g.nodes["list_orders"]["entity_type"] == "tool"


def test_tools_kg_node_attributes_graphgen_compatible():
    g = kg_builder.build_kg_from_tools(_descriptors())
    n = g.nodes["list_orders"]
    for attr in ("entity_name", "entity_type", "description"):
        assert attr in n


def test_build_kg_from_spec():
    spec = {
        "entities": [
            {"name": "Refund request", "type": "ticket",
             "description": "customer wants a refund"},
            {"name": "Order 9", "type": "order"},
        ],
        "relations": [
            {"head": "Refund request", "relation": "relates_to",
             "tail": "Order 9", "description": "ticket concerns order"},
        ],
    }
    g = kg_builder.build_kg_from_spec(spec)
    assert g.number_of_nodes() == 2
    assert g.has_edge("Refund request", "Order 9")


def test_spec_validation_rejects_bad_relation():
    spec = {"entities": [{"name": "A", "type": "t"}],
            "relations": [{"head": "A", "relation": "r", "tail": "nope",
                           "description": ""}]}
    # unknown endpoints are auto-created as concepts, not rejected
    g = kg_builder.build_kg_from_spec(spec)
    assert g.has_node("nope")
    # but a malformed relation is rejected
    bad = {"entities": [], "relations": [{"head": "a"}]}
    try:
        kg_builder.build_kg_from_spec(bad)
    except ConfigError as exc:
        assert "head" in str(exc) and "relation" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_spec_validation_rejects_non_mapping():
    try:
        kg_builder.build_kg_from_spec([1])
    except ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def _write_corpus(tmp, files):
    import os
    for name, text in files.items():
        with open(os.path.join(tmp, name), "w") as fh:
            fh.write(text)


def test_build_kg_from_corpus():
    import tempfile
    def fake_llm(prompt):
        assert len(prompt) > 100  # prompt carries the chunk
        return ('{"entities": [{"name": "Order", "type": "doc", '
                '"description": "a doc"}], "relations": []}')

    with tempfile.TemporaryDirectory() as tmp:
        _write_corpus(tmp, {"doc.md": "# Doc\n\nSome text about orders.\n"})
        g = kg_builder.build_kg_from_corpus(tmp, fake_llm, max_chunks=4)
    assert "Order" in g.nodes
    assert g.nodes["Order"]["entity_type"] == "doc"


def test_corpus_requires_llm():
    try:
        kg_builder.build_kg("corpus", corpus_dir="/tmp")
    except ConfigError as exc:
        assert "LLM" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_corpus_requires_dir_or_texts():
    try:
        kg_builder.build_kg("corpus", llm_fn=lambda p: "{}")
    except ConfigError as exc:
        assert "corpus_dir" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_unknown_source_rejected():
    try:
        kg_builder.build_kg("oracle")
    except ConfigError as exc:
        assert "unknown kg_source" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_load_spec_file():
    import json
    import tempfile
    spec = {"entities": [], "relations": []}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    assert kg_builder.load_spec_file(path) == spec


def test_corpus_chunking_caps_entities():
    import json
    import tempfile
    def noisy_llm(prompt):
        ents = [{"name": f"E{i}", "type": "t"} for i in range(50)]
        return '{"entities": ' + json.dumps(ents) + ', "relations": []}'

    with tempfile.TemporaryDirectory() as tmp:
        _write_corpus(tmp, {"a.md": "x " * 5000})
        g = kg_builder.build_kg_from_corpus(tmp, noisy_llm, max_chunks=8,
                                            max_entities=40)
    # node count bounded by cap even with a noisy extractor
    assert g.number_of_nodes() <= 40
