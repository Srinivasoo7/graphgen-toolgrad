"""Shared fixtures for the Phase 4 tests (keyless)."""

import networkx as nx

from bridge import kg_context_exporter, toolkg_builder
from bridge.chain_verifier import MockToolExecutor
from bridge.tests.test_toolkg_builder import _catalog, _HINTS
from bridge.tests.test_trace_shape import FABRICATED_SAMPLE


def kg_context():
    g = nx.Graph()
    g.add_node("n1", entity_name="Tesla", entity_type="ORG",
               description="Electric vehicle manufacturer")
    g.add_node("n2", entity_name="Gigafactory Texas", entity_type="FACILITY",
               description="Tesla factory in Austin producing Model Y and Cybertruck")
    g.add_node("n3", entity_name="Model Y", entity_type="PRODUCT",
               description="Tesla crossover SUV")
    g.add_edge("n1", "n2", description="Tesla operates Gigafactory Texas")
    g.add_edge("n2", "n3", description="Gigafactory Texas produces Model Y")
    return kg_context_exporter.export_kg_context(g)


def toolkg():
    return toolkg_builder.build_toolkg(_catalog(), output_hints=_HINTS)


def executor():
    ex = MockToolExecutor()
    ex.register(
        "list_directory",
        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]},
        lambda ti: {"files": ["gigafactory_notes.txt"]},
    )
    ex.register(
        "read_text_file",
        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]},
        lambda ti: "Gigafactory Texas produces Model Y and Cybertruck.",
    )
    return ex


def sample():
    return FABRICATED_SAMPLE
