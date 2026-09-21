"""Tests for bridge/toolkg_builder.py (keyless)."""

import json
import os
import tempfile
from types import SimpleNamespace

import networkx as nx

from bridge import toolkg_builder


def _fake_tool(name, description, properties):
    return SimpleNamespace(
        name=name,
        description=description,
        args_schema={"properties": properties},
    )


def _catalog():
    return [
        _fake_tool(
            "list_directory",
            "Get a detailed listing of all files and directories in a specified path.",
            {"path": {"type": "string"}},
        ),
        _fake_tool(
            "directory_tree",
            "Get a recursive tree view of files and directories as a JSON structure.",
            {"path": {"type": "string"}},
        ),
        _fake_tool(
            "read_text_file",
            "Read the complete contents of a file from the file system as text.",
            {"path": {"type": "string"}},
        ),
        _fake_tool(
            "read_multiple_files",
            "Read the contents of multiple files simultaneously.",
            {"paths": {"type": "array", "items": {"type": "string"}}},
        ),
        _fake_tool(
            "search_files",
            "Search for files matching a filename pattern under a directory.",
            {"pattern": {"type": "string"}, "path": {"type": "string"}},
        ),
        _fake_tool(
            "get_file_info",
            "Return metadata for a single file.",
            {"file_path": {"type": "string"}},
        ),
    ]


_HINTS = {
    "list_directory": [("path", "path")],
    "directory_tree": [("path", "path")],
    "read_text_file": [("content", "string")],
    "read_multiple_files": [("content", "string")],
    "search_files": [("path", "path")],
    "get_file_info": [("metadata", "object")],
}


def _build(threshold=toolkg_builder.DEFAULT_THRESHOLD):
    return toolkg_builder.build_toolkg(_catalog(), output_hints=_HINTS,
                                       threshold=threshold)


def test_expected_edges_present():
    g = _build()
    assert g.has_edge("list_directory", "read_text_file")
    assert g["list_directory"]["read_text_file"]["score"] == 1.0
    assert g["list_directory"]["read_text_file"]["via"] == "path->path"
    # plural fold: out "path" matches in "paths"
    assert g.has_edge("list_directory", "read_multiple_files")
    assert g["list_directory"]["read_multiple_files"]["via"] == "path->paths"
    assert g.has_edge("search_files", "read_text_file")
    # partial token overlap still clears the default threshold
    assert g.has_edge("list_directory", "get_file_info")
    assert g["list_directory"]["get_file_info"]["via"] == "path->file_path"


def test_no_self_loops():
    g = _build()
    assert not any(u == v for u, v in g.edges())


def test_directedness_and_negatives():
    g = _build()
    # content-producing tools feed nothing in this catalog
    assert not g.has_edge("read_text_file", "list_directory")
    assert not g.has_edge("read_text_file", "read_multiple_files")
    assert not g.has_edge("read_multiple_files", "search_files")
    assert not g.has_edge("get_file_info", "read_text_file")


def test_threshold_filters_partial_matches():
    g_lo = _build(threshold=0.34)
    g_hi = _build(threshold=0.9)
    # 0.817 edge survives 0.34 but not 0.9
    assert g_lo.has_edge("list_directory", "get_file_info")
    assert not g_hi.has_edge("list_directory", "get_file_info")
    # exact matches survive both
    assert g_hi.has_edge("list_directory", "read_text_file")


def test_description_inference_without_hints():
    g = toolkg_builder.build_toolkg(_catalog())  # no output_hints
    # "files", "directories", "path" in the description -> inferred ("path","path")
    assert g.has_edge("list_directory", "read_text_file")


def test_node_attributes_follow_graphgen_conventions():
    g = _build()
    node = g.nodes["read_text_file"]
    assert node["entity_name"] == "read_text_file"
    assert node["entity_type"] == "tool"
    assert "complete contents" in node["description"]
    inputs = json.loads(node["inputs_json"])
    assert {p["name"] for p in inputs} == {"path"}
    outputs = json.loads(node["outputs_json"])
    assert outputs == [{"name": "content", "type": "string"}]


def test_determinism():
    g1, g2 = _build(), _build()
    assert set(g1.edges()) == set(g2.edges())
    assert all(g1[u][v]["score"] == g2[u][v]["score"] for u, v in g1.edges())


def _edge_signature(g):
    return {
        (u, v): (float(d["score"]), d["via"], d["relation"])
        for u, v, d in g.edges(data=True)
    }


def test_graphml_roundtrip():
    g = _build()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "toolkg.graphml")
        toolkg_builder.save_toolkg(g, p)
        g2 = toolkg_builder.load_toolkg(p)
    assert set(g2.nodes()) == set(g.nodes())
    assert _edge_signature(g2) == _edge_signature(g)
    assert g2.nodes["read_text_file"]["entity_type"] == "tool"


def test_json_roundtrip():
    g = _build()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "toolkg.json")
        toolkg_builder.save_toolkg(g, p)
        g2 = toolkg_builder.load_toolkg(p)
    assert set(g2.nodes()) == set(g.nodes())
    assert _edge_signature(g2) == _edge_signature(g)


def test_stats():
    g = _build()
    s = toolkg_builder.toolkg_stats(g)
    assert s["num_tools"] == 6
    assert s["num_edges"] == g.number_of_edges() > 0
    assert 0.0 < s["density"] < 1.0


def test_generic_only_overlap_is_penalized():
    # ticket_id -> article_id shares only the generic token "id": weak
    # evidence, must not clear the default threshold.
    score, _ = toolkg_builder.composability_score(
        [("ticket_id", "string")], [("article_id", "string")])
    assert score < toolkg_builder.DEFAULT_THRESHOLD
    # ...while a specific shared token keeps the full score + type bonus.
    score, pair = toolkg_builder.composability_score(
        [("ticket_id", "string")], [("ticket_id", "string")])
    assert score == 1.0
    assert pair == ("ticket_id", "ticket_id")


def test_generic_only_match_scores_half_dice():
    # title -> title: Dice 1.0 halved to 0.5 — a weak but defensible edge,
    # gone under a stricter threshold.
    score, _ = toolkg_builder.composability_score(
        [("title", "string")], [("title", "string")])
    assert score == 0.5


def _itsm_catalog():
    return [
        _fake_tool("search_kb", "Search the IT knowledge base.",
                   {"query": {"type": "string"}}),
        _fake_tool("get_kb_article",
                   "Fetch the full text of a knowledge-base article by id.",
                   {"article_id": {"type": "string"}}),
        _fake_tool("create_ticket", "Open a service-desk ticket.",
                   {"title": {"type": "string"},
                    "description": {"type": "string"}}),
        _fake_tool("get_ticket", "Fetch a ticket by id.",
                   {"ticket_id": {"type": "string"}}),
    ]


_ITSM_HINTS = {
    "search_kb": [("article_id", "string")],
    "get_kb_article": [("article_id", "string")],
    "create_ticket": [("ticket_id", "string")],
    "get_ticket": [("ticket_id", "string")],
}


def test_itsm_hints_yield_composable_edges():
    # Regression test for the 2026-09-20 ITSM run: without hints the
    # description lexicon inferred nothing and the ToolKG had zero edges.
    g = toolkg_builder.build_toolkg(_itsm_catalog(), output_hints=_ITSM_HINTS)
    assert g.has_edge("search_kb", "get_kb_article")
    assert g["search_kb"]["get_kb_article"]["via"] == "article_id->article_id"
    assert g.has_edge("create_ticket", "get_ticket")
    assert g["create_ticket"]["get_ticket"]["via"] == "ticket_id->ticket_id"
    # ...and the bogus cross-id matches stay out.
    assert not g.has_edge("create_ticket", "get_kb_article")
    assert not g.has_edge("search_kb", "get_ticket")
