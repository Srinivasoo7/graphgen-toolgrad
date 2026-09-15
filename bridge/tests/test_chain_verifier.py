"""Tests for bridge/chain_verifier.py (keyless)."""

import networkx as nx

from bridge import chain_verifier
from bridge.chain_verifier import MockToolExecutor, ToolGradExecutor


def _executor():
    ex = MockToolExecutor()
    ex.register(
        "list_directory",
        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]},
        lambda ti: {"files": ["a.txt", "b.txt"]},
    )
    ex.register(
        "read_text_file",
        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]},
        lambda ti: "file contents here",
    )
    return ex


def _qa(steps):
    return {
        "question": "Q?",
        "answer_draft": "A.",
        "required_tools": [s["tool"] for s in steps],
        "chain": steps,
        "entity_refs": [],
        "provenance": {},
    }


_GOOD_STEPS = [
    {"tool": "list_directory", "tool_input": {"path": "/data"}, "result_preview": "..."},
    {"tool": "read_text_file", "tool_input": {"path": "/data/a.txt"}, "result_preview": "..."},
]


def test_valid_chain_passes():
    report = chain_verifier.verify_qa(_qa(_GOOD_STEPS), _executor())
    assert report["chain_valid"] is True
    assert report["missing_tools"] == []
    assert report["schema_mismatches"] == []
    assert all(s["ok"] for s in report["executed_steps"])


def test_missing_tool_fails_without_raising():
    steps = [{"tool": "no_such_tool", "tool_input": {}, "result_preview": "..."}]
    report = chain_verifier.verify_qa(_qa(steps), _executor())
    assert report["chain_valid"] is False
    assert report["missing_tools"] == ["no_such_tool"]
    assert report["executed_steps"][0]["ok"] is False


def test_missing_required_property_fails():
    steps = [{"tool": "list_directory", "tool_input": {}, "result_preview": "..."}]
    report = chain_verifier.verify_qa(_qa(steps), _executor())
    assert report["chain_valid"] is False
    assert len(report["schema_mismatches"]) == 1
    mismatch = report["schema_mismatches"][0]
    assert mismatch["tool"] == "list_directory"
    assert any("path" in p for p in mismatch["problems"])


def test_wrong_property_type_fails():
    steps = [{"tool": "read_text_file", "tool_input": {"path": 123}, "result_preview": "..."}]
    report = chain_verifier.verify_qa(_qa(steps), _executor())
    assert report["chain_valid"] is False
    assert report["schema_mismatches"][0]["problems"]


def test_executor_exception_is_recorded_not_raised():
    ex = MockToolExecutor()

    def boom(tool_input):
        raise RuntimeError("disk on fire")

    ex.register("fragile", {"type": "object"}, boom)
    steps = [{"tool": "fragile", "tool_input": {}, "result_preview": "..."}]
    report = chain_verifier.verify_qa(_qa(steps), ex)
    assert report["chain_valid"] is False
    assert report["executed_steps"][0]["ok"] is False
    assert "RuntimeError" in report["executed_steps"][0]["error"]


def test_empty_chain_is_invalid():
    report = chain_verifier.verify_qa(_qa([]), _executor())
    assert report["chain_valid"] is False


def test_toolgrad_executor_schema_from_pydantic():
    # Live-path wrapper: schema comes from a pydantic args_schema, keyless.
    from pydantic import BaseModel, Field

    class Args(BaseModel):
        path: str = Field(description="file path")
        limit: int = 10

    class FakeTool:
        name = "fake_read"
        args_schema = Args

        def invoke(self, tool_input):
            return f"read {tool_input['path']}"

    ex = ToolGradExecutor({"fake_read": FakeTool()})
    assert ex.has_tool("fake_read")
    assert not ex.has_tool("other")
    schema = ex.input_schema("fake_read")
    assert "path" in schema.get("properties", {})
    assert ex.call("fake_read", {"path": "/x"}) == "read /x"

    qa = _qa([{"tool": "fake_read", "tool_input": {"path": "/x"},
               "result_preview": "..."}])
    report = chain_verifier.verify_qa(qa, ex)
    assert report["chain_valid"] is True

    qa_bad = _qa([{"tool": "fake_read", "tool_input": {"nope": 1},
                   "result_preview": "..."}])
    assert chain_verifier.verify_qa(qa_bad, ex)["chain_valid"] is False


def test_chain_toolkg_coverage():
    g = nx.DiGraph()
    g.add_edge("list_directory", "read_text_file")
    g.add_edge("read_text_file", "search_files")
    steps = [{"tool": "list_directory"}, {"tool": "read_text_file"},
             {"tool": "search_files"}]
    assert chain_verifier.chain_toolkg_coverage(steps, g) == 1.0
    assert chain_verifier.chain_toolkg_coverage(["list_directory", "search_files"], g) == 0.0
    assert chain_verifier.chain_toolkg_coverage(["only_one"], g) == 0.0
    assert chain_verifier.chain_toolkg_coverage(steps, None) == 0.0
