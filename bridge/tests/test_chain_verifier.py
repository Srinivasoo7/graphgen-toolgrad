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
    assert chain_verifier.chain_toolkg_coverage(["only_one"], g) == 1.0
    assert chain_verifier.chain_toolkg_coverage([], g) == 0.0
    assert chain_verifier.chain_toolkg_coverage(steps, None) == 0.0


# --- Async-aware executor tests (the live-run verifier fix, done properly) ---

import asyncio


class _AsyncOnlyMCPTool:
    """Mimics the fork's ``_wrap_with_path_prefix`` output for async-only MCP tools:

    sync ``func`` is built over the original tool's ``func`` which is None,
    so sync ``invoke`` dies with TypeError while the ``coroutine`` path works.
    """

    def __init__(self):
        self.func = None
        self.coroutine = self._run
        self.args_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        self.calls = []

    async def _run(self, tool_input):
        self.calls.append(tool_input)
        return {"async": True, "input": tool_input}

    async def ainvoke(self, tool_input):
        return await self._run(tool_input)

    def invoke(self, tool_input):
        raise TypeError("'NoneType' object is not callable")


class _SyncOnlyTool:
    def __init__(self):
        self.func = self._run
        self.coroutine = None
        self.args_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def _run(self, tool_input):
        return {"sync": True}

    def invoke(self, tool_input):
        return self._run(tool_input)


def _async_executor():
    return ToolGradExecutor({"mcp_ls": _AsyncOnlyMCPTool()})


def test_async_only_tool_sync_call_uses_coroutine_path():
    ex = _async_executor()
    result = ex.call("mcp_ls", {"path": "."})
    assert result == {"async": True, "input": {"path": "."}}


def test_async_only_tool_call_inside_running_loop():
    # Sync call() from inside a running loop must thread-hop, not deadlock.
    ex = _async_executor()

    async def main():
        return ex.call("mcp_ls", {"path": "."})

    result = asyncio.run(main())
    assert result["async"] is True


def test_acall_async_only_tool():
    ex = _async_executor()

    async def main():
        return await ex.acall("mcp_ls", {"path": "."})

    assert asyncio.run(main())["async"] is True


def test_acall_sync_only_tool_runs_in_thread():
    ex = ToolGradExecutor({"sh": _SyncOnlyTool()})

    async def main():
        return await ex.acall("sh", {"path": "."})

    assert asyncio.run(main()) == {"sync": True}


def test_sync_only_tool_call_unchanged():
    ex = ToolGradExecutor({"sh": _SyncOnlyTool()})
    assert ex.call("sh", {"path": "."}) == {"sync": True}


def test_tool_with_both_paths_prefers_async():
    # Matches generation: the agent executor drives tools via ainvoke.
    tool = _AsyncOnlyMCPTool()
    tool.func = lambda ti: {"sync": True}  # now both paths exist
    ex = ToolGradExecutor({"t": tool})
    assert ex.call("t", {"path": "."})["async"] is True


def test_averify_qa_matches_verify_qa():
    async def go():
        ex = MockToolExecutor()
        ex.register(
            "ls",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"]},
            ahandler=lambda ti: {"files": ["a.txt"]},
        )
        qa = _qa([{"tool": "ls", "tool_input": {"path": "."},
                   "result_preview": "..."}])
        sync_report = chain_verifier.verify_qa(qa, ex)
        async_report = await chain_verifier.averify_qa(qa, ex)
        assert sync_report["chain_valid"] is True
        assert async_report["chain_valid"] is True
        assert [s["ok"] for s in async_report["executed_steps"]] == [True]

    asyncio.run(go())


def test_mock_acall_falls_back_to_sync_handler():
    async def go():
        ex = MockToolExecutor()
        ex.register("echo", {}, lambda ti: {"echo": ti["v"]})
        assert await ex.acall("echo", {"v": 1}) == {"echo": 1}

    asyncio.run(go())


def test_averify_qa_records_async_handler_errors():
    async def go():
        ex = MockToolExecutor()

        async def boom(ti):
            raise RuntimeError("kaput")

        ex.register("bad", {"type": "object"}, ahandler=boom)
        qa = _qa([{"tool": "bad", "tool_input": {}, "result_preview": "..."}])
        report = await chain_verifier.averify_qa(qa, ex)
        assert report["chain_valid"] is False
        assert "kaput" in report["executed_steps"][0]["error"]

    asyncio.run(go())


def test_executor_verify_timeout_records_step_failure():
    import asyncio

    class SlowTool:
        name = "slow"
        args_schema = {"type": "object", "properties": {}}
        coroutine = True  # async path present; replaced with bound method below

        async def ainvoke(self, tool_input):
            await asyncio.sleep(30)
            return "never"

    tool = SlowTool()
    tool.coroutine = tool.ainvoke  # mark async path
    ex = ToolGradExecutor({"slow": tool}, verify_timeout_s=0.2)
    try:
        ex.call("slow", {})
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError, not a hang")

    qa = _qa([{"tool": "slow", "tool_input": {}, "result_preview": "..."}])
    report = chain_verifier.verify_qa(qa, ex)
    assert report["chain_valid"] is False
    step = report["executed_steps"][0]
    assert step["ok"] is False
    assert "TimeoutError" in step["error"]


def test_executor_loop_runner_pins_async_calls():
    import asyncio

    from bridge.mcp_client import LoopRunner

    seen_loops = []

    class AsyncTool:
        name = "pinned"
        args_schema = {"type": "object", "properties": {}}

        async def ainvoke(self, tool_input):
            seen_loops.append(asyncio.get_running_loop())
            return "ok"

    tool = AsyncTool()
    tool.coroutine = tool.ainvoke
    runner = LoopRunner()
    try:
        ex = ToolGradExecutor({"pinned": tool}, loop_runner=runner)
        assert ex.call("pinned", {}) == "ok"
        assert seen_loops and seen_loops[0] is runner.loop
    finally:
        runner.close()


def test_executor_acall_uses_timeout_without_loop_runner():
    import asyncio

    async def main():
        class SlowTool:
            name = "slow2"
            args_schema = {"type": "object", "properties": {}}

            async def ainvoke(self, tool_input):
                await asyncio.sleep(30)
                return "never"

        tool = SlowTool()
        tool.coroutine = tool.ainvoke
        ex = ToolGradExecutor({"slow2": tool}, verify_timeout_s=0.2)
        try:
            await ex.acall("slow2", {})
        except TimeoutError:
            return "timed-out"
        return "no-timeout"

    assert asyncio.run(main()) == "timed-out"
