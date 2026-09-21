"""Tests for bridge/mcp_client.py (keyless: fake client factories)."""

import asyncio
from types import SimpleNamespace

from bridge import mcp_client
from bridge.mcp_client import DiscoveredTools, ToolDescriptor, merge_discovered


def _server_cfg(name="fs", timeout_s=5.0, retries=0):
    return SimpleNamespace(
        name=name, transport="stdio", command="npx", args=["-y", "s"],
        env={}, url="", timeout_s=timeout_s, retries=retries,
    )


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} tool"
        self.args_schema = {"type": "object", "properties": {"p": {"type": "string"}}}


class _FakeClient:
    def __init__(self, tools, delay=0.0, fail_with=None):
        self._tools = tools
        self._delay = delay
        self._fail_with = fail_with
        self.calls = 0

    def __call__(self, entry):
        self.calls += 1
        return self

    async def get_tools(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_with:
            raise self._fail_with
        return list(self._tools)


def _discover(cfg, factory):
    return asyncio.run(mcp_client.discover_mcp_tools(cfg, _client_factory=factory))


def test_discover_returns_tools_and_descriptors():
    cfg = _server_cfg()
    factory = _FakeClient([_FakeTool("read_file"), _FakeTool("write_file")])
    disc = _discover(cfg, factory)
    assert set(disc.tools) == {"read_file", "write_file"}
    assert {d.name for d in disc.descriptors} == {"read_file", "write_file"}
    assert all(d.server == "fs" for d in disc.descriptors)
    assert disc.descriptors[0].input_schema["type"] == "object"


def test_discover_timeout_becomes_mcperror():
    cfg = _server_cfg(timeout_s=0.05)
    factory = _FakeClient([_FakeTool("x")], delay=1.0)
    try:
        _discover(cfg, factory)
    except mcp_client.MCPError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected MCPError")


def test_discover_retries_then_raises_last_error():
    cfg = _server_cfg(retries=2)
    factory = _FakeClient([], fail_with=ConnectionError("refused"))
    try:
        _discover(cfg, factory)
    except mcp_client.MCPError as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("expected MCPError")
    assert factory.calls == 3  # initial + 2 retries


def test_discover_retry_recovers():
    cfg = _server_cfg(retries=2)

    class Flaky(_FakeClient):
        async def get_tools(self):
            self.calls_inner = getattr(self, "calls_inner", 0) + 1
            if self.calls_inner < 2:
                raise ConnectionError("flaky")
            return [_FakeTool("ok_tool")]

    flaky = Flaky([])
    disc = asyncio.run(mcp_client.discover_mcp_tools(cfg, _client_factory=flaky))
    assert set(disc.tools) == {"ok_tool"}


def test_merge_prefixes_duplicate_tool_names():
    a = DiscoveredTools(
        server_name="crm",
        tools={"get_order": _FakeTool("get_order")},
        descriptors=[ToolDescriptor("get_order", "d", {}, "crm")],
    )
    b = DiscoveredTools(
        server_name="erp",
        tools={"get_order": _FakeTool("get_order")},
        descriptors=[ToolDescriptor("get_order", "d", {}, "erp")],
    )
    merged = merge_discovered([a, b])
    assert "get_order" in merged.tools
    assert "erp__get_order" in merged.tools
    names = [d.name for d in merged.descriptors]
    assert "get_order" in names and "erp__get_order" in names
    # the underlying tool objects are preserved, not copies
    assert merged.tools["erp__get_order"] is b.tools["get_order"]


def test_merge_no_clobber_without_duplicates():
    a = DiscoveredTools(
        server_name="crm", tools={"t1": _FakeTool("t1")},
        descriptors=[ToolDescriptor("t1", "d", {}, "crm")],
    )
    b = DiscoveredTools(
        server_name="erp", tools={"t2": _FakeTool("t2")},
        descriptors=[ToolDescriptor("t2", "d", {}, "erp")],
    )
    merged = merge_discovered([a, b])
    assert set(merged.tools) == {"t1", "t2"}


def test_aclose_never_fails():
    disc = DiscoveredTools(server_name="x", _client=object())
    asyncio.run(disc.aclose())
    asyncio.run(disc.__aexit__(None, None, None))


def test_loop_runner_reuses_one_loop():
    import asyncio

    from bridge.mcp_client import LoopRunner

    async def which_loop():
        return asyncio.get_running_loop()

    runner = LoopRunner()
    try:
        first = runner.run(which_loop())
        second = runner.run(which_loop())
        assert first is second is runner.loop
        assert runner._thread.is_alive()
    finally:
        runner.close()
    assert not runner._thread.is_alive()


class _CapturingClient(_FakeClient):
    def __init__(self):
        super().__init__([_FakeTool("t")])
        self.seen = None

    def __call__(self, connections):
        self.seen = connections
        return self


def _with_env(extra):
    import os

    saved = dict(os.environ)
    os.environ.update(extra)
    return saved


def _restore_env(saved):
    import os

    os.environ.clear()
    os.environ.update(saved)


def test_connection_entry_passes_proxy_env_to_stdio():
    import os

    from bridge.mcp_client import _connection_entry

    saved = _with_env({"HTTPS_PROXY": "http://proxy:8080", "NO_PROXY": "localhost"})
    try:
        entry = _connection_entry(_server_cfg())
    finally:
        _restore_env(saved)
    assert entry["env"]["HTTPS_PROXY"] == "http://proxy:8080"
    assert entry["env"]["NO_PROXY"] == "localhost"
    # unrelated process env must not leak into the server env
    assert "JARVIS_RUNTIME_CONTEXT_TOKEN" not in entry["env"]
    assert "PATH" not in entry["env"]  # adapter supplies its own safe subset


def test_connection_entry_explicit_env_wins_over_network_env():
    from bridge.mcp_client import _connection_entry

    saved = _with_env({"HTTPS_PROXY": "http://proxy:8080"})
    try:
        cfg = _server_cfg()
        cfg.env = {"HTTPS_PROXY": "http://custom:9090"}
        entry = _connection_entry(cfg)
    finally:
        _restore_env(saved)
    assert entry["env"]["HTTPS_PROXY"] == "http://custom:9090"


def test_discover_sends_env_through_to_client_factory():
    factory = _CapturingClient()
    saved = _with_env({"HTTPS_PROXY": "http://proxy:8080"})
    try:
        _discover(_server_cfg(), factory)
    finally:
        _restore_env(saved)
    entry = factory.seen["fs"]
    assert entry["transport"] == "stdio"
    assert entry["env"]["HTTPS_PROXY"] == "http://proxy:8080"


def test_connection_entry_sse_untouched():
    from types import SimpleNamespace

    from bridge.mcp_client import _connection_entry

    cfg = SimpleNamespace(name="w", transport="sse", url="http://x",
                          command="", args=[], env={})
    assert _connection_entry(cfg) == {"transport": "sse", "url": "http://x"}
