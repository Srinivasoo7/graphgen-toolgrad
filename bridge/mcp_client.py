"""Generic MCP discovery for the factory: any server, any tool surface.

The bridge used to assume one hardcoded filesystem MCP server. This module
connects to arbitrary MCP servers from the run config (stdio or SSE),
discovers their tools, and returns them in two shapes:

- ``tools`` — ``{name: langchain tool}``, ready for
  :class:`bridge.chain_verifier.ToolGradExecutor` and the ToolGrad fork's
  generation graph;
- ``descriptors`` — plain ``ToolDescriptor`` dicts (name, description,
  input schema, server), the input to ToolKG construction and the domain
  KG builder — no langchain objects leak into those paths.

Connections are kept alive on the returned :class:`DiscoveredTools`
(use it as an async context manager or call :meth:`aclose`); discovery
has per-server timeouts and retries. The ``mcp`` /
``langchain_mcp_adapters`` packages are imported lazily so keyless unit
tests never need them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MCPError(Exception):
    """Discovery or connection failure for an MCP server."""


@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str


@dataclass
class DiscoveredTools:
    """Discovered tools from one MCP server; owns the client session."""
    server_name: str
    tools: Dict[str, Any] = field(default_factory=dict)
    descriptors: List[ToolDescriptor] = field(default_factory=list)
    _client: Any = None

    async def aclose(self) -> None:
        # MultiServerMCPClient holds sessions internally; dropping our
        # reference lets them close. Best-effort: never fail teardown.
        self._client = None

    async def __aenter__(self) -> "DiscoveredTools":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def __len__(self) -> int:
        return len(self.tools)


class LoopRunner:
    """One dedicated thread running a single asyncio event loop.

    A factory run keeps one LoopRunner alive from discovery through
    refinement, so every MCP call executes on the same loop that created
    the client sessions (loop affinity avoids cross-loop surprises with
    third-party session objects).
    """

    def __init__(self) -> None:
        import threading

        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(
            target=self._serve, name="ggt-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise MCPError("event-loop thread failed to start")

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def run(self, coro):
        """Drive ``coro`` on the home loop; blocks until done. Thread-safe."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)


def _describe_tool(tool: Any, server_name: str) -> ToolDescriptor:
    name = getattr(tool, "name", "unknown_tool")
    description = getattr(tool, "description", "") or ""
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if isinstance(schema, dict) and schema.get("type") != "object":
        # langchain-mcp-adapters sometimes surfaces {"type": ...} fragments
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return ToolDescriptor(
        name=str(name),
        description=str(description),
        input_schema=schema,
        server=server_name,
    )


# Env vars passed through to stdio MCP server subprocesses (under any
# user-configured values). langchain-mcp-adapters otherwise spawns servers
# with a minimal environment (HOME/PATH only); without proxy and CA-bundle
# vars, commands like `npx` cannot reach their registry through corporate
# proxies and retry until the discovery/tool timeout fires — which looks
# exactly like a hung MCP session. Verified 2026-09-20: npx-spawned
# servers answered in ~1.2s with these vars, hung indefinitely without.
_NETWORK_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    "NODE_USE_ENV_PROXY",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE", "GIT_SSL_CAINFO",
)


def _network_env() -> Dict[str, str]:
    """Proxy/TLS env vars inherited from this process for MCP servers."""
    return {k: v for k, v in os.environ.items() if k in _NETWORK_ENV_KEYS}


def _connection_entry(server_cfg) -> Dict[str, Any]:
    if server_cfg.transport == "sse":
        return {"transport": "sse", "url": server_cfg.url}
    entry: Dict[str, Any] = {
        "transport": "stdio",
        "command": server_cfg.command,
        "args": list(server_cfg.args),
    }
    env = _network_env()
    if server_cfg.env:
        env.update(server_cfg.env)  # explicit config wins
    if env:
        entry["env"] = env
    return entry


async def _discover_once(server_cfg, *, _client_factory=None) -> DiscoveredTools:
    if _client_factory is not None:
        factory = _client_factory
    else:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise MCPError(
                "langchain_mcp_adapters is required for MCP discovery "
                "(pip install langchain-mcp-adapters)"
            ) from exc
        factory = MultiServerMCPClient
    client = factory({_connection_name(server_cfg): _connection_entry(server_cfg)})
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=server_cfg.timeout_s)
    except asyncio.TimeoutError as exc:
        raise MCPError(
            f"server {server_cfg.name!r}: discovery timed out "
            f"after {server_cfg.timeout_s}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — normalize connect failures
        raise MCPError(f"server {server_cfg.name!r}: discovery failed: {exc}") from exc
    by_name = {t.name: t for t in tools}
    return DiscoveredTools(
        server_name=server_cfg.name,
        tools=by_name,
        descriptors=[_describe_tool(t, server_cfg.name) for t in tools],
        _client=client,
    )


def _connection_name(server_cfg) -> str:
    return server_cfg.name


async def discover_mcp_tools(
    server_cfg,
    *,
    _client_factory=None,
) -> DiscoveredTools:
    """Connect to one MCP server and discover its tools, with retries.

    Retries transient failures ``server_cfg.retries`` times with a short
    linear backoff; the last error is raised as :class:`MCPError`.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max(0, server_cfg.retries) + 1):
        try:
            return await _discover_once(server_cfg, _client_factory=_client_factory)
        except MCPError as exc:
            last_exc = exc
            if attempt < server_cfg.retries:
                await asyncio.sleep(1.0 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


async def discover_all(server_cfgs, *, _client_factory=None) -> List[DiscoveredTools]:
    """Discover tools from every configured server (sequentially).

    Sequential on purpose: parallel stdio spawns are flaky on small boxes,
    and discovery is a one-time per-run cost.
    """
    return [
        await discover_mcp_tools(cfg, _client_factory=_client_factory)
        for cfg in server_cfgs
    ]


def merge_discovered(all_discovered: List[DiscoveredTools]) -> DiscoveredTools:
    """Merge per-server results; duplicate tool names get server prefixes."""
    merged = DiscoveredTools(server_name="merged")
    for disc in all_discovered:
        for name, tool in disc.tools.items():
            final = name
            if final in merged.tools:
                final = f"{disc.server_name}__{name}"
            merged.tools[final] = tool
        for desc in disc.descriptors:
            final = desc.name
            if any(d.name == final for d in merged.descriptors):
                final = f"{desc.server}__{desc.name}"
            merged.descriptors.append(
                ToolDescriptor(final, desc.description, desc.input_schema, desc.server)
            )
        if merged._client is None:
            merged._client = []
        merged._client.append(disc._client)
    return merged
