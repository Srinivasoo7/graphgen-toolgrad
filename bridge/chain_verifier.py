"""Phase 3b: verify that a QA pair's claimed tool chain is actually executable.

A tool-grounded QA pair is only as good as its chain: if a tool is missing
or its inputs don't match the declared schema, the "verified" execution is
a fiction. This module checks every emitted chain against a tool executor
*before* it enters the SFT mix.

Two executors:
- ``MockToolExecutor`` — keyless. Implements the minimal tool-call
  interface (``call(tool_name, tool_input)`` / ``acall``) with registered
  input schemas and canned handlers. Used by tests and by offline
  validation.
- ``ToolGradExecutor`` — live path. Wraps real ToolGrad tools
  (``name -> StructuredTool``; from the fork's ``discover_mcp_tools``).
  ``call`` is async-aware: async-only MCP tools (the fork's
  ``_wrap_with_path_prefix`` builds the sync ``func`` over the original
  tool's ``func``, which is ``None`` for async-only tools) are executed
  via ``ainvoke`` run to completion; tools with a real sync ``func`` go
  through ``invoke`` as before. ``acall`` is the native async entry point
  for callers already inside an event loop.

``verify_qa(qa, executor)`` is the sync verifier; ``averify_qa`` is its
async sibling and uses ``executor.acall``.

``verify_qa(qa, executor)`` returns::

    {
      "chain_valid": bool,
      "missing_tools": [str],        # tools with no registered executor
      "schema_mismatches": [         # per-step input problems
        {"step": int, "tool": str, "problems": [str]}],
      "executed_steps": [            # per-step execution outcome
        {"step": int, "tool": str, "ok": bool, "error": str | None}],
    }

``chain_valid`` is True iff there are no missing tools, no schema
mismatches, and every step executed without error.

``chain_toolkg_coverage(chain, toolkg)`` measures what fraction of
consecutive tool pairs in the chain are edges in the Phase-2 ToolKG — a
cheap "does this chain follow real composability?" signal for Phase 4
reranking (cf. ``kg_sampler.neighborhood_density``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Sequence


def _schema_problems(tool_input: dict, schema: dict) -> List[str]:
    """Check ``tool_input`` against a JSON-schema-ish ``schema``.

    Reports missing required properties and coarse type mismatches for
    the common JSON types. Unknown / untyped properties are ignored.
    """
    problems: List[str] = []
    if not isinstance(tool_input, dict):
        return ["tool_input is not an object"]
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    for name in schema.get("required", []) if isinstance(schema, dict) else []:
        if name not in tool_input:
            problems.append(f"missing required property '{name}'")
    type_map = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict,
    }
    for name, value in tool_input.items():
        spec = props.get(name, {})
        want = spec.get("type") if isinstance(spec, dict) else None
        py = type_map.get(want)
        if py is not None and not isinstance(value, py):
            problems.append(
                f"property '{name}' should be {want}, got {type(value).__name__}"
            )
        # bool is a subclass of int: a bool where an integer is expected is fine.
        if want == "integer" and isinstance(value, bool):
            problems[:] = [p for p in problems if f"'{name}'" not in p]
    return problems


def _await_sync(coro) -> Any:
    """Run ``coro`` to completion from synchronous code.

    Uses ``asyncio.run`` when no event loop is running; otherwise hops to
    a dedicated thread (a running loop's thread cannot be blocked). This
    is what lets the sync ``verify_qa`` path execute async-only MCP tools.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class MockToolExecutor:
    """Keyless executor: registered tools with schemas + canned handlers."""

    def __init__(self) -> None:
        self._tools: Dict[str, dict] = {}

    def register(
        self,
        name: str,
        input_schema: Optional[dict] = None,
        handler: Optional[Callable[[dict], Any]] = None,
        ahandler: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self._tools[name] = {
            "input_schema": input_schema or {},
            "handler": handler or (lambda tool_input: {"ok": True, "input": tool_input}),
            "ahandler": ahandler,
        }

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def input_schema(self, name: str) -> dict:
        return self._tools[name]["input_schema"]

    def call(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        return self._tools[tool_name]["handler"](tool_input)

    async def acall(self, tool_name: str, tool_input: dict) -> Any:
        """Async entry: uses the registered async handler, else the sync one."""
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        entry = self._tools[tool_name]
        if entry["ahandler"] is not None:
            result = entry["ahandler"](tool_input)
        else:
            result = entry["handler"](tool_input)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        return result


class ToolGradExecutor:
    """Live executor over real ToolGrad / langchain tools.

    ``tools_by_name`` maps tool name -> tool object. ``call`` prefers the
    tool's async path whenever one exists (``coroutine`` set) — the same
    call shape the agent executor uses during generation — running it to
    completion; tools with only a sync ``func`` go through ``invoke``.
    This covers the fork's ``_wrap_with_path_prefix`` MCP tools, whose
    sync ``func`` is built over ``None`` for async-only tools.
    ``acall`` is the native async entry point.

    ``loop_runner`` (a :class:`bridge.mcp_client.LoopRunner`) pins every
    async tool call to the event loop that owns the MCP sessions, keeping
    discovery and invocation on one loop. Every async call is additionally
    bounded by ``verify_timeout_s`` so a stuck tool becomes a recorded
    verification failure, never a dead run.
    """

    def __init__(
        self,
        tools_by_name: Dict[str, Any],
        *,
        loop_runner: Any = None,
        verify_timeout_s: float = 60.0,
    ) -> None:
        self._tools = dict(tools_by_name)
        self._loop_runner = loop_runner
        self._timeout = verify_timeout_s

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def input_schema(self, name: str) -> dict:
        tool = self._tools[name]
        schema = getattr(tool, "args_schema", None) or getattr(tool, "args", {})
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        return schema if isinstance(schema, dict) else {}

    def _prefers_async(self, tool: Any) -> bool:
        # Prefer the async path whenever the tool has one: it is the path the
        # agent executor uses during generation, so verification replays the
        # same call shape. This also covers the fork's wrapped MCP tools,
        # whose sync ``func`` may be built over ``None`` for async-only tools.
        return getattr(tool, "coroutine", None) is not None

    def _invoke_coro(self, tool: Any, tool_input: dict):
        """Coroutine running one async tool call bounded by the timeout."""

        async def go():
            return await asyncio.wait_for(
                tool.ainvoke(tool_input), timeout=self._timeout)

        return go()

    def call(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        tool = self._tools[tool_name]
        if self._prefers_async(tool):
            if self._loop_runner is not None:
                return self._loop_runner.run(self._invoke_coro(tool, tool_input))
            return _await_sync(self._invoke_coro(tool, tool_input))
        return tool.invoke(tool_input)

    async def acall(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        tool = self._tools[tool_name]
        if getattr(tool, "coroutine", None) is not None:
            if self._loop_runner is not None:
                return await asyncio.to_thread(
                    self._loop_runner.run,
                    self._invoke_coro(tool, tool_input))
            return await asyncio.wait_for(
                tool.ainvoke(tool_input), timeout=self._timeout)
        return await asyncio.to_thread(tool.invoke, tool_input)


def verify_qa(qa: dict, executor) -> dict:
    """Verify one QA record's chain against ``executor``.

    Steps: for each chain step, (1) the tool must exist, (2) its input
    must satisfy the declared schema, (3) calling it must not raise.
    Execution errors are captured per step — one failing step does not
    abort the rest.
    """
    missing_tools: List[str] = []
    schema_mismatches: List[dict] = []
    executed_steps: List[dict] = []

    for i, step in enumerate(qa.get("chain", [])):
        tool = step.get("tool")
        tool_input = step.get("tool_input", {})
        record = {"step": i, "tool": tool, "ok": False, "error": None}

        if not executor.has_tool(tool):
            missing_tools.append(tool)
            record["error"] = f"missing tool: {tool!r}"
            executed_steps.append(record)
            continue

        try:
            problems = _schema_problems(tool_input, executor.input_schema(tool))
        except Exception as exc:  # noqa: BLE001 — schema lookup must not crash verify
            problems = [f"schema lookup failed: {exc}"]
        if problems:
            schema_mismatches.append({"step": i, "tool": tool, "problems": problems})
            record["error"] = "; ".join(problems)
            executed_steps.append(record)
            continue

        try:
            executor.call(tool, tool_input)
        except Exception as exc:  # noqa: BLE001 — record, don't raise
            record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["ok"] = True
        executed_steps.append(record)

    return _verify_report(missing_tools, schema_mismatches, executed_steps)


async def averify_qa(qa: dict, executor) -> dict:
    """Async sibling of :func:`verify_qa`; executes steps via ``acall``.

    Preferred when the caller already runs inside an event loop — avoids
    the thread hop that sync ``call`` needs for async-only tools.
    """
    missing_tools: List[str] = []
    schema_mismatches: List[dict] = []
    executed_steps: List[dict] = []

    for i, step in enumerate(qa.get("chain", [])):
        tool = step.get("tool")
        tool_input = step.get("tool_input", {})
        record = {"step": i, "tool": tool, "ok": False, "error": None}

        if not executor.has_tool(tool):
            missing_tools.append(tool)
            record["error"] = f"missing tool: {tool!r}"
            executed_steps.append(record)
            continue

        try:
            problems = _schema_problems(tool_input, executor.input_schema(tool))
        except Exception as exc:  # noqa: BLE001 — schema lookup must not crash verify
            problems = [f"schema lookup failed: {exc}"]
        if problems:
            schema_mismatches.append({"step": i, "tool": tool, "problems": problems})
            record["error"] = "; ".join(problems)
            executed_steps.append(record)
            continue

        try:
            await executor.acall(tool, tool_input)
        except Exception as exc:  # noqa: BLE001 — record, don't raise
            record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["ok"] = True
        executed_steps.append(record)

    return _verify_report(missing_tools, schema_mismatches, executed_steps)


def _verify_report(missing_tools, schema_mismatches, executed_steps) -> dict:
    chain_valid = (
        not missing_tools
        and not schema_mismatches
        and all(s["ok"] for s in executed_steps)
        and bool(executed_steps)
    )
    return {
        "chain_valid": chain_valid,
        "missing_tools": missing_tools,
        "schema_mismatches": schema_mismatches,
        "executed_steps": executed_steps,
    }


def chain_toolkg_coverage(chain: Sequence[dict], toolkg) -> float:
    """Fraction of consecutive tool pairs in ``chain`` that are ToolKG edges.

    ``chain`` may be QA ``"chain"`` step dicts (``{"tool": ...}``) or bare
    tool-name strings. A single-step chain is vacuously composable and
    scores 1.0; an empty chain, or a missing ToolKG, scores 0.0.
    """
    names = [s.get("tool") if isinstance(s, dict) else s for s in chain]
    names = [n for n in names if n]
    if not names:
        return 0.0
    if len(names) == 1:
        return 1.0
    if toolkg is None:
        return 0.0
    hits = sum(1 for a, b in zip(names, names[1:]) if toolkg.has_edge(a, b))
    return hits / (len(names) - 1)
