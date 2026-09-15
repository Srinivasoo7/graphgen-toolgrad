"""Phase 3b: verify that a QA pair's claimed tool chain is actually executable.

A tool-grounded QA pair is only as good as its chain: if a tool is missing
or its inputs don't match the declared schema, the "verified" execution is
a fiction. This module checks every emitted chain against a tool executor
*before* it enters the SFT mix.

Two executors:
- ``MockToolExecutor`` — keyless. Implements the minimal tool-call
  interface (``call(tool_name, tool_input)``) with registered input
  schemas and canned handlers. Used by tests and by offline validation.
- ``ToolGradExecutor`` — live path. Wraps real ToolGrad tools
  (``name -> StructuredTool``; from the fork's ``discover_mcp_tools``) and
  calls ``tool.invoke(tool_input)`` — the same call the executor agent
  makes.

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

from typing import Any, Callable, Dict, List, Optional


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


class MockToolExecutor:
    """Keyless executor: registered tools with schemas + canned handlers."""

    def __init__(self) -> None:
        self._tools: Dict[str, dict] = {}

    def register(
        self,
        name: str,
        input_schema: Optional[dict] = None,
        handler: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self._tools[name] = {
            "input_schema": input_schema or {},
            "handler": handler or (lambda tool_input: {"ok": True, "input": tool_input}),
        }

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def input_schema(self, name: str) -> dict:
        return self._tools[name]["input_schema"]

    def call(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        return self._tools[tool_name]["handler"](tool_input)


class ToolGradExecutor:
    """Live executor over real ToolGrad / langchain tools.

    ``tools_by_name`` maps tool name -> tool object exposing
    ``.invoke(tool_input)`` (langchain ``StructuredTool``, incl. the MCP
    tools discovered via the fork's ``discover_mcp_tools``). No LLM is
    involved at this layer.
    """

    def __init__(self, tools_by_name: Dict[str, Any]) -> None:
        self._tools = dict(tools_by_name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def input_schema(self, name: str) -> dict:
        tool = self._tools[name]
        schema = getattr(tool, "args_schema", None) or getattr(tool, "args", {})
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        return schema if isinstance(schema, dict) else {}

    def call(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name!r}")
        return self._tools[tool_name].invoke(tool_input)


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
    tool-name strings. Returns 0.0 for chains shorter than 2 steps.
    """
    names = [s.get("tool") if isinstance(s, dict) else s for s in chain]
    names = [n for n in names if n]
    if toolkg is None or len(names) < 2:
        return 0.0
    hits = sum(1 for a, b in zip(names, names[1:]) if toolkg.has_edge(a, b))
    return hits / (len(names) - 1)
