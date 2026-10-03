"""Pinned Utopia MCP tools. query_data and admin/ontology tools are not on this path."""

from __future__ import annotations

from crux.access import AccessDenied, Operation

READ_TOOLS = frozenset(
    {
        "search_chunks",
        "get_document",
        "search_docs",
        "find_entities",
        "list_rules",
        "rule_matches",
        "changes",
        "entity_facts",
    }
)

WRITE_TOOLS = frozenset({"remember"})


def operation_for(tool: str) -> Operation:
    if tool in WRITE_TOOLS:
        return Operation.PROPOSE
    if tool in READ_TOOLS:
        return Operation.READ
    raise AccessDenied(f"{tool} is not an instrumented MCP tool")


def tools_for_operations(operations: frozenset[Operation]) -> list[str]:
    names = set()
    if Operation.READ in operations:
        names.update(READ_TOOLS)
    if Operation.PROPOSE in operations:
        names.update(WRITE_TOOLS)
    return sorted(names)


def tools_manifest() -> dict[str, object]:
    return {
        "read": sorted(READ_TOOLS),
        "write": sorted(WRITE_TOOLS),
        "remember_is_propose": True,
    }
