"""RED then GREEN: agents may only call pinned MCP tools."""

import pytest

from crux.access import AccessDenied, Operation
from crux.tools import READ_TOOLS, WRITE_TOOLS, operation_for


def test_pinned_read_tools_are_read():
    assert "search_chunks" in READ_TOOLS
    assert "query_data" not in READ_TOOLS
    assert "query_data" not in WRITE_TOOLS
    assert operation_for("search_chunks") is Operation.READ
    assert operation_for("entity_facts") is Operation.READ


def test_remember_is_propose_not_read():
    assert "remember" in WRITE_TOOLS
    assert operation_for("remember") is Operation.PROPOSE


def test_unknown_tool_is_rejected():
    with pytest.raises(AccessDenied, match="not an instrumented MCP tool"):
        operation_for("query_data")
    with pytest.raises(AccessDenied, match="not an instrumented MCP tool"):
        operation_for("ontology_admin")
