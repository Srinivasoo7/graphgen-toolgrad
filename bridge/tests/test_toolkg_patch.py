"""Tests for bridge/toolkg_patch.py (keyless).

The patch is exercised with a prebuilt ToolKG over fake tools, so no MCP
server (npx) is spawned. Also verifies the plumbing fact the patch relies
on: ``graph_lib`` resolves ``mcp.get_mcp_apis`` via module attribute at
call time.
"""

from types import SimpleNamespace

from bridge import toolkg_builder, toolkg_patch


def _fake_tool(name, description, properties):
    return SimpleNamespace(
        name=name,
        description=description,
        args_schema={"properties": properties},
    )


def _catalog():
    return [
        _fake_tool("list_directory", "List files in a path.",
                   {"path": {"type": "string"}}),
        _fake_tool("read_text_file", "Read a file as text.",
                   {"path": {"type": "string"}}),
        _fake_tool("read_multiple_files", "Read several files.",
                   {"paths": {"type": "array"}}),
        _fake_tool("directory_tree", "Tree view as JSON.",
                   {"path": {"type": "string"}}),
    ]


_HINTS = {
    "list_directory": [("path", "path")],
    "read_text_file": [("content", "string")],
    "read_multiple_files": [("content", "string")],
    "directory_tree": [("path", "path")],
}


def _prebuilt():
    catalog = _catalog()
    g = toolkg_builder.build_toolkg(catalog, output_hints=_HINTS)
    return g, {t.name: t for t in catalog}


def test_patch_plumbs_through_graph_lib():
    # graph_lib must resolve get_mcp_apis on the mcp *module* at call time;
    # otherwise the attribute swap would not take effect.
    import toolgrad.modules.graph_lib as graph_lib
    from toolgrad.utils import mcp as mcp_module

    assert graph_lib.mcp is mcp_module


def test_apply_sample_remove_cycle():
    import toolgrad.utils.mcp as mcp_module

    original = mcp_module.get_mcp_apis
    g, tools_by_name = _prebuilt()
    try:
        toolkg_patch.apply_toolkg_patch(toolkg=g, tools_by_name=tools_by_name)
        assert toolkg_patch.is_patched()
        assert mcp_module.get_mcp_apis is not original

        got = mcp_module.get_mcp_apis({}, num_apis=3, seed=1)
        assert len(got) == 3
        names = [t.name for t in got]
        assert len(set(names)) == 3
        assert set(names) <= set(tools_by_name)

        # Determinism: re-apply resets the call counter -> same sample.
        toolkg_patch.remove_toolkg_patch()
        toolkg_patch.apply_toolkg_patch(toolkg=g, tools_by_name=tools_by_name)
        got2 = mcp_module.get_mcp_apis({}, num_apis=3, seed=1)
        assert [t.name for t in got2] == names

        # num_apis validation mirrors upstream's error contract.
        try:
            mcp_module.get_mcp_apis({}, num_apis=99, seed=1)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        # Double-apply stays consistent.
        toolkg_patch.apply_toolkg_patch(toolkg=g, tools_by_name=tools_by_name)
        assert toolkg_patch.is_patched()
        assert mcp_module.get_mcp_apis is not original
    finally:
        toolkg_patch.remove_toolkg_patch()
    assert not toolkg_patch.is_patched()
    assert mcp_module.get_mcp_apis is original


def test_remove_when_unpatched_is_safe():
    toolkg_patch.remove_toolkg_patch()  # must not raise
    assert not toolkg_patch.is_patched()


def test_get_toolkg_accessor():
    g, tools_by_name = _prebuilt()
    try:
        toolkg_patch.apply_toolkg_patch(toolkg=g, tools_by_name=tools_by_name)
        assert toolkg_patch.get_toolkg() is g
    finally:
        toolkg_patch.remove_toolkg_patch()
