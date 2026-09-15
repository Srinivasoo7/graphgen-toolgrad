"""Tests for bridge/toolkg_sampler.py (keyless).

The sampler closure is exercised with a prebuilt ToolKG over fake tools,
so no MCP server (npx) is spawned. ``build_toolkg_for_mcp`` (live
discovery) is covered by the real-ToolKG check in Phase 2, not here.
"""

from types import SimpleNamespace

from bridge import toolkg_builder, toolkg_sampler


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
    return g, catalog


def test_sampler_returns_tool_objects_from_catalog():
    g, catalog = _prebuilt()
    sampler = toolkg_sampler.make_toolkg_sampler(g, seed=1)
    got = sampler(catalog, 3)
    assert len(got) == 3
    assert len({t.name for t in got}) == 3
    assert all(t in catalog for t in got)  # identity, not copies


def test_sampler_is_deterministic_per_instance():
    g, catalog = _prebuilt()
    s1 = toolkg_sampler.make_toolkg_sampler(g, seed=7)
    s2 = toolkg_sampler.make_toolkg_sampler(g, seed=7)
    first = [t.name for t in s1(catalog, 3)]
    # Fresh instance, same seed -> same first sample.
    assert [t.name for t in s2(catalog, 3)] == first


def test_sampler_varies_across_calls():
    # Upstream's default path reseeds the global RNG identically every call
    # (same bundle each iteration); the sampler must not repeat that.
    g, catalog = _prebuilt()
    sampler = toolkg_sampler.make_toolkg_sampler(g, seed=7)
    seen = set()
    for _ in range(6):
        seen.add(tuple(t.name for t in sampler(catalog, 2)))
    assert len(seen) > 1, "sampler returned the same bundle every call"


def test_sampler_validates_num_apis():
    g, catalog = _prebuilt()
    sampler = toolkg_sampler.make_toolkg_sampler(g)
    for bad in (0, 99):
        try:
            sampler(catalog, bad)
            raise AssertionError(f"expected ValueError for num_apis={bad}")
        except ValueError:
            pass


def test_sampler_matches_fork_seam_signature():
    # The fork calls sampler(tools, num_apis) positionally.
    g, catalog = _prebuilt()
    sampler = toolkg_sampler.make_toolkg_sampler(g)
    got = sampler(catalog, 2)  # must not raise on positional args
    assert len(got) == 2
