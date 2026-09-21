"""Native-seam contract: fail loudly if the forks' integration points move.

The bridge now integrates through public seams on Sri's forks
(``Srinivasoo7/toolgrad@integrate/graphgen-toolgrad``), not monkeypatches:

* ``PREDICT_WORKFLOW`` carries a ``{kg_context}`` variable;
* ``create_workflow_updater(kg_context="")`` binds it (default: unchanged);
* ``ToolGradState.kg_context`` threads it from graph state;
* ``get_mcp_apis(..., sampler=None)`` / ``discover_mcp_tools()`` /
  ``sample_apis_or_end(..., api_sampler=None)`` /
  ``create_graph_on_mcp(..., api_sampler=None)`` form the sampling seam.

Skipped entirely when ToolGrad is not installed.
"""


def _mods():
    try:
        from toolgrad.modules import graph_lib, module_lib, prompt_lib
        from toolgrad.prebuilt import toolgrad_on_mcp
        from toolgrad.utils import mcp
    except ImportError:
        return None
    return prompt_lib, module_lib, graph_lib, toolgrad_on_mcp, mcp


def test_predict_workflow_has_kg_context_variable():
    mods = _mods()
    if mods is None:
        return "SKIP"
    prompt_lib, _, _, _, _ = mods
    tmpl = prompt_lib.PREDICT_WORKFLOW
    assert type(tmpl).__name__ == "ChatPromptTemplate", type(tmpl)
    assert set(tmpl.input_variables) == {"api_use_chains", "kg_context"}, (
        tmpl.input_variables
    )


def test_workflow_updater_accepts_kg_context():
    mods = _mods()
    if mods is None:
        return "SKIP"
    import inspect

    _, module_lib, _, _, _ = mods
    params = inspect.signature(module_lib.create_workflow_updater).parameters
    assert "kg_context" in params, list(params)
    assert params["kg_context"].default == "", params["kg_context"].default


def test_toolgrad_state_carries_kg_context():
    mods = _mods()
    if mods is None:
        return "SKIP"
    _, _, graph_lib, _, _ = mods
    field = graph_lib.ToolGradState.model_fields.get("kg_context")
    assert field is not None, "ToolGradState lost its kg_context field"
    assert field.default == ""


def test_kg_context_renders_into_prompt():
    mods = _mods()
    if mods is None:
        return "SKIP"
    try:
        from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
    except ImportError:
        return "SKIP"
    prompt_lib, module_lib, _, _, _ = mods
    # This is exactly what create_workflow_updater(kg_context=...) does
    # before attaching the LLM — no API key needed to check rendering.
    prompt = prompt_lib.PREDICT_WORKFLOW.partial(kg_context="ENTITIES: Tesla")
    text = "\n".join(
        m.content for m in prompt.format_messages(api_use_chains="CHAINS")
    )
    assert "ENTITIES: Tesla" in text
    assert "CHAINS" in text
    assert "{kg_context}" not in text
    # Default path: empty context still renders (unchanged behavior).
    empty = prompt_lib.PREDICT_WORKFLOW.partial(kg_context="").format_messages(
        api_use_chains="CHAINS"
    )
    assert "{kg_context}" not in "\n".join(m.content for m in empty)


def test_sampler_seam_signatures():
    mods = _mods()
    if mods is None:
        return "SKIP"
    import inspect

    _, _, graph_lib, toolgrad_on_mcp, mcp = mods
    assert callable(mcp.discover_mcp_tools), "discover_mcp_tools missing"
    params = inspect.signature(mcp.get_mcp_apis).parameters
    assert params["sampler"].default is None, list(params)
    assert "api_sampler" in inspect.signature(
        graph_lib.sample_apis_or_end
    ).parameters
    assert "api_sampler" in inspect.signature(
        toolgrad_on_mcp.create_graph_on_mcp
    ).parameters


def _seam_tool(name, *, func=None, coroutine=None):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    class Args(BaseModel):
        path: str = ""

    return StructuredTool(
        name=name, description="d", func=func, coroutine=coroutine,
        args_schema=Args,
    )


def test_wrap_tool_with_tracking_async_only_tool():
    # langchain-mcp-adapters MCP tools are async-only (func=None). The
    # wrapper must track the coroutine instead of wrapping None (the old
    # code produced TypeError: 'NoneType' object is not callable).
    mods = _mods()
    if mods is None:
        return "SKIP"
    import asyncio

    _, module_lib, _, _, _ = mods

    async def fake_coro(path=""):
        return f"ok:{path}"

    tool = _seam_tool("t", coroutine=fake_coro)
    assert tool.func is None
    tracker = {}
    wrapped = module_lib.wrap_tool_with_tracking(tool, tracker)
    assert wrapped.func is None  # stays async-only; sync bridge lives in mcp.py
    assert asyncio.run(wrapped.coroutine(path="x")) == "ok:x"
    assert tracker.get("called") is True


def test_wrap_tool_with_tracking_sync_tool():
    mods = _mods()
    if mods is None:
        return "SKIP"

    _, module_lib, _, _, _ = mods
    tracker = {}
    wrapped = module_lib.wrap_tool_with_tracking(
        _seam_tool("t", func=lambda path="": f"ok:{path}"), tracker)
    assert wrapped.func(path="y") == "ok:y"
    assert tracker.get("called") is True


def test_wrap_tool_with_tracking_rejects_empty_tool():
    mods = _mods()
    if mods is None:
        return "SKIP"

    _, module_lib, _, _, _ = mods
    try:
        module_lib.wrap_tool_with_tracking(_seam_tool("t"), {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for tool with no callables")


def test_wrap_with_path_prefix_async_only_sync_invoke():
    # Async-only MCP tools get a working sync func via the sync bridge
    # (asyncio.run outside a loop, worker thread inside one).
    mods = _mods()
    if mods is None:
        return "SKIP"
    import asyncio

    _, _, _, _, mcp_mod = mods

    async def fake_coro(path=""):
        return f"ok:{path}"

    wrapped = mcp_mod._wrap_with_path_prefix(
        _seam_tool("t", coroutine=fake_coro), "/base")
    assert wrapped.coroutine is not None
    assert wrapped.func(path="rel") == "ok:/base/rel"
    assert wrapped.func(path="/abs") == "ok:/abs"

    async def inside_loop():
        return wrapped.func(path="rel")

    assert asyncio.run(inside_loop()) == "ok:/base/rel"
