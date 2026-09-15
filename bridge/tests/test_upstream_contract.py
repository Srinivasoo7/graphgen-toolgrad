"""Upstream contract test: fail loudly if ToolGrad's internals move under us.

The bridge integrates via runtime monkeypatching (intentional and
documented in the README's Known limitations). These tests pin the exact
integration points so an upstream refactor becomes a clear failure here
instead of silent breakage. Skipped entirely when ToolGrad is not
installed.
"""


def _toolgrad_modules():
    try:
        from toolgrad.modules import module_lib, prompt_lib
        from toolgrad.utils import mcp
    except ImportError:
        return None
    return prompt_lib, module_lib, mcp


def test_predict_workflow_is_template_with_api_use_chains():
    mods = _toolgrad_modules()
    if mods is None:
        return "SKIP"
    prompt_lib, _, _ = mods
    tmpl = prompt_lib.PREDICT_WORKFLOW
    assert type(tmpl).__name__ == "ChatPromptTemplate", type(tmpl)
    assert "api_use_chains" in tmpl.input_variables, tmpl.input_variables
    # bridge/toolgrad_patch.py replaces this attribute at call time.


def test_create_workflow_updater_exists():
    mods = _toolgrad_modules()
    if mods is None:
        return "SKIP"
    _, module_lib, _ = mods
    assert callable(module_lib.create_workflow_updater)
    # bridge/toolgrad_patch.py swaps this factory.


def test_get_mcp_apis_signature():
    mods = _toolgrad_modules()
    if mods is None:
        return "SKIP"
    import inspect

    _, _, mcp = mods
    params = list(inspect.signature(mcp.get_mcp_apis).parameters)
    assert params[:3] == ["mcp_dict", "num_apis", "seed"], params
    # bridge/toolkg_patch.py::_toolkg_get_mcp_apis mirrors this signature.


def test_private_wrap_with_path_prefix_exists():
    mods = _toolgrad_modules()
    if mods is None:
        return "SKIP"
    _, _, mcp = mods
    # Private upstream API — the most fragile coupling in the bridge.
    # bridge/toolkg_patch.py::_discover_all_tools calls it; if it ever
    # disappears, replicate the path-prefix wrapping locally rather than
    # chasing renames.
    assert callable(getattr(mcp, "_wrap_with_path_prefix", None))
