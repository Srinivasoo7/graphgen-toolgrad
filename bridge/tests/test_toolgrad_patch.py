"""Unit tests for the ToolGrad KG patch (no API keys, no LLM calls).

The monkeypatch mechanics are tested against fake ``toolgrad.modules``
objects injected into ``sys.modules`` so no real ToolGrad import (and no
LLM backend) is needed. Template rendering is tested with the real
langchain ``ChatPromptTemplate`` when available, otherwise skipped.
"""

import sys
import types

from bridge.prompts.predict_workflow_kg import (
    PREDICT_WORKFLOW_KG_TEXT,
    TEMPLATE_VARIABLES,
    build_predict_workflow_kg,
)


def _langchain_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        return False


def test_template_text_has_both_variables():
    assert "{api_use_chains}" in PREDICT_WORKFLOW_KG_TEXT
    assert "{kg_context}" in PREDICT_WORKFLOW_KG_TEXT
    assert "Domain Knowledge" in PREDICT_WORKFLOW_KG_TEXT


def test_template_text_no_stray_braces():
    # Every {…} group must be one of the declared variables, otherwise
    # ChatPromptTemplate/str.format would choke or silently drop text.
    import re

    groups = set(re.findall(r"\{([^{}]+)\}", PREDICT_WORKFLOW_KG_TEXT))
    assert groups == set(TEMPLATE_VARIABLES), groups


def test_template_renders_with_fake_context():
    try:
        from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
    except ImportError:
        return "SKIP"
    template = build_predict_workflow_kg()
    assert set(template.input_variables) == set(TEMPLATE_VARIABLES)
    messages = template.format_messages(
        api_use_chains="<<CHAINS>>",
        kg_context="<<CTX>>",
    )
    text = "\n".join(m.content for m in messages)
    assert "<<CHAINS>>" in text
    assert "<<CTX>>" in text
    assert "{kg_context}" not in text  # variable actually substituted


def _install_fake_toolgrad_modules():
    """Shadow toolgrad.modules.{prompt_lib,module_lib} with fakes.

    Returns (saved, prompt_lib_fake, module_lib_fake); caller must restore
    via _restore_fake_toolgrad_modules(saved).
    """
    saved = {}
    for name in ("toolgrad", "toolgrad.modules",
                 "toolgrad.modules.prompt_lib", "toolgrad.modules.module_lib"):
        saved[name] = sys.modules.get(name, None)

    pkg = types.ModuleType("toolgrad")
    pkg.__path__ = []
    sub = types.ModuleType("toolgrad.modules")
    sub.__path__ = []
    prompt_lib = types.ModuleType("toolgrad.modules.prompt_lib")
    prompt_lib.PREDICT_WORKFLOW = "ORIGINAL_TEMPLATE"
    module_lib = types.ModuleType("toolgrad.modules.module_lib")

    def original_updater():
        return "ORIGINAL_UPDATER"

    module_lib.create_workflow_updater = original_updater

    sys.modules["toolgrad"] = pkg
    sys.modules["toolgrad.modules"] = sub
    sys.modules["toolgrad.modules.prompt_lib"] = prompt_lib
    sys.modules["toolgrad.modules.module_lib"] = module_lib
    return saved, prompt_lib, module_lib, original_updater


def _restore_fake_toolgrad_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_apply_and_remove_patch_with_fake_modules():
    if not _langchain_available():
        return "SKIP"  # apply_kg_patch builds the real template object
    from bridge import toolgrad_patch

    saved, prompt_lib, module_lib, original_updater = _install_fake_toolgrad_modules()
    try:
        assert not toolgrad_patch.is_patched()
        toolgrad_patch.apply_kg_patch("CTX")
        assert toolgrad_patch.is_patched()
        # Template swapped…
        assert prompt_lib.PREDICT_WORKFLOW is toolgrad_patch.get_patched_template()
        assert set(prompt_lib.PREDICT_WORKFLOW.input_variables) == {"api_use_chains", "kg_context"}
        # …and the updater factory replaced (not called: would need an LLM).
        assert module_lib.create_workflow_updater is not original_updater

        toolgrad_patch.remove_kg_patch()
        assert not toolgrad_patch.is_patched()
        assert prompt_lib.PREDICT_WORKFLOW == "ORIGINAL_TEMPLATE"
        assert module_lib.create_workflow_updater is original_updater
    finally:
        _restore_fake_toolgrad_modules(saved)
        # Defensive: never leak a patched state into other tests.
        toolgrad_patch._PATCHED = False
        toolgrad_patch._ORIGINALS.clear()


def test_apply_patch_is_idempotent():
    if not _langchain_available():
        return "SKIP"  # apply_kg_patch builds the real template object
    from bridge import toolgrad_patch

    saved, prompt_lib, module_lib, original_updater = _install_fake_toolgrad_modules()
    try:
        toolgrad_patch.apply_kg_patch("CTX-1")
        toolgrad_patch.apply_kg_patch("CTX-2")  # re-apply must not stack
        assert toolgrad_patch.is_patched()
        toolgrad_patch.remove_kg_patch()
        assert prompt_lib.PREDICT_WORKFLOW == "ORIGINAL_TEMPLATE"
        assert module_lib.create_workflow_updater is original_updater
    finally:
        _restore_fake_toolgrad_modules(saved)
        toolgrad_patch._PATCHED = False
        toolgrad_patch._ORIGINALS.clear()
