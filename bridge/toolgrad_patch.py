"""Apply the KG-context patch to ToolGrad without editing the upstream checkout.

Recommended path — runtime monkeypatch::

    from bridge import kg_context_exporter, toolgrad_patch
    from bridge.prompts.predict_workflow_kg import PREDICT_WORKFLOW_KG_TEXT  # noqa

    context = kg_context_exporter.export_kg_context(graph)
    toolgrad_patch.apply_kg_patch(kg_context_exporter.render_kg_context(context))
    # ... build and invoke the ToolGrad graph as usual ...
    toolgrad_patch.remove_kg_patch()  # optional cleanup

Why this works without touching upstream code:

* ``toolgrad.modules.module_lib.create_workflow_updater`` reads
  ``prompt_lib.PREDICT_WORKFLOW`` via module-attribute lookup *at call time*,
  so replacing the attribute is enough to swap the template.
* The replacement rebuilds the updater exactly like the original
  (``prompt | llm.with_structured_output(states.WorkflowQueryResponse)``)
  but pre-binds ``kg_context`` via ``ChatPromptTemplate.partial``. The
  existing call site in ``graph_lib.inverse_predictor`` —
  ``updater.invoke({"api_use_chains": chains_new})`` — keeps working
  unchanged because ``api_use_chains`` remains a free variable.

Alternative path — manual edit of the upstream checkout (documented in
``docs/phase1-bridge.md``): copy ``PREDICT_WORKFLOW_KG_TEXT`` into
``toolgrad/modules/prompt_lib.py`` and pass ``kg_context`` at the invoke
site in ``graph_lib.py``.
"""

from __future__ import annotations

from .prompts.predict_workflow_kg import build_predict_workflow_kg

_ORIGINALS: dict = {}
_PATCHED = False
_CACHED_TEMPLATE = None


def get_patched_template():
    """The KG-grounded ``ChatPromptTemplate`` (built lazily so the template
    *text* stays importable without the langchain stack)."""
    global _CACHED_TEMPLATE
    if _CACHED_TEMPLATE is None:
        _CACHED_TEMPLATE = build_predict_workflow_kg()
    return _CACHED_TEMPLATE


def is_patched() -> bool:
    """Whether the KG patch is currently applied."""
    return _PATCHED


def apply_kg_patch(kg_context: str) -> None:
    """Patch ToolGrad's inverse predictor to ground queries in ``kg_context``.

    ``kg_context`` is the rendered text from
    :func:`bridge.kg_context_exporter.render_kg_context`. Must be called
    before the ToolGrad graph's ``inverse_predictor`` node runs (i.e. before
    ``app.invoke(...)``). Idempotent: re-applying replaces the context.
    """
    import toolgrad.modules.module_lib as module_lib
    import toolgrad.modules.prompt_lib as prompt_lib

    global _PATCHED
    if _PATCHED:
        remove_kg_patch()

    _ORIGINALS["PREDICT_WORKFLOW"] = prompt_lib.PREDICT_WORKFLOW
    _ORIGINALS["create_workflow_updater"] = module_lib.create_workflow_updater

    prompt_lib.PREDICT_WORKFLOW = get_patched_template()

    def create_workflow_updater_with_kg():
        # Mirrors the original create_workflow_updater, but pre-binds
        # kg_context so the existing invoke({"api_use_chains": ...}) call
        # site in graph_lib.inverse_predictor needs no change.
        from toolgrad.modules import states
        from toolgrad.utils import langchain as langchain_utils

        prompt = prompt_lib.PREDICT_WORKFLOW.partial(kg_context=kg_context)
        llm = langchain_utils.create_llm()
        return prompt | llm.with_structured_output(states.WorkflowQueryResponse)

    module_lib.create_workflow_updater = create_workflow_updater_with_kg
    _PATCHED = True


def remove_kg_patch() -> None:
    """Restore the original ToolGrad prompt and updater (undoes the patch)."""
    import toolgrad.modules.module_lib as module_lib
    import toolgrad.modules.prompt_lib as prompt_lib

    global _PATCHED
    if not _PATCHED:
        return
    prompt_lib.PREDICT_WORKFLOW = _ORIGINALS["PREDICT_WORKFLOW"]
    module_lib.create_workflow_updater = _ORIGINALS["create_workflow_updater"]
    _ORIGINALS.clear()
    _PATCHED = False
