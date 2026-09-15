"""Version-controlled copy of ToolGrad's PREDICT_WORKFLOW prompt with KG grounding.

Source: ``toolgrad/modules/prompt_lib.py`` (``PREDICT_WORKFLOW``,
zhongyi-zhou/toolgrad @ c9544f84). The original text is reproduced verbatim
except for the marked ``KG-CONTEXT`` insertion, which adds a ``{kg_context}``
variable plus a grounding instruction. Diffing this file against the upstream
template should show exactly one inserted block.
"""
