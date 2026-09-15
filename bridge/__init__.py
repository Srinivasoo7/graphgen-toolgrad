"""graphgen-toolgrad bridge package.

Phase 1: export a GraphGen knowledge graph as compact domain context and
inject it into ToolGrad's query/response synthesis (``PREDICT_WORKFLOW``).

Phase 2: build a ToolKG (knowledge graph over the tool catalog, edges =
composability) and replace ToolGrad's uniform API sampling with
KG-neighborhood sampling (``toolkg_builder``, ``kg_sampler``,
``toolkg_patch``).

Phase 3: convert ToolGrad execution traces into tool-grounded QA pairs in
GraphGen's SFT format (``trace_to_qa`` operator, ``chain_verifier``).

Phase 4: unified refinement loop (``refinement_loop``: generate -> verify ->
filter -> textual-gradient refine, with a run ledger), SFT dataset assembly
(``sft_mix``), and a keyless dataset-level eval harness (``eval_harness``).
"""

__version__ = "0.4.0"
