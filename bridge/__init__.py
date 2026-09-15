"""graphgen-toolgrad bridge package.

Integrates Sri's forks — ``Srinivasoo7/toolgrad`` and ``Srinivasoo7/GraphGen``
(both on branch ``integrate/graphgen-toolgrad``) — into a knowledge-grounded
tool-use data factory, through public seams (no monkeypatching):

Phase 1: export a GraphGen knowledge graph as compact domain context and
inject it into ToolGrad's query/response synthesis via the fork's native
``kg_context`` variable on ``PREDICT_WORKFLOW`` (``kg_context_exporter``).

Phase 2: build a ToolKG (knowledge graph over the tool catalog, edges =
composability) and plug KG-neighborhood sampling into the fork's
``sampler`` seam (``toolkg_builder``, ``kg_sampler``, ``toolkg_sampler``).

Phase 3: convert ToolGrad execution traces into tool-grounded QA pairs in
GraphGen's SFT format (``trace_to_qa`` operator, ``chain_verifier``). The
same operator is registered natively in the GraphGen fork as
``graphgen.operators.trace_qa.TraceQAService``.

Phase 4: unified refinement loop (``refinement_loop``: generate -> verify ->
filter -> textual-gradient refine, with a run ledger), SFT dataset assembly
(``sft_mix``), and a keyless dataset-level eval harness (``eval_harness``).
"""

__version__ = "0.5.0"
