# Ponytail review — bridge/ (2026-09-15)

Revalidation pass over the Phase 1–4 bridge code, per the ponytail-review
format. Applied findings are already cut; `bridge/tests/run_tests.py`
is 70/70 green after the cuts. `docs/` prose and `bridge/tests/` coverage
were out of scope (a smoke test is the ponytail minimum, not bloat).

## Findings (all applied)

`bridge/kg_sampler.py:L51-60`: delete: `max_steps`/`steps` bookkeeping plus the
"unreachable in practice" safety-net block. Nothing replaces it — every loop
iteration appends exactly one unvisited node, so the loop provably fills.

`bridge/toolkg_patch.py:L138`: yagni: `vary_per_call` param and the
`_VARY_PER_CALL` global. Config nobody sets; the sampler now always varies
per call (the upstream RNG-reseed quirk fix). `Nothing` replaces the toggle.

`bridge/trace_to_qa.py:L215`: yagni: `working_dir` / `op_name` constructor
params. GraphGen-convention cargo — set, never read anywhere.

`bridge/trace_to_qa.py:L298`: shrink: `[base] * 1` → `[base]`.

`bridge/eval_harness.py:L102-108`: shrink: nested ternary for
`fraction_chain_verified` → plain if/else. Same three outcomes, readable.

`bridge/refinement_loop.py:L253`: shrink: redundant `attempt` pre-init dict —
`.get(id(qa), 0)` on the next line already handled misses. `{}` suffices.

net: -14 lines possible. Cut: -14.

## Kept deliberately (judgment calls, not cut)

- `load_graphml` / `load_from_working_dir` (kg_context_exporter) and
  `load_workflow_sample` / `load_tracer` (trace_to_qa): the public loading
  API for real GraphGen/ToolGrad outputs — the live run's input contract,
  not scaffolding. The `<namespace>.graphml` naming convention is real
  knowledge worth one function.
- `MockToolExecutor` + `ToolGradExecutor` (chain_verifier): two
  implementations, but the mock is what makes the 70 keyless tests possible.
- `TraceToQAOperator._toolkg_edges_used` vs
  `chain_verifier.chain_toolkg_coverage`: count (provenance schema, tested)
  vs fraction (quality gate). Reusing one via float round-trip would be
  worse than 5 honest lines.
- `RefineConfig` / `MixConfig` hand-rolled `as_dict`: dataclass conversion
  is a readability wash; skipped.
- `TOOLKG_VERSION` stamp on serialized ToolKGs: 2 lines, standard for a
  persistence format.
- `_schema_problems` bool/int special-case (chain_verifier): trust-boundary
  validation, tested — awkward but correct; not touched.
- `uniform_sample` (kg_sampler): only the test baseline, but it is the
  comparison the sampler tests measure against.
- `estimate_tokens` char-based heuristic: documented approximation, one line.
- `get_patched_template` memoization (toolgrad_patch): tiny, and a test
  asserts identity against the patched attribute.
- Tests: fixtures already shared via `tests/_fixtures.py`; per-file
  `_fake_tool` / `_catalog` helpers are minimal. No pure duplication found.
