# Phase 4 — Unified refinement loop, SFT mix, eval harness

Date: 2026-09-15. All claims below are from code actually read and
commands actually run on this box (2 vCPUs, 7.7 GB RAM, no GPU).
`GOOGLE_API_KEY` / `OPENAI_API_KEY` are absent, so the live LLM loop is
**not** run here; the keyless orchestration build + tests are the
deliverable. This is the last planned phase.

## The idea

Phases 1–3 built the pipes: KG context into ToolGrad (P1), ToolKG-guided
sampling (P2), traces out to verified QA pairs (P3). Phase 4 closes the
loop around them: generate → verify → filter → **refine with textual
critiques** (ToolGrad's textual-gradient pattern, where the "gradient" is
natural-language feedback about what's wrong), then merge everything into
one SFT mix and measure the dataset keylessly — no trained model needed.

## What was built

| File | Purpose |
|---|---|
| `bridge/refinement_loop.py` | `RefinementLoop`: runs the full pipeline on workflow samples. Each iteration scores every QA pair (`score_qa`: chain validity, ToolKG coverage, entity grounding), filters by `RefineConfig` thresholds, and regenerates failures with a rule-based textual critique appended to the prompt (`critique_qa` + `prompt_suffix` hook added to `TraceToQAOperator.generate_for_chain`). Per-iteration metrics go to a JSON run ledger (`save_ledger` / `load_ledger`). With `llm_fn=None` it runs generation + filtering once and stops (`no_llm_refinement`) — refinement is honestly an LLM operation. |
| `bridge/sft_mix.py` | `assemble()`: quality cutoffs → dedupe (normalized questions) → seeded shuffle → train/valid split → `sft_mix.jsonl` (ChatML `{"messages": [...]}` + `provenance` per row: entities, tool chain, verification status, coverage) + `write_dataset_card()`. Merges GraphGen's own SFT rows (ChatML or ShareGPT, normalized, tagged `source: "graphgen"`). |
| `bridge/eval_harness.py` | `evaluate()`: dataset-level metrics with no model — fraction chain-verified (needs an executor or precomputed verifications; reports `None` + a note rather than guessing), mean ToolKG coverage, entity-grounding rate, tool diversity, question/answer token stats, and a `parametric_answerability` heuristic flag (answer verbatim in KG context ⇒ possibly answerable without tools). Always reports its limits. |
| `bridge/tests/test_refinement_loop.py` | 8 tests |
| `bridge/tests/test_sft_mix.py` | 5 tests |
| `bridge/tests/test_eval_harness.py` | 6 tests |
| Suite | **70/70 green** via `../vendor/.venv_toolgrad/bin/python bridge/tests/run_tests.py` (repo root) |

## Architecture

```
workflow samples (ToolGrad) ──┐
                              ▼
                 TraceToQAOperator (P3) ── template or llm_fn
                              │
                    score_qa: verify_qa (P3) + coverage (P3) + entity grounding
                              │
                    filter_pairs (RefineConfig thresholds)
                        ┌───────┴───────┐
                     kept            rejected ── critique_qa ──► regenerate
                        │              (prompt_suffix)              │ (llm_fn)
                        │◄─────────────┴──────────────────────────┘
                        ▼ (≤ max_iterations)
                 kept pairs (+verification, +coverage)
                        │
                        ▼
              sft_mix.assemble ──► sft_mix.jsonl + dataset_card.md
                        │
                        ▼
              eval_harness.evaluate ──► dataset metrics (keyless)
```

## Run-ledger schema

```json
{
  "config": {"max_iterations": 3, "min_coverage": 0.5,
             "require_chain_valid": true, "require_entities": true, "seed": 42},
  "num_samples": 12,
  "gen_stats": {"num_samples": 12, "num_chains": 12, "num_qa_pairs": 12, ...},
  "iterations": [
    {"iteration": 0, "num_pairs": 12, "num_chain_valid": 9,
     "num_entity_grounded": 11, "mean_coverage": 0.71,
     "num_refined": 3, "num_kept": 9, "num_rejected": 3},
    {"iteration": 1, "num_pairs": 12, "num_chain_valid": 12, ...}
  ],
  "stopped_reason": "all_pass | max_iterations | no_llm_refinement"
}
```

## Dataset card template

`write_dataset_card(report, eval_metrics, path)` emits: composition counts
(in → cutoffs → dedupe → merged → train/valid), quality-gate settings, an
eval snapshot (if the harness ran), and a limits section. See
`bridge/sft_mix.py` for the exact template.

## Eval metrics and their limits

| Metric | Needs | Honest limit |
|---|---|---|
| `fraction_chain_verified` | executor or precomputed verifications | replays recorded inputs; doesn't check the answer summarizes results |
| `mean_toolkg_coverage` | ToolKG | schema-overlap heuristic from P2, not a semantic judge |
| `entity_grounding_rate` | KG context | substring matching, not entity linking |
| `tool_diversity` | — | counts, not quality |
| `length_stats` | — | token estimate (chars/4), not real tokenization |
| `parametric_answerability` | KG context | verbatim-overlap heuristic: flagged ⇒ suspect, unflagged ⇏ proven tool-dependent |

The harness deliberately measures the *dataset*, not the *model*. Whether
SFT on this data improves an agent needs the live fine-tune + BFCL /
knowledge-QA probe — that is the documented next step, not a claim.

## Validated keylessly vs needs the live run

**Validated here (70/70 tests):** loop orchestration on fixtures (ledger
records iterations; filtering drops invalid chains; refinement improves a
scripted bad→good case and records `refinement_attempt` + critique);
mix merge/dedupe/split/card; harness metrics on a hand-made dataset with
known properties (2/3 verified, 2/3 grounded, coverage means, flag hits).

**Needs the live LLM run:** everything about *quality* — whether critiques
actually improve questions, whether answers are faithful, whether SFT on
the mix beats baselines. The plumbing is ready; the judgment isn't.

## Live-run command sequence (once `GOOGLE_API_KEY` exists)

```bash
# 0. Env
export GOOGLE_API_KEY=...            # Gemini backend for ToolGrad + GraphGen
VENV=~/workspace/vendor/.venv_toolgrad

# 1. Real KG -> domain context (GraphGen run produces <wd>/<ns>.graphml)
$VENV/bin/python - <<'EOF'
from bridge import kg_context_exporter
g = kg_context_exporter.load_from_working_dir("/path/to/graphgen/working_dir")
ctx = kg_context_exporter.export_kg_context(g)
open("kg_context.json","w").write(kg_context_exporter.export_kg_context_json(g))
print(kg_context_exporter.render_kg_context(ctx)[:500])
EOF

# 2. Patched ToolGrad generation: KG-grounded prompts (P1) + ToolKG sampling (P2)
$VENV/bin/python - <<'EOF'
import json
from bridge import toolgrad_patch, toolkg_patch
ctx_text = open("kg_context.json").read()
toolgrad_patch.apply_kg_patch(ctx_text)   # {kg_context} in PREDICT_WORKFLOW
toolkg_patch.apply_toolkg_patch()          # neighborhood sampling
# ... run ToolGrad's examples/mcp_filesystem.py (or your graph) as usual ...
# outputs: seed=*.json samples + trace/*.json
toolkg_patch.remove_toolkg_patch()
toolgrad_patch.remove_kg_patch()
EOF

# 3. Traces -> refined, verified QA mix (P3 + P4)
$VENV/bin/python - <<'EOF'
import glob, json
from bridge import refinement_loop, sft_mix, eval_harness, chain_verifier, toolkg_patch
from bridge.refinement_loop import RefineConfig, RefinementLoop, save_ledger
from bridge.trace_to_qa import load_workflow_sample, load_tracer

def llm_fn(prompt):  # your backend here (Gemini via langchain)
    ...

samples = [load_workflow_sample(p) for p in sorted(glob.glob("seed=*.json"))]
tracers = [load_tracer(p) for p in sorted(glob.glob("trace/*.json"))]
kg_context = json.load(open("kg_context.json"))

loop = RefinementLoop(
    llm_fn=llm_fn, kg_context=kg_context,
    toolkg=toolkg_patch.get_toolkg(),
    executor=chain_verifier.ToolGradExecutor(toolkg_patch._TOOLS_BY_NAME),
    config=RefineConfig(max_iterations=3, min_coverage=0.5),
)
kept, ledger = loop.run(samples, tracers)
save_ledger(ledger, "ledger.json")
print(ledger["stopped_reason"], len(kept))

train, valid, report = sft_mix.assemble(kept)
sft_mix.write_jsonl(train, "sft_mix/train.jsonl")
sft_mix.write_jsonl(valid, "sft_mix/valid.jsonl")
metrics = eval_harness.evaluate(train + valid, kg_context=kg_context,
                                toolkg=toolkg_patch.get_toolkg())
sft_mix.write_dataset_card(report, metrics, "sft_mix/DATASET_CARD.md")
print(json.dumps({k: metrics[k] for k in
      ("fraction_chain_verified","mean_toolkg_coverage","entity_grounding_rate")}, indent=1))
EOF

# 4. (Next, not in this repo) SFT a small model on sft_mix/train.jsonl with
#    ToolGrad's train recipe (src/train/train_sft.py), then eval on BFCL +
#    a knowledge-QA probe. That result — not this doc — is the quality claim.
```

## Notes for whoever runs it live

- The critic is rule-based and cheap; the expensive part is the LLM
  regeneration calls (≤ `max_iterations` × rejected). Start with
  `max_iterations=2` on a small sample batch.
- `min_coverage=0.5` is a guess from the 5-tool MCP catalog (density 0.4).
  Recalibrate per catalog: log the coverage distribution from the ledger
  before filtering aggressively.
- Keep `GOOGLE_API_KEY` in the environment, never in files — the repo's
  ground rules forbid committed secrets, and the ledger/dataset card
  contain no keys.
