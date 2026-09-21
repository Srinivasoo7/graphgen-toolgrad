# Jev filter spike (experiment, 2026-09-20)

**Question:** can TypeSafe's Jev (System One decision model, `jev-1.13.0`) serve
as a learned quality gate behind the Phase 4 refinement loop, replacing or
augmenting the keyless heuristic gates (`passes_thresholds`)?

## What was built

- `bridge/jev_filter.py` — `JevDecider` (asks Jev one `choice` accept/reject +
  one `noul` groundedness question per scored pair; subprocesses the
  `typesafe` skill's `jev_decide` CLI so the bridge never touches the API
  key) and `jev_filter_pairs`, a drop-in with the same return contract as
  `refinement_loop.filter_pairs`. Kept pairs carry a `qa["jev"]` annotation
  (verdict, P(accept), latency, model, source) for the dataset card.
  Any provider failure falls back to the heuristic gates, never to blind accept.
- `bridge/scripts/ab_jev.py` — A/B harness: 12 labeled pairs spanning the
  heuristic axes plus defects heuristics can't see (empty / contradictory /
  fluff answers); agreement table, stability re-runs, latency and cost.
- `bridge/tests/test_jev_filter.py` — 5 keyless tests (stub decider; fallback
  tested with a broken CLI path, no network).

## Result (n=12, synthetic fixtures, one domain)

Agreement **9/12 (75%)**. Jev agreed with every heuristic reject (P(accept)
0.00–0.24) and every clean accept (0.66–0.85). All 3 disagreements went one
way — heuristic accepts that Jev rejected, exactly the defect classes the
keyless gates cannot see:

| pair | heuristic | Jev P(accept) |
|---|---|---|
| empty_answer | accept | 0.05 |
| contradictory_answer | accept | 0.02 |
| fluff_answer | accept | 0.10 |

Stability: P(accept) spread ≤ 0.08 across 3 runs per pair, zero verdict flips.
Latency: p50 ~775ms, max ~1.1s per decision (vendor claims 70–500ms; measured
here includes subprocess + auth exchange from this box). Cost: ~$0.0006 for
24 calls (~570 input tokens/call at $0.042/MTok, output free).

## Gotchas found while building

- The top-level `confidence` on a Choice answer is a **margin**, not P(choice)
  (observed: choice=accept, confidence=0.50, P(accept)=0.75). The parser uses
  the `probabilities` map; responses without it are treated as unusable and
  fall back to heuristics.
- The `noul` groundedness question must ask about *specific proper nouns vs
  generic references* — asking "does it name a listed entity" just parrots the
  `entity_refs` the keyless scorer already computed.

## Recommendation

Jev works as a **second-stage filter** (heuristics first, Jev on the survivors
or on heuristic-accepts): on this probe it added recall on bad pairs without
dropping a good one, at negligible cost. Do not use it as the *only* gate —
n=12 is a smoke test, not a calibration; real thresholds need labeled pairs
from the live loop. Next step if this graduates: optional `filter_fn` hook on
`RefinementLoop` defaulting to `filter_pairs`.
