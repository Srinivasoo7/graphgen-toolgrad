# ADR: Tabular foundation model as the analysis stage (`kg_source: tables`)

**Status:** Proposed, 2026-09-30; Kumo Tabular backend adopted 2026-10-03. Decision owner: Sri.
**Question:** where, if anywhere, does a tabular FM sit in the factory pipeline?

## Addendum (2026-10-03): TabPFN removed; Kumo is the backend

- TabPFN was the original candidate: implemented and keyless-tested the
  same day the ADR was written (`bridge/tabpfn_probe.py`, same findings
  contract), but never ran live — the package stopped cleanly at the
  Prior Labs license gate, and its 2.5/3 checkpoints are non-commercial.
- At Sri's direction, NVIDIA's Kumo Tabular (released Sep 29, 2026) was
  validated live instead: `bridge/kumo_probe.py` mirrors the TabPFN
  findings contract with raw-row estimators.
- Live run on CPU (this box has no GPU): small/28M checkpoint,
  OpenMDW-1.1 weights confirmed on the Hugging Face listing itself,
  no license click-through, no token. 400-row synthetic tickets
  table with a known noisy priority->status relationship: holdout
  accuracy 0.80 (Bayes-optimal for the injected noise), permutation
  importance correctly isolated `priority` as the only predictive
  feature (0.325; all noise columns 0.0), TSTR parity 1.0.
- Debugging notes for reproducers: the sdm TableTensor blocks raw
  aten ops, so predictions are decoded via the output columns, which
  the model names by class value; and bracketed IPv6 literals in
  `no_proxy` break the vendored httpx2 URL parser during the weight
  download (export cleaned values first).
- With Kumo validated, the TabPFN module was deleted from this repo.
  The tabular probe stage is now Kumo-only; shared plumbing
  (loading, column stats, narration) lives in `bridge/probe_common.py`.
  Commercial read: Kumo's OpenMDW-1.1 weights are the licensing-clean
  path for a commercial product.

## Context

The product thesis (whiteboard, 2026-09-30) is two failure loops:

1. **The privacy loop.** Production data is needed to test, privacy blocks it,
   the POC passes on fake data, the system fails in production, and the need
   for production data returns.
2. **The fidelity loop.** Naive synthetic data breaks the first loop but not
   the second: it still fails in production because it does not encode the
   company's **business rules and enterprise context**.

The factory answers the fidelity loop: KG-grounded, execution-verified,
intent-aligned, expert-accepted rows. But its inputs today are a tool surface
(MCP schemas), a document corpus, or a hand-authored spec KG. The largest
store of enterprise business context — **relational tables** (tickets,
orders, customers, claims) — has no input path.

TabPFN is a candidate for that path. What it actually is: a tabular
foundation model that performs supervised prediction by in-context learning —
feature matrix plus labels in, predictions out, no gradient updates. What it
is not: it has no text generation and no explanation interface. It cannot
read a table and write a document about what it understood.

## Decision

1. **Add a tabular input path, `domain.kg_source: tables`.** The pipeline
   ordering is **TabPFN → KG → ToolGrad**:
   - Sample the approved tables.
   - A TabPFN probe stage computes structured findings: which columns
     predict which outcomes, conditional distributions, class balance,
     relationship strengths.
   - The configured LLM **narrates those findings into domain documents**
     (no raw rows in the narration input — findings and aggregates only).
   - The existing GraphGen KG extraction consumes the documents unchanged;
     ToolGrad and every downstream gate are untouched.
2. **TabPFN also serves as a synthetic-data quality gate** for tabular
   outputs: train-on-synthetic-test-on-real (TSTR) parity. Synthetic rows
   that break the real data's predictive relationships fail the gate, the
   same way a misaligned intent fails verification.
3. **TabPFN is not a front door for all inputs and does not write the
   understanding doc.** It measures; the LLM narrates; the KG structures;
   ToolGrad executes. Each stage does the job its output type fits.

## Alternatives considered

- **TabPFN as the universal front door** ("all data passes through TabPFN,
  it writes a doc, the doc feeds everything"). Rejected: output-type
  mismatch. A predictor's output is predictions; KG construction consumes
  entities, relations, and semantics. The doc-writing step would still need
  an LLM, so the front door adds a stage without removing one.
- **TabPFN as world-state simulator** (answering "what would this lookup
  return" during chain generation, STATEGEN-style). Deferred, not rejected:
  it leans on TabPFN's generative/synthetic support, which the 2026-09-23
  landscape review noted but this project has not verified. Verify first;
  if it holds, this becomes a separate ADR.
- **No tabular path.** Rejected: it leaves databases — where most enterprise
  context actually lives — outside the factory, and cedes the ground to
  test-data vendors (K2view, Delphix) whose synthetic output carries rules
  but no verified behavior.

## Whiteboard review notes folded into this decision

- **The verification loop is inside the engine, not a post-filter.** The
  gates (chain execution, intent alignment, expected outcomes, expert
  acceptance) are what make "business-rules-grounded" a guarantee rather
  than a label. Rows from `tables` pass the same gates as rows from any
  other source; this ADR adds an input, it does not lower the bar.
- **The outer loop is future work:** production failures feed back into the
  engine as new scenarios (failure → generated scenario → verified training
  data). It spans all pillars and gets its own design when scheduled.
- **Process flows are a candidate first-class source** (`kg_source: flows` —
  BPMN diagrams, runbooks, incident timelines), distinct from the document
  corpus. Separate ADR when taken up.
- **"EA tools" means the entity inventory:** CMDB / service catalog systems.
  If a deployment has one, it is the authoritative entity list the KG should
  reconcile against.
- **ToolGrad's evolution toward SIT** (system integration testing) synthetic
  data stands: verified chains double as integration-test scenarios. That is
  an output adapter (a DevOps emitter), not a pipeline change, and is
  tracked separately.

## Privacy constraint

Synthetic is not private by default, and this design names its leak surface:
the LLM narration step. Constraints:

- Narration input is TabPFN findings and aggregates only — never raw rows.
- Narration output passes the existing PII redaction and secret scan before
  KG extraction.
- TabPFN's in-context learning does not persist customer rows into model
  weights; probe artifacts (findings JSON) are run artifacts and inherit the
  run directory's handling.
- Before any regulated-domain claim, the project needs an explicit position
  on memorization / differential privacy. This ADR does not make that claim.

## Consequences and sequencing

1. **Prerequisite already open:** the generator must emit
   `expected_outcomes` (see the 457aa5f review) before `safe_to_review` is
   load-bearing. Tabular rows are subject to the same gate; there is no
   point feeding a new source into a gate with no key.
2. **Validation still owed:** the post-fix ITSM re-run (needs
   `OPENROUTER_API_KEY` via the supported env path). TabPFN work does not
   substitute for measuring the current pipeline's yield.
3. **Verify before building:** TabPFN v2 input limits (on the order of
   10k rows × 500 features per in-context pass — sampling strategy needed
   for production tables), license terms for commercial use (modified
   Apache per the 2026-09-23 review), and the generative-support claim
   behind the deferred simulator option.
4. **Implementation sketch (when approved):** `bridge/tabpfn_probe.py`
   (probe stage emitting a findings JSON), a `tables:` config block under
   `domain.kg_source: tables`, narration reusing the existing corpus path,
   and keyless tests with a stub probe — mirroring the Jev filter's
   discipline: probe failure is an explicit failure, never a silent pass.
