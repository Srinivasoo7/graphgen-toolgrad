# Scenario package

The scenario package is the product's central artifact. One package
describes one business scenario — intent, entities, rules, the executed
tool chain, expected outcomes, verification evidence, and release
evidence. Every downstream output is a transform over this object:

- Copilot / agent evaluation datasets (`bridge/exporters/copilot_eval.py`)
- agent training traces (ChatML — existing `to_chatml` path)
- CI test cases, TDM data requests, regression fixtures (planned)

## Contract (`bridge/models/scenario_package.py`)

```python
ScenarioPackage(
    scenario_id=...,          # e.g. chain_0
    intent=...,               # business question, as asked
    intent_type=...,           # create | update | read | close | unknown
    category=...,              # normal | edge | security | recovery | adversarial
    entities=[...],
    business_rules=[...],
    initial_state={...},
    tool_chain=[{tool, tool_input, result_preview}, ...],
    expected_outcomes=[ExpectedOutcome(...)],
    expected_answer=...,
    failure_mode=...,          # None on the happy path
    verification=VerificationEvidence(...),   # from chain_verifier
    expert_review=...,         # human review, when present
    approvals=[ApprovalRecord(...)],          # Crux evidence, when wired
    provenance={...},
)
```

`ScenarioPackage.from_qa(qa, report)` builds a package from the bridge's
current output shape (generated QA dict + verifier report). Intent type
is resolved with the same mapping the release gate uses, normalized to
lowercase.

`ApprovalRecord.evidence_id` is the seam for the Crux adapter: the
identifier Crux returns for an approval decision. Until the adapter
lands, approvals stay empty and expert acceptance is recorded on
`expert_review` alone.

## Copilot evaluation exporter

`bridge/exporters/copilot_eval.py` emits one JSONL row per scenario:

```json
{
  "scenario_id": "chain_0",
  "category": "normal",
  "prompt": "Create a ticket for the VPN outage",
  "expected_answer": "Created: {...}",
  "expected_tool_calls": [{"tool": "create_ticket", "arguments": {"title": "VPN outage"}}],
  "expected_outcomes": ["New"],
  "business_rules": ["Tickets must have an owner before closing"],
  "ground_truth": {"intent_type": "create", "verified": true, "checks": {...}},
  "entities": ["Ticket"],
  "provenance": {"source": "toolgrad_trace", "chain_id": "chain_0"}
}
```

Rows carry aggregates and verified outcomes only — never raw rows.

## What's not here yet

- **Coverage planner**: category is operator-tagged today. A generator
  that deliberately bids across normal/edge/security/recovery is the
  next generator-side change.
- **Crux approvals**: wiring `ApprovalRecord` to real Crux approval
  decisions (the `integrations/crux/` adapter).
- **More exporters**: pytest/Playwright/Postman/Gherkin and TDM
  request transforms follow the same one-function-per-format pattern.
