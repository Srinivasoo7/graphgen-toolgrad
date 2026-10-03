# Crux integration

The combined product has two bounded responsibilities:

| Layer | Responsibility |
| --- | --- |
| Crux | Identity, workspace/environment isolation, authorization, policy checks, approvals, audit, revocation, agent access, and governed export |
| GraphGen–ToolGrad bridge | Context extraction, scenario generation, executable tool chains, verification, refinement, training/evaluation datasets, and failure-derived regression data |

## Request lifecycle

```text
request
  -> authenticate principal
  -> authorize context, environment, and action in Crux
  -> create or update a scenario run in the bridge
  -> execute tools in the approved boundary
  -> verify intent, outputs, side effects, and privacy
  -> create evidence and an artifact version
  -> route export or production-impacting actions through Crux approval
  -> audit the decision and result
```

## Product objects

Crux governs:

- workspace
- principal (human or agent)
- environment
- policy
- approval request
- export target
- audit event

The bridge owns:

- context snapshot
- business rule
- scenario pack
- scenario
- synthetic dataset
- execution trace
- verification evidence
- run ledger

## Safety rules

- Raw production rows remain inside the configured customer boundary.
- Chat requests become structured proposals; they do not silently mutate data.
- Generated artifacts are versioned and reversible.
- Production execution and sensitive exports require explicit approval.
- A successful tool call is not sufficient for release; intent, outcome, policy,
  privacy, and expert-acceptance gates must be recorded.
- The bridge must not implement a third authorization layer.

## Current composition

Crux is included under `platform/crux/` as a bounded subproject. Its existing
Docker, CLI, operator-console, ontology, audit, and test assets are preserved.
The next integration step is an adapter that lets the bridge request Crux
authorization and attach Crux evidence identifiers to scenario runs and
exports.
