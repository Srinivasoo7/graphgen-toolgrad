# Enterprise Scenario Plant

This directory contains the enterprise control-plane components that sit around
the GraphGen–ToolGrad scenario and training-data engine.

## Current modules

- `crux/` — Crux Enterprise Ontology and Token Optimization packaging.
  Crux owns identity, enrollment, environment isolation, authorization,
  proposal/review, audit, and agent access.
- `../bridge/` — GraphGen–ToolGrad context grounding, executable tool-chain
  generation, verification, refinement, SFT assembly, and evaluation.

## Product boundary

Crux is the trust plane. GraphGen–ToolGrad is the scenario/data plane.

```text
user or agent
    -> Crux authorization, policy, audit, approval
    -> GraphGen–ToolGrad scenario generation and execution
    -> evidence and quality gates
    -> Crux approval and governed export
```

Crux must not become a second scenario engine, and the bridge must not create a
competing authorization system. Integrations should use explicit contracts and
keep raw enterprise data inside the customer-controlled execution boundary.

See [`docs/crux-integration.md`](../docs/crux-integration.md) for the planned
integration contract and operating model.
