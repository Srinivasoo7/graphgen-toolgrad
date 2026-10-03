"""Copilot evaluation dataset exporter.

Transforms ScenarioPackage objects into JSONL rows for agent-evaluation
harnesses (Microsoft Copilot evaluation, internal copilots, custom
evaluators). One row per scenario:

    {
      "scenario_id": ...,
      "category": "normal | edge | security | recovery | adversarial",
      "failure_mode": ... | null,
      "prompt": "<business question>",
      "expected_answer": "<answer draft>",
      "expected_tool_calls": [{"tool": ..., "arguments": {...}}, ...],
      "expected_outcomes": ["<assertion>", ...],
      "business_rules": ["<rule>", ...],
      "ground_truth": {
        "intent_type": ...,
        "verified": <safe_to_review>,
        "checks": {"intent_aligned": ..., ...}
      },
      "entities": [...],
      "provenance": {...}
    }

Rows carry only aggregates and verified outcomes — never raw production
rows.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

from bridge.models.scenario_package import ScenarioPackage

_CHECK_FIELDS = (
    "intent_aligned",
    "output_assertions_passed",
    "state_assertions_passed",
    "policy_checks_passed",
    "answer_faithful",
)


def expected_tool_calls(pkg: ScenarioPackage) -> List[Dict]:
    return [
        {"tool": step.get("tool"), "arguments": step.get("tool_input", {})}
        for step in pkg.tool_chain
    ]


def eval_row(pkg: ScenarioPackage) -> Dict:
    calls = expected_tool_calls(pkg)
    return {
        "scenario_id": pkg.scenario_id,
        "category": pkg.category,
        "failure_mode": pkg.failure_mode,
        "prompt": pkg.intent,
        "expected_answer": pkg.expected_answer,
        "expected_tool_calls": calls,
        "expected_outcomes": [o.assertion for o in pkg.expected_outcomes],
        "business_rules": list(pkg.business_rules),
        "ground_truth": {
            "intent_type": pkg.intent_type,
            "verified": pkg.verification.safe_to_review,
            "checks": {
                name: bool(getattr(pkg.verification, name))
                for name in _CHECK_FIELDS
            },
        },
        "entities": list(pkg.entities),
        "provenance": dict(pkg.provenance),
    }


def to_jsonl(packages: Iterable[ScenarioPackage]) -> str:
    lines = [json.dumps(eval_row(pkg), ensure_ascii=False) for pkg in packages]
    return "\n".join(lines) + ("\n" if lines else "")


def write_jsonl(packages: Iterable[ScenarioPackage], path: str) -> str:
    content = to_jsonl(packages)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
