"""The central product artifact: one scenario, many export shapes.

A ScenarioPackage is the single object every downstream consumer
transforms: Copilot evaluation datasets, agent training traces, CI test
cases, TDM data requests, regression fixtures. The exporters in
bridge/exporters.py are pure transforms over this contract.

The approval records are the seam where the Crux governance plane will
attach: `evidence_id` is the identifier Crux returns for an approval
decision. Until the Crux adapter lands, approvals stay empty and
"expert acceptance" is recorded via `expert_review`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from bridge.models.scenario import (
    ExpectedOutcome,
    ExpertReview,
    VerificationEvidence,
)

CATEGORIES = ("normal", "edge", "security", "recovery", "adversarial")


@dataclass
class ApprovalRecord:
    """One approval decision on a scenario package.

    The Crux adapter will fill `evidence_id` with the identifier Crux
    returns for the approval request, tying the decision to the audit
    trail. The role names mirror the approval inbox: business, security,
    technical.
    """

    approver: str
    role: str = ""  # business | security | technical
    decision: str = "approved"  # approved | rejected
    decided_at: str = ""
    evidence_id: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioPackage:
    """One business scenario, fully specified and verified.

    - category: the coverage bucket this scenario serves. A generator
      that deliberately bids across normal/edge/security/recovery is
      the coverage planner; until then the operator tags the category.
    - failure_mode: what goes wrong, e.g. "authorization_denied",
      "tool_error", "duplicate_request", "expired_window". None when the
      scenario is a happy path.
    - tool_chain: executed steps as {tool, tool_input, result_preview}.
    - verification: the verifier's business checks.
    - expert_review / approvals: the release evidence. The bridge emits
      the former; Crux will own the latter.
    """

    scenario_id: str
    intent: str
    intent_type: str = "unknown"  # create | update | read | close | unknown
    category: str = "normal"
    entities: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    tool_chain: List[Dict[str, Any]] = field(default_factory=list)
    expected_outcomes: List[ExpectedOutcome] = field(default_factory=list)
    expected_answer: str = ""
    failure_mode: Optional[str] = None
    verification: VerificationEvidence = field(default_factory=VerificationEvidence)
    expert_review: Optional[ExpertReview] = None
    approvals: List[ApprovalRecord] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(
                f"category must be one of {CATEGORIES}, got {self.category!r}"
            )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioPackage":
        data = dict(data)
        data["expected_outcomes"] = [
            o if isinstance(o, ExpectedOutcome) else ExpectedOutcome(**o)
            for o in data.get("expected_outcomes", [])
        ]
        data["verification"] = _as_verification(data.get("verification"))
        data["expert_review"] = _as_expert_review(data.get("expert_review"))
        data["approvals"] = [
            a if isinstance(a, ApprovalRecord) else ApprovalRecord(**a)
            for a in data.get("approvals", [])
        ]
        return cls(**data)

    @classmethod
    def from_qa(cls, qa: Dict[str, Any], report: Optional[Dict[str, Any]]) -> "ScenarioPackage":
        """Build a package from a generated QA pair and its verifier report.

        The bridge's current output shape: QA dict (question, answer_draft,
        chain, entity_refs, expected_outcomes, provenance) plus the
        verification report from chain_verifier.verify_qa. Anything the QA
        does not carry stays at its default: category defaults to "normal",
        failure_mode to None, approvals to [].
        """
        report = report or {}
        provenance = dict(qa.get("provenance", {}))
        intent_type = qa.get("intent_type") or _question_intent(
            str(qa.get("question", ""))
        )
        return cls(
            scenario_id=str(
                qa.get("scenario_id")
                or provenance.get("chain_id")
                or "scenario-0"
            ),
            intent=str(qa.get("question", "")),
            intent_type=intent_type,
            category=str(qa.get("category", "normal")),
            entities=list(qa.get("entity_refs", [])),
            business_rules=list(qa.get("business_rules", [])),
            initial_state=dict(qa.get("initial_state", {})),
            tool_chain=list(qa.get("chain", [])),
            expected_outcomes=[
                o if isinstance(o, ExpectedOutcome) else ExpectedOutcome(**o)
                for o in qa.get("expected_outcomes", [])
            ],
            expected_answer=str(qa.get("answer_draft", "")),
            failure_mode=qa.get("failure_mode"),
            verification=VerificationEvidence(
                execution_completed=bool(report.get("executed_steps")),
                intent_aligned=bool(report.get("intent_aligned")),
                output_assertions_passed=bool(report.get("output_assertions_passed")),
                state_assertions_passed=bool(report.get("state_assertions_passed")),
                policy_checks_passed=bool(report.get("policy_checks_passed")),
                answer_faithful=bool(report.get("answer_faithful")),
                safe_to_review=bool(report.get("safe_to_review")),
            ),
            expert_review=_as_expert_review(qa.get("expert_review")),
            approvals=[
                a if isinstance(a, ApprovalRecord) else ApprovalRecord(**a)
                for a in qa.get("approvals", [])
            ],
            provenance=provenance,
        )


def _question_intent(question: str) -> str:
    try:
        from bridge.verification.intent import question_intent

        found = question_intent(question)
        return found.lower() if found else "unknown"
    except Exception:
        return "unknown"


def _as_verification(data: Any) -> VerificationEvidence:
    if isinstance(data, VerificationEvidence):
        return data
    if isinstance(data, dict):
        known = {
            k: v
            for k, v in data.items()
            if k in VerificationEvidence.__dataclass_fields__
        }
        return VerificationEvidence(**known)
    return VerificationEvidence()


def _as_expert_review(data: Any) -> Optional[ExpertReview]:
    if data is None or isinstance(data, ExpertReview):
        return data
    if isinstance(data, dict):
        known = {
            k: v for k, v in data.items() if k in ExpertReview.__dataclass_fields__
        }
        if "scenario_id" not in known or "decision" not in known:
            return None
        return ExpertReview(**known)
    return None
