"""Canonical objects. One module so the shapes stay consistent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _dump(obj) -> dict:
    return asdict(obj)


@dataclass
class SourceRecord:
    source_id: str
    origin: str
    payload: Dict[str, Any] = field(default_factory=dict)
    holdout: bool = False

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ContextArtifact:
    artifact_id: str
    kind: str
    text: str = ""

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class BusinessRule:
    rule_id: str
    text: str

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ProcessStep:
    step_id: str
    name: str
    process_name: str = ""

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ToolAction:
    tool: str
    tool_input: Dict[str, Any] = field(default_factory=dict)
    effect: str = "READ"

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ExpectedOutcome:
    description: str
    assertion: str = ""
    state_after: str = ""

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ScenarioCandidate:
    scenario_id: str
    source_record_ids: List[str] = field(default_factory=list)
    process_step_ids: List[str] = field(default_factory=list)
    policy_ids: List[str] = field(default_factory=list)
    intent: str = ""
    preconditions: Dict[str, Any] = field(default_factory=dict)
    tool_actions: List[ToolAction] = field(default_factory=list)
    expected_outcomes: List[ExpectedOutcome] = field(default_factory=list)
    failure_variant: bool = False
    generation_provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class VerificationEvidence:
    """execution_completed means the calls ran. safe_to_review means the
    business result was confirmed. Those are different facts.
    """

    execution_completed: bool = False
    intent_aligned: bool = False
    output_assertions_passed: bool = False
    state_assertions_passed: bool = False
    policy_checks_passed: bool = False
    answer_faithful: bool = False
    safe_to_review: bool = False

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class ExpertReview:
    scenario_id: str
    decision: str
    reason_codes: List[str] = field(default_factory=list)
    corrected_question: Optional[str] = None
    corrected_answer: Optional[str] = None
    reviewer: str = ""
    reviewed_at: str = ""

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class TrainingRow:
    scenario_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class QAFixture:
    scenario_id: str
    case: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class DatasetRelease:
    release_id: str
    status: str
    training_rows: List[str] = field(default_factory=list)
    qa_fixtures: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return _dump(self)


@dataclass
class HoldoutResult:
    source_id: str
    withheld: bool = True

    def as_dict(self) -> dict:
        return _dump(self)
