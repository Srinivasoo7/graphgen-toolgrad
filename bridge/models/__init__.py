"""Canonical records for candidate → verification → acceptance → release.

A scenario candidate is a proposal. Machine verification records whether
the tools ran and whether the business result held. Expert review is a
separate decision. A dataset release contains only accepted rows.
"""

from bridge.models.scenario import (
    BusinessRule,
    ContextArtifact,
    DatasetRelease,
    ExpectedOutcome,
    ExpertReview,
    HoldoutResult,
    ProcessStep,
    QAFixture,
    ScenarioCandidate,
    SourceRecord,
    ToolAction,
    TrainingRow,
    VerificationEvidence,
)

__all__ = [
    "BusinessRule",
    "ContextArtifact",
    "DatasetRelease",
    "ExpectedOutcome",
    "ExpertReview",
    "HoldoutResult",
    "ProcessStep",
    "QAFixture",
    "ScenarioCandidate",
    "SourceRecord",
    "ToolAction",
    "TrainingRow",
    "VerificationEvidence",
]
