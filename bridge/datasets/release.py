"""Decide whether machine-verified rows may be called a dataset release.

A release requires expert acceptance. Execution alone is not acceptance.
Holdout records are never part of the released training rows.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


class ReleasePolicy:
    def __init__(
        self,
        min_accepted_rows: int = 100,
        min_train_rows: int = 80,
        min_validation_rows: int = 10,
        min_failure_variants: int = 10,
        require_expert_review: bool = True,
    ) -> None:
        self.min_accepted_rows = min_accepted_rows
        self.min_train_rows = min_train_rows
        self.min_validation_rows = min_validation_rows
        self.min_failure_variants = min_failure_variants
        self.require_expert_review = require_expert_review


def assess_release(
    *,
    accepted_ids: Sequence[str],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    failure_variant_ids: Sequence[str] = (),
    holdout_ids: Sequence[str] = (),
    expert_decisions: int = 0,
    outcome_assertions: bool = False,
    privacy_clean: bool = True,
    policy: ReleasePolicy | None = None,
) -> Dict[str, Any]:
    """Return a release verdict. ``released`` is the only success status."""
    policy = policy or ReleasePolicy()
    reasons = []
    leaked = sorted(set(accepted_ids) & set(holdout_ids))
    if leaked:
        reasons.append("holdout leakage")
    if not privacy_clean:
        reasons.append("privacy failure")
    if not outcome_assertions:
        reasons.append("no outcome assertions")
    if policy.require_expert_review and expert_decisions <= 0:
        reasons.append("no expert decisions")
    if len(accepted_ids) < policy.min_accepted_rows:
        reasons.append("insufficient accepted rows")
    if len(train_ids) < policy.min_train_rows:
        reasons.append("insufficient training rows")
    if len(validation_ids) < policy.min_validation_rows:
        reasons.append("insufficient validation rows")
    if len(failure_variant_ids) < policy.min_failure_variants:
        reasons.append("insufficient failure variants")

    if leaked or not privacy_clean:
        status = "failed"
    elif not accepted_ids:
        status = "machine_verified" if outcome_assertions else "generated"
    elif reasons:
        status = "awaiting_review" if "no expert decisions" in reasons else "accepted"
    else:
        status = "released"
    return {
        "status": status,
        "reasons": reasons,
        "n_accepted": len(accepted_ids),
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
    }
