"""The promotion gate: provenance verification for runtime adoption.

Architecture (per the lab audit, final form):

    Search discovers candidates.
        -> record FINDING
    Experiments generate evidence.
        -> ExperimentManager records EXPERIMENT (passed_policy, treatment_id)
    Policy decides if evidence is sufficient.   (lives in the experiment subsystem)
    Ledger records the decision.
    Runtime adopts the artifact.                (only via promote_claim)

This module is the LAST link. It does NOT perform hypothesis testing and
does NOT know how to compare variants -- that belongs to the experiment
subsystem (cog/science/experiment.py, which uses cog/learning/stats.py).
The gate only verifies PROVENANCE:

    - the referenced experiment_id exists as a claim_type == "experiment"
    - it completed (status == "completed")
    - it passed policy (passed_policy == True)
    - its treatment_id matches the artifact being promoted
    - it has not been superseded (no superseded_by)

If any check fails, PromotionDenied is raised. The caller must then record
a plain FINDING (decision/claim_type) instead of adopting -- never bypass.

claim_type disambiguates events permanently:
    "finding"    -- a scientific conclusion / observation / candidate
    "experiment" -- a completed experiment with a policy decision
    "promotion"  -- an artifact entering the runtime (gated)

Only "promotion" claims are governed. Findings and experiments stay free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cog.memory.router import MemoryRouter

if TYPE_CHECKING:
    from cog.science.ledger import Ledger


class PromotionDenied(Exception):
    """Raised when an artifact promotion fails provenance verification.

    The correct recovery is to record a FINDING instead of adopting -- never
    to bypass the gate or to manufacture a fake experiment claim.
    """


def promote_claim(
    ledger: "Ledger",
    *,
    subject_id: str,
    hypothesis: str,
    experiment: str,
    dataset: list[str],
    metrics: dict[str, Any],
    confidence: float,
    experiment_id: str,
    baseline_id: str,
    treatment_id: str,
    claim_id: str | None = None,
) -> Any:
    """The ONLY sanctioned path for adopting an artifact into the runtime.

    Verifies the referenced experiment's provenance (it exists, completed,
    passed policy, treatment matches, not superseded) and, on success,
    records a claim_type==promotion claim with full provenance. On failure
    raises PromotionDenied -- the caller records a FINDING instead.
    """
    from cog.science.ledger import Ledger, _PROTECTED_EVIDENCE_KINDS  # local: breaks cycle

    exp = ledger.claims.get(experiment_id)
    if exp is None:
        raise PromotionDenied(f"referenced experiment {experiment_id} does not exist")
    c = exp.content
    m = c.get("metrics", {})
    meta = c.get("meta", {})
    # Compatibility resolver for experiment governance fields. Canonical home
    # is ``meta`` (top-level), but tolerate records that nested these under
    # ``metrics`` during the transition.
    status = meta.get("status", m.get("status"))
    passed = meta.get("passed_policy", m.get("passed_policy"))
    treatment = meta.get("treatment_id", m.get("treatment_id"))
    superseded = meta.get("superseded_by", m.get("superseded_by"))

    if c.get("claim_type") not in _PROTECTED_EVIDENCE_KINDS:
        raise PromotionDenied(
            f"{experiment_id} is not an authorized evidence claim "
            f"(claim_type={c.get('claim_type')!r}; accepted={sorted(_PROTECTED_EVIDENCE_KINDS)})"
        )
    if status != "completed":
        raise PromotionDenied(f"experiment {experiment_id} did not complete")
    if not passed:
        raise PromotionDenied(f"experiment {experiment_id} did not pass promotion policy")
    if treatment != treatment_id:
        raise PromotionDenied(
            f"experiment treatment {treatment} != promoted artifact {treatment_id}"
        )
    if superseded:
        raise PromotionDenied(f"experiment {experiment_id} was superseded")

    full_metrics = {
        **metrics,
        "experiment_id": experiment_id,
        "baseline_id": baseline_id,
        "treatment_id": treatment_id,
    }
    return ledger.record_claim(
        subject_id=subject_id,
        hypothesis=hypothesis,
        experiment=experiment,
        dataset=dataset,
        metrics=full_metrics,
        decision="adopted",
        confidence=confidence,
        claim_type="promotion",
        claim_id=claim_id,
        _via_promotion_gate=True,  # authorized: provenance gate already passed above
    )
