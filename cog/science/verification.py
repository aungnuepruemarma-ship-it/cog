"""FormalVerifier: an authorized evidence producer for formal proofs.

A formal proof is NOT an experiment -- it establishes correctness by
construction, not by evaluating evidence under a protocol. So it gets its
own evidence claim_type ("formal_verification") and its own producer, rather
than being forced through the experiment runner.

Authority model (per the lab audit): no module may construct an evidence
record directly. Only registered producers may. This module is registered as
"formal_verifier" in cog.science.ledger.EVIDENCE_PRODUCERS, so it is the
sole legitimate source of "formal_verification" claims. The promotion gate
consumes it exactly like an experiment claim -- it verifies provenance and
policy, never how the evidence was produced.

Usage:
    verifier = FormalVerifier(ledger)
    rec = verifier.verify(
        subject_id="compress:Y",
        hypothesis="X reproduces Y exactly on its goal",
        proof= lambda: check_equality(X, Y),   # returns True iff provably equal
        treatment_id="compress:Y", baseline_id="compress:X",
    )
    # then promote via ledger.promote_claim(experiment_id=rec.id, ...)
"""

from __future__ import annotations

from typing import Any, Callable

from cog.science.ledger import Ledger


class FormalVerifier:
    """Authorized producer of formal_verification evidence records."""

    AUTHORITY_KEY = "formal_verifier"

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def verify(
        self,
        *,
        subject_id: str,
        hypothesis: str,
        proof: Callable[[], bool],
        treatment_id: str,
        baseline_id: str,
        experiment: str = "formal proof by construction",
        dataset: list[str] | None = None,
        claim_id: str | None = None,
    ) -> Any:
        """Run a deterministic proof; record a formal_verification evidence claim.

        The proof must be reproducible and deterministic -- it returns True iff
        the artifact is provably correct. On success the evidence claim carries
        passed_policy=True, status=completed, reproducible=True. The promotion
        gate later verifies this provenance exactly as it would for an experiment.
        """
        holds = bool(proof())
        claim = self.ledger.record_claim(
            subject_id=subject_id,
            hypothesis=hypothesis,
            experiment=experiment,
            dataset=dataset or [],
            metrics={
                "evidence_class": "formal",
                "proof_result": holds,
                "reproducible": True,
            },
            decision="adopted" if holds else "rejected",
            confidence=1.0 if holds else 0.0,
            reproducible=True,
            claim_type="formal_verification",
            # Sole authorized producer of this claim type.
            _evidence_authority=self.AUTHORITY_KEY,
            meta={
                "status": "completed",
                "passed_policy": holds,
                "treatment_id": treatment_id,
                "baseline_id": baseline_id,
                "policy_reason": "formal proof holds" if holds else "formal proof failed",
            },
            claim_id=claim_id,
        )
        return claim
