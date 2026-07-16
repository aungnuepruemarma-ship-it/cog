"""Corroboration: trust an answer that has no declared expectation by
cross-checking independent methods.

The verification gate's ``OutputCheck`` can only compare an output against an
expectation the *task* supplied. When a goal arrives with none — the common
real case — the check auto-passes, and a wrong answer (a drifted skill, a model
slip) sails straight into trusted memory. Belief revision cleans that up *after*
the fact, once a contradiction accumulates. Corroboration is the complementary
move: catch it *before* the write, with no ground truth at all.

The idea is N-version agreement. Cog solves the same goal two structurally
independent ways —

- **skill_replay**: deterministically replay an induced program;
- **model_plan**:   ask the model to plan from scratch —

and treats their *agreement* as evidence. The two are not perfectly independent
(the skill was compiled from past model plans), but at run time they are
different mechanisms with different failure modes: a drifted skill and a correct
model plan disagree, and that disagreement is exactly the signal the gate could
not otherwise see. Agreement across k methods with independent error rates
``eᵢ`` leaves a joint-wrong probability of ``∏ eᵢ`` — so corroborated confidence
rises above any single method, while a disagreement is surfaced as untrusted
rather than silently believed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def canonical(output: object) -> str:
    """Order-insensitive equality key for two methods' outputs."""
    try:
        return json.dumps(output, sort_keys=True, default=str)
    except TypeError:
        return repr(output)


@dataclass
class Corroboration:
    """The result of cross-checking one goal across independent methods."""

    goal: str
    methods: list[str] = field(default_factory=list)
    outputs: list[object] = field(default_factory=list)
    priors: list[float] = field(default_factory=list)  # per-method accuracy prior

    @property
    def agreed(self) -> bool:
        if len(self.outputs) < 2:
            return False
        first = canonical(self.outputs[0])
        return all(canonical(o) == first for o in self.outputs)

    @property
    def corroborated(self) -> bool:
        """True only when >=2 independent methods produced the same answer."""
        return len(self.methods) >= 2 and self.agreed

    @property
    def output(self) -> object | None:
        return self.outputs[0] if self.outputs else None

    @property
    def confidence(self) -> float:
        """Agreement of independent methods: 1 - ∏(1 - priorᵢ). A disagreement
        (or too few methods) is untrusted -> 0.0."""
        if not self.corroborated:
            return 0.0
        joint_wrong = 1.0
        for prior in self.priors:
            joint_wrong *= max(0.0, 1.0 - prior)
        return round(1.0 - joint_wrong, 4)

    def summary(self) -> str:
        pairs = ", ".join(f"{m}->{o!r}" for m, o in zip(self.methods, self.outputs, strict=False))
        if self.corroborated:
            return f"{self.goal!r}: {len(self.methods)} methods agree [{pairs}]"
        if len(self.methods) < 2:
            return f"{self.goal!r}: only {len(self.methods)} method available — cannot corroborate"
        return f"{self.goal!r}: methods DISAGREE [{pairs}] — answer untrusted"
