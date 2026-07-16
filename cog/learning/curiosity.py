"""Intrinsic motivation: Cog proposes and runs its own experiments.

Every engine so far reacts to tasks the world hands Cog. The Curiosity Engine
is the first that acts on its own initiative: it looks at the knowledge graph,
finds where Cog is *most uncertain*, and proposes the experiment that would
reduce that uncertainty the most — classic active learning / intrinsic
motivation, grounded in the evidence Cog already has.

Uncertainty is scored, not guessed:

- **Skills** by the Bernoulli entropy of their confidence — a skill sitting at
  0.5 is maximally uncertain (one more test is maximally informative); a skill
  at 0.99 or 0.01 is settled.
- **Representations** that were *rejected but close* to the acceptance bar — a
  little more evidence might flip them.
- **Domains** by sparsity — a domain with few experiences is under-explored.

For skills whose program is a single deterministic calculator step, the engine
does more than propose: it **synthesizes a runnable probe** — a fresh goal plus
the answer it can compute itself — so ``CogRuntime.explore`` can actually run
the experiment and fold the result back in. That closes the loop: Cog notices
its own uncertainty, designs an experiment, runs it, and learns.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cog.execution.tools import CalculatorTool
from cog.memory.router import MemoryRouter

# Novel arithmetic probes the engine can pose to a "Compute {p0}" calculator
# skill (and score itself, since the calculator is deterministic).
_PROBE_EXPRESSIONS = ["7 + 7", "9 * 3", "100 - 58", "12 * 12", "(4 + 5) * 2", "81 / 9"]


def _entropy(p: float) -> float:
    """Bernoulli entropy in bits; 0 at the extremes, 1.0 at p=0.5."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class ExperimentProposal:
    target_id: str
    kind: str  # "skill" | "representation" | "domain"
    reason: str
    info_gain: float  # higher = more uncertainty to resolve
    suggested_goal: str | None = None  # a runnable probe, when one can be synthesized
    expected_output: object | None = None  # the answer, when the engine can compute it


@dataclass
class ExplorationOutcome:
    """The result of running one self-proposed experiment."""

    proposal: ExperimentProposal
    experience: object  # the Experience produced by the probe
    confidence_before: float
    confidence_after: float

    @property
    def uncertainty_reduced(self) -> bool:
        return self.experience.verified and _entropy(self.confidence_after) < _entropy(
            self.confidence_before
        )


class CuriosityEngine:
    def __init__(self, min_gain: float = 0.05) -> None:
        self.min_gain = min_gain

    def propose(self, memory: MemoryRouter) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []

        for record in memory.skills.search(limit=200):
            if {"retired", "compressed"} & set(record.tags):
                continue
            gain = _entropy(record.confidence)
            if gain < self.min_gain:
                continue
            goal, expected = self._synthesize_probe(record)
            proposals.append(
                ExperimentProposal(
                    target_id=record.id,
                    kind="skill",
                    reason=f"skill confidence {record.confidence:.2f} is uncertain "
                    f"(entropy {gain:.2f})",
                    info_gain=round(gain, 4),
                    suggested_goal=goal,
                    expected_output=expected,
                )
            )

        for record in memory.concepts.search(tags=["searched"], limit=200):
            metrics = record.content.get("metrics", {})
            holdout = float(metrics.get("holdout_coverage", 0.0))
            null = float(metrics.get("null_coverage", 0.0))
            # a rejected candidate that only just failed is worth more evidence
            if 0.0 < holdout - null < 0.2:
                gain = 0.2 - (holdout - null)
                proposals.append(
                    ExperimentProposal(
                        target_id=record.id,
                        kind="representation",
                        reason=f"representation is near the acceptance bar "
                        f"(holdout {holdout:.2f} vs null {null:.2f})",
                        info_gain=round(gain, 4),
                    )
                )

        for record in memory.concepts.search(tags=["domain"], limit=200):
            members = record.content.get("members", [])
            gain = 1.0 / (1 + len(members))
            if gain < self.min_gain:
                continue
            proposals.append(
                ExperimentProposal(
                    target_id=record.id,
                    kind="domain",
                    reason=f"domain '{record.content.get('name')}' is sparse "
                    f"({len(members)} experiences)",
                    info_gain=round(gain, 4),
                )
            )

        proposals.sort(key=lambda p: -p.info_gain)
        return proposals

    def _synthesize_probe(self, skill_record) -> tuple[str | None, object | None]:
        """If the skill is a single deterministic calculator step over a goal
        template like 'Compute {p0}', pose a fresh expression and compute the
        answer ourselves — a runnable, self-verifiable experiment."""
        content = skill_record.content
        steps = content.get("steps", [])
        template = content.get("goal_template", "")
        params = content.get("parameters", [])
        if (
            len(steps) != 1
            or steps[0].get("tool") != "calculator"
            or steps[0].get("args", {}).get("expression") != "{p0}"
            or params != ["p0"]
            or "{p0}" not in template
        ):
            return None, None
        # pick a probe expression deterministically from the skill id
        digest = sum(ord(c) for c in skill_record.id)
        expression = _PROBE_EXPRESSIONS[digest % len(_PROBE_EXPRESSIONS)]
        goal = re.sub(r"\{p0\}", expression, template)
        try:
            expected = CalculatorTool().run(expression=expression)
        except (ValueError, SyntaxError):
            return None, None
        return goal, expected
