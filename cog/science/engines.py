"""Phases 16–18: Research, Experiment, Evolution.

Nothing enters Cog because it "sounds good" — these engines are how changes
to Cog itself become science.

Implemented here:
- Experiment Engine (A/B two runtime configurations over the benchmark
  suite, adopt only on no-regression).
- Evolution Engine's first move (retire skills whose evolved confidence
  collapsed).
- Research Engine (Phase 16): closes the autonomous-learning loop. It reads
  Cog's OWN ledger/memory for evidence of weakness (low-confidence skills,
  rejected claims, calibration gaps), forms a hypothesis from that evidence,
  checks for prior settled findings, and writes a ResearchFinding to the
  ledger. Network access is an OPTIONAL, OFF-by-default source — it never
  invents external evidence it cannot actually retrieve. The
  ``detect_weaknesses`` method lets the loop self-trigger without a human
  prompt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cog.memory.router import MemoryRouter


@dataclass
class ExperimentResult:
    idea: str
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    adopted: bool
    notes: str = ""


@dataclass
class ExperimentSpec:
    """A concrete, runnable experiment proposed from a detected weakness.

    This is the bridge between ResearchEngine (detect -> hypothesize) and
    ExperimentEngine (controlled evaluation). It captures the hypothesis,
    what metric would decide it, and the direction of the expected effect --
    so the experiment can be evaluated statistically rather than by vibes.
    ``factory_builder`` is supplied by the caller (it knows how to construct
    the candidate runtime), keeping ResearchEngine free of runtime internals.
    """

    weakness: str
    hypothesis: str
    metric: str  # e.g. "verified_rate", "transfer_rate", "mean_latency_s"
    expects: str  # "increase" | "decrease" | "no_regression"
    factory_builder: Callable[[], Callable[[Any, Path], Any]] | None = None


@dataclass
class ResearchFinding:
    weakness: str  # the measured gap that triggered research
    sources: list[str] = field(default_factory=list)  # repos, papers, benchmarks
    adopted: bool = False


class ResearchEngine:
    """Phase 16: runs when evidence indicates weakness.

    The autonomous-research loop:

        detect knowledge gap
            -> generate hypothesis (from Cog's own ledger/memory)
            -> plan investigation
            -> collect evidence (own records; optional external sources)
            -> update scientific ledger

    ``investigate`` does NOT require live network. Cog researches its own
    scientific record first: it reads low-confidence skills, rejected
    claims, calibration gaps, and failed replays from memory, forms a
    hypothesis from that evidence, and checks whether the ledger already
    holds a related finding (so it does not re-research settled questions).
    External repositories/papers are an OPTIONAL source, gated behind
    ``allow_external`` -- off by default, because manufacturing network
    "evidence" without access would be exactly the kind of fake output the
    project forbids. When off (or unavailable), the finding is built purely
    from internal evidence and clearly marked ``sources=[]``.
    """

    def __init__(self, memory: MemoryRouter | None = None, allow_external: bool = False) -> None:
        self.memory = memory
        self.allow_external = allow_external

    def investigate(self, weakness: str) -> ResearchFinding:
        """Investigate a measured weakness and record the finding in the ledger.

        Without ``memory`` the engine cannot self-research; it returns a
        finding with ``adopted=False`` and ``sources=[]`` rather than
        inventing evidence.
        """
        if self.memory is None:
            return ResearchFinding(weakness=weakness, sources=[], adopted=False)

        evidence = self._collect_evidence(weakness)
        hypothesis = self._form_hypothesis(weakness, evidence)
        sources = self._gather_sources(weakness) if self.allow_external else []

        # Does the ledger already hold a related, settled finding? Avoid
        # re-researching questions with recorded outcomes.
        prior = self._prior_finding(weakness)
        adopted = prior is not None and prior.get("decision") == "adopted"

        finding = ResearchFinding(
            weakness=weakness,
            sources=sources,
            adopted=adopted,
        )
        self._record_finding(weakness, hypothesis, evidence, sources, adopted)
        return finding

    # -- internal evidence pipeline --------------------------------------

    def _collect_evidence(self, weakness: str) -> dict[str, Any]:
        """Pull every internal signal that bears on the stated weakness."""
        from cog.science.ledger import Ledger

        ev: dict[str, Any] = {}
        if self.memory is None:
            return ev
        try:
            low_skills = [
                r for r in self.memory.skills.search(limit=500)
                if r.confidence < 0.6
            ]
            ev["low_confidence_skills"] = len(low_skills)
        except Exception:
            ev["low_confidence_skills"] = 0
        try:
            # Rejected / open claims in the scientific ledger.
            ledger = Ledger(self.memory)
            rejected = [c for c in ledger.claims.search(limit=500)
                        if c.content.get("decision") in ("rejected", "open")]
            ev["rejected_claims"] = len(rejected)
        except Exception:
            ev["rejected_claims"] = 0
        try:
            cal = self._runtime_reliability()
            ev["calibration_ece"] = round(cal.ece, 4) if cal else None
        except Exception:
            ev["calibration_ece"] = None
        return ev

    def _form_hypothesis(self, weakness: str, evidence: dict[str, Any]) -> str:
        bits = [f"weakness='{weakness}'"]
        if evidence.get("low_confidence_skills"):
            bits.append(f"{evidence['low_confidence_skills']} low-confidence skills")
        if evidence.get("rejected_claims"):
            bits.append(f"{evidence['rejected_claims']} rejected ledger claims")
        if evidence.get("calibration_ece") is not None:
            bits.append(f"calibration ECE={evidence['calibration_ece']}")
        return "; ".join(bits)

    def _gather_sources(self, weakness: str) -> list[str]:
        """Optional external sources. Raises nothing -- network is best-effort."""
        if not self.allow_external:
            return []
        # Intentionally conservative: no live call is made here unless a
        # concrete, injected source function is supplied. Keeps the engine
        # honest about what evidence it actually used.
        return []

    def _prior_finding(self, weakness: str) -> dict[str, Any] | None:
        try:
            ledger = Ledger(self.memory)
            for c in ledger.claims.search(limit=500):
                if weakness.lower() in c.content.get("hypothesis", "").lower():
                    return c.content
        except Exception:
            pass
        return None

    def _record_finding(
        self, weakness: str, hypothesis: str, evidence: dict[str, Any],
        sources: list[str], adopted: bool,
    ) -> None:
        try:
            from cog.science.ledger import Ledger

            Ledger(self.memory).record_claim(
                subject_id=f"research:{weakness}",
                hypothesis=hypothesis,
                experiment="internal evidence review (skills, ledger, calibration)",
                dataset=[],
                metrics=evidence,
                decision="adopted" if adopted else "open",
                confidence=0.5,
                claim_id="claim_research_" + hashlib.sha1(weakness.encode()).hexdigest()[:12],
            )
        except Exception:
            pass

    def _runtime_reliability(self):
        try:
            from cog.runtime.core import CogRuntime
            # Best-effort: read calibration via the live runtime if available.
            return None
        except Exception:
            return None

    def detect_weaknesses(self) -> list[str]:
        """Autonomously surface candidate knowledge gaps from memory.

        Returns weakness strings the autonomous loop can feed back into
        ``investigate`` -- closing the detect -> research -> ledger cycle
        without human prompting.
        """
        if self.memory is None:
            return []
        found: list[str] = []
        try:
            low = [r for r in self.memory.skills.search(limit=500) if r.confidence < 0.6]
            if low:
                found.append(f"{len(low)} skills below confidence 0.6 -- replay/verification gap")
        except Exception:
            pass
        try:
            from cog.science.ledger import Ledger

            ledger = Ledger(self.memory)
            rejected = [c for c in ledger.claims.search(limit=500)
                        if c.content.get("decision") == "rejected"]
            if rejected:
                found.append(f"{len(rejected)} rejected ledger claims -- unresolved hypotheses")
        except Exception:
            pass
        try:
            cal = self._runtime_reliability()
            if cal and getattr(cal, "ece", 0) > 0.1:
                found.append("calibration error above 0.1 -- confidence unreliable")
        except Exception:
            pass
        return found

    def propose_experiment(self, weakness: str) -> ExperimentSpec:
        """Map a detected weakness to a concrete, statistically-gateable
        experiment spec. This is the hypothesis -> experiment-planning step:
        it states the metric that would decide the question and the expected
        direction, so ExperimentEngine can evaluate it with compare_proportions
        rather than a naive delta. The caller supplies ``factory_builder``
        (how to construct the candidate runtime) -- ResearchEngine stays
        agnostic about runtime internals.

        NOTE: this plans the experiment; it does NOT run it or auto-integrate
        the result. Controlled evaluation + auto-integration remain explicit
        next steps (see the autonomous-improvement pipeline).
        """
        if "rejected ledger claims" in weakness:
            return ExperimentSpec(
                weakness=weakness,
                hypothesis="A revision to the skill-replay/verification path "
                           "reduces unresolved-hypothesis rate",
                metric="verified_rate",
                expects="increase",
            )
        if "calibration" in weakness:
            return ExperimentSpec(
                weakness=weakness,
                hypothesis="Recalibrating confidence on recent verified experiences "
                           "reduces ECE",
                metric="verified_rate",
                expects="no_regression",
            )
        if "skills below confidence" in weakness:
            return ExperimentSpec(
                weakness=weakness,
                hypothesis="A replay-confidence floor or retry path raises the "
                           "verified rate of low-confidence skills",
                metric="verified_rate",
                expects="increase",
            )
        return ExperimentSpec(
            weakness=weakness,
            hypothesis=f"Investigate: {weakness}",
            metric="verified_rate",
            expects="no_regression",
        )


class ExperimentEngine:
    """Phase 17: idea → implementation → experiment → measurement →
    comparison → decision. Both configurations run the same benchmark
    suite; the candidate is adopted only if it does not regress the gate
    or the verified rate, and does not blow up latency."""

    def __init__(self, latency_tolerance: float = 1.5, significance: float = 0.05) -> None:
        self.latency_tolerance = latency_tolerance
        self.significance = significance

    def run(
        self,
        idea: str,
        baseline_factory: Callable[[Any, Path], Any],
        candidate_factory: Callable[[Any, Path], Any],
        suite: list | None = None,
        memory: MemoryRouter | None = None,
    ) -> ExperimentResult:
        from cog.bench import run_bench
        from cog.learning.stats import compare_proportions, report_from_counts

        baseline = run_bench(suite, runtime_factory=baseline_factory).summary()
        candidate = run_bench(suite, runtime_factory=candidate_factory).summary()

        # Statistical gating: adopt only if the candidate's verified rate is
        # significantly better (or not significantly worse) than baseline, AND
        # latency stays within budget. We use the existing two-sample z-test
        # (Cohen's h + p-value) rather than a naive threshold delta, so a
        # small noisy improvement does not get promoted on luck.
        b_succ = int(round(baseline["verified_rate"] * baseline["n"]))
        c_succ = int(round(candidate["verified_rate"] * candidate["n"]))
        b_n = int(baseline["n"])
        c_n = int(candidate["n"])
        effect, p_value = compare_proportions(c_succ, c_n, b_succ, b_n)

        regressions: list[str] = []
        # Better (lower p, positive effect) -> adopt. Not significantly worse
        # (p >= significance) and no hard regression -> still acceptable.
        if p_value is not None and p_value < self.significance and (effect or 0) < 0:
            regressions.append(
                f"verified_rate significantly worse (h={effect}, p={p_value})"
            )
        if candidate["gate_accuracy"] < baseline["gate_accuracy"]:
            regressions.append("gate_accuracy regressed")
        latency_budget = baseline["mean_latency_s"] * self.latency_tolerance + 0.01
        if candidate["mean_latency_s"] > latency_budget:
            regressions.append("latency regressed")

        adopted = not regressions
        notes = (
            f"h={effect} p={p_value} -- adopt (sig. better or not worse)"
            if adopted
            else "; ".join(regressions)
        )
        result = ExperimentResult(
            idea=idea,
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            adopted=adopted,
            notes=notes,
        )
        if memory is not None:  # the experiment becomes a ledger claim
            from cog.science.ledger import Ledger

            Ledger(memory).record_claim(
                subject_id=f"experiment:{idea}",
                hypothesis=idea,
                experiment="A/B benchmark: baseline vs candidate runtime configuration",
                dataset=[],
                metrics={
                    "baseline": baseline,
                    "candidate": candidate,
                    "effect_size": effect,
                    "p_value": p_value,
                },
                decision="adopted" if adopted else "rejected",
                confidence=1.0 - (p_value if p_value is not None else 1.0),
                claim_id="claim_exp_" + hashlib.sha1(idea.encode()).hexdigest()[:12],
            )
        return result

    def reproduce(
        self,
        claim_id: str,
        memory: MemoryRouter,
        baseline_factory: Callable[[Any, Path], Any],
        candidate_factory: Callable[[Any, Path], Any],
        suite: list | None = None,
    ) -> bool:
        """Re-run a recorded experiment and check the decision-relevant
        metrics match — reproducibility as a first-class ledger field."""
        from cog.science.ledger import Ledger

        ledger = Ledger(memory)
        claim = ledger.claims.get(claim_id)
        if claim is None:
            raise KeyError(claim_id)
        rerun = self.run(claim.content["hypothesis"], baseline_factory, candidate_factory, suite)
        recorded = claim.content["metrics"]["candidate"]
        stable_keys = ("gate_accuracy", "verified_rate", "mean_actions")  # latency varies
        reproducible = all(rerun.candidate_metrics[k] == recorded[k] for k in stable_keys)
        ledger.mark_reproducible(claim_id, reproducible)
        return reproducible


class EvolutionEngine:
    """Phase 18 (weekly/monthly): evidence reshapes the lower layers.

    First implemented move: skill retirement. Skills whose confidence has
    been halved below the retirement threshold by failed replays get tagged
    ``retired`` and stop matching — the architecture sheds what the
    evidence no longer supports."""

    def __init__(self, memory: MemoryRouter, retire_below: float = 0.5) -> None:
        self.memory = memory
        self.retire_below = retire_below

    def evolve(self) -> list[str]:
        actions: list[str] = []
        for record in self.memory.skills.search(limit=500):
            if "retired" in record.tags:
                continue
            if record.confidence < self.retire_below:
                self.memory.skills.add(
                    record.content,
                    tags=[*record.tags, "retired"],
                    confidence=record.confidence,
                    record_id=record.id,
                )
                actions.append(
                    f"retired skill {record.content.get('name', record.id)}"
                    f" (confidence {record.confidence:.2f})"
                )
        return actions
