"""Primitive Discovery: Cog invents a better primitive than the ones it knows.

Every engine so far *composes* primitives that already exist — skills chain
tools, representations group genes, theories frame flows. None of them can ask
the question at the top of the ladder: *are these two fragments that always
appear together actually one primitive I haven't named yet?* That is the move
from learning skills to learning the **organization** of skills.

This engine mines Cog's own execution logs for reasoning fragments — adjacent
tool n-grams — that recur *together*, and proposes each frequent one as a
candidate primitive: a single named operation that would replace the fragment
sequence everywhere it occurs (``perceive`` for ``observe → gather``, say). A
candidate is *proposed* cheaply and *promoted* almost never: promotion demands
every quality gate *and* one of two breadth paths, and any shortfall rejects it,
with the failing gate recorded. Nothing is invented because it looks elegant.

The gates — the whole point of the engine. A candidate needs **breadth** (one of
two alternative paths) *and* every quality gate:

    (generality OR reuse) AND compression AND predictive
      AND efficiency AND interpretability AND reversibility

- **compression**  — corpus description shrinks: it eliminates
  ``occurrences × (n-1)`` fragment-instances from the logs;
- **generality** (breadth path 1) — it recurs across two or more *distinct
  emergent domains*: a broad primitive that *transfers*;
- **reuse** (breadth path 2) — it recurs across ``>= N`` *distinct goals /
  parameterizations* inside one domain: a tight primitive that *repeatedly
  compresses* one domain. This is the path a genuinely useful within-domain
  primitive (``perceive = observe ▸ gather``, which is intra-perception) takes,
  since a flow-fixed n-gram can never span two flow-clustered domains;
- **predictive power** — workflows that use it verify at a high rate;
- **efficiency** — each invocation is cheaper: it collapses ``n`` steps to one,
  an ``(n-1)/n`` action saving;
- **interpretability** — the definition is short enough for a human to read
  (a bounded n-gram, named);
- **reversibility** — *removing* it demonstrably hurts: the workflows that use
  it verify strictly better than those that do not, so reverting to the
  unmerged fragments is a measurable loss.

Splitting breadth into ``generality OR reuse`` keeps the benchmark strict — the
quality gates (predictive, reversibility) still guard every promotion — while no
longer being structurally blind to intra-domain primitives. On today's
single-tool, deterministic benchmark almost nothing clears these bars, and that
is the honest, intended result: a promotion, when the evidence finally earns one,
means something because the gate turned so many candidates away.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cog.experience.record import Experience
from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger

if TYPE_CHECKING:
    from cog.science.pipeline import Candidate

FlowFn = Callable[[Experience], list[str]]
DomainFn = Callable[[Experience], str | None]

# Deliberately strict. Promotion should be rare and earned.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "compression": 8.0,  # >= 8 fragment-instances eliminated from the corpus
    "generality": 2.0,  # breadth path 1: recurs across >= 2 distinct emergent domains
    "reuse": 5.0,  # breadth path 2: recurs across >= 5 distinct goals in one domain
    "predictive": 0.9,  # >= 90% of workflows using it verify
    "efficiency": 0.3,  # >= 30% action saving per invocation
    "interpretability": 4.0,  # definition is an n-gram of length <= 4
    "reversibility": 0.0,  # using it verifies STRICTLY better than not (> 0)
}

# Quality gates that must ALL pass; breadth (generality OR reuse) is separate.
_MANDATORY = ("compression", "predictive", "efficiency", "interpretability", "reversibility")


def _default_flow(experience: Experience) -> list[str]:
    return [step["tool"] for step in experience.execution if step.get("tool")]


@dataclass
class GateResult:
    name: str
    value: float
    threshold: float
    passed: bool


@dataclass
class PrimitiveCandidate:
    """A proposed merged primitive and the evidence for (or against) it."""

    fragments: tuple[str, ...]  # the adjacent tool n-gram
    occurrences: int  # adjacent appearances across the corpus
    support_ids: list[str]  # experiences that contain it
    domains: frozenset[str]  # distinct domains those experiences span
    used_verified: int  # supporters that verified
    other_total: int  # experiences that do NOT contain it
    other_verified: int  # of those, how many verified
    distinct_goals: int = 0  # distinct goals/parameterizations among supporters

    @property
    def n(self) -> int:
        return len(self.fragments)

    @property
    def compression(self) -> float:
        return float(self.occurrences * (self.n - 1))

    @property
    def efficiency(self) -> float:
        return (self.n - 1) / self.n if self.n else 0.0

    @property
    def generality(self) -> int:
        return len(self.domains)

    @property
    def reuse(self) -> int:
        """Distinct goals the fragment recurs across — breadth *within* a domain,
        independent of raw occurrences (which one repeated goal could inflate)."""
        return self.distinct_goals

    @property
    def predictive(self) -> float:
        return self.used_verified / len(self.support_ids) if self.support_ids else 0.0

    @property
    def baseline_predictive(self) -> float:
        return self.other_verified / self.other_total if self.other_total else 0.0

    @property
    def reversibility(self) -> float:
        """Counterfactual value: how much better workflows using it verify than
        those that do not. > 0 means removing it (reverting to the fragments)
        is a measurable loss."""
        return round(self.predictive - self.baseline_predictive, 4)

    @property
    def id(self) -> str:
        digest = hashlib.sha1(">".join(self.fragments).encode()).hexdigest()[:12]
        return f"prim_{digest}"

    def name(self) -> str:
        return "primitive: " + " ▸ ".join(self.fragments)

    def gates(self, thresholds: dict[str, float]) -> list[GateResult]:
        checks = [
            ("compression", self.compression, thresholds["compression"], lambda v, t: v >= t),
            ("generality", float(self.generality), thresholds["generality"], lambda v, t: v >= t),
            ("reuse", float(self.reuse), thresholds["reuse"], lambda v, t: v >= t),
            ("predictive", self.predictive, thresholds["predictive"], lambda v, t: v >= t),
            ("efficiency", self.efficiency, thresholds["efficiency"], lambda v, t: v >= t),
            (
                "interpretability",
                float(self.n),
                thresholds["interpretability"],
                lambda v, t: v <= t,  # SHORT enough to read — an upper bound
            ),
            ("reversibility", self.reversibility, thresholds["reversibility"], lambda v, t: v > t),
        ]
        return [GateResult(name, value, thr, test(value, thr)) for name, value, thr, test in checks]

    def _breadth_passed(self, results: dict[str, GateResult]) -> bool:
        return results["generality"].passed or results["reuse"].passed

    def promoted(self, thresholds: dict[str, float]) -> bool:
        results = {g.name: g for g in self.gates(thresholds)}
        return all(results[name].passed for name in _MANDATORY) and self._breadth_passed(results)

    def failing_gates(self, thresholds: dict[str, float]) -> list[str]:
        """Binding constraints: each failing mandatory quality gate, plus a single
        ``breadth`` entry when neither breadth path (generality/reuse) is met."""
        results = {g.name: g for g in self.gates(thresholds)}
        failing = [name for name in _MANDATORY if not results[name].passed]
        if not self._breadth_passed(results):
            failing.append("breadth")
        return failing


class PrimitiveDiscoveryEngine:
    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 4,
        min_occurrences: int = 2,
        thresholds: dict[str, float] | None = None,
        flow_fn: FlowFn = _default_flow,
        domain_fn: DomainFn | None = None,
    ) -> None:
        self.min_n = min_n
        self.max_n = max_n
        self.min_occurrences = min_occurrences
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.flow_fn = flow_fn
        self.domain_fn = domain_fn or (lambda _e: None)

    def _ngrams(self, flow: Sequence[str]) -> list[tuple[str, ...]]:
        grams: list[tuple[str, ...]] = []
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(flow) - n + 1):
                grams.append(tuple(flow[i : i + n]))
        return grams

    def discover(self, experiences: list[Experience]) -> list[PrimitiveCandidate]:
        # Tally, per n-gram: adjacent occurrences and which experiences use it.
        occurrences: dict[tuple[str, ...], int] = {}
        support: dict[tuple[str, ...], set[str]] = {}
        goals: dict[tuple[str, ...], set[str]] = {}
        verified_ids: set[str] = set()
        domain_of: dict[str, str | None] = {}
        for experience in experiences:
            verified = experience.verified
            if verified:
                verified_ids.add(experience.id)
            domain_of[experience.id] = self.domain_fn(experience)
            goal = getattr(experience, "goal", experience.id)
            flow = self.flow_fn(experience)
            seen_here: set[tuple[str, ...]] = set()
            for gram in self._ngrams(flow):
                occurrences[gram] = occurrences.get(gram, 0) + 1
                if gram not in seen_here:
                    support.setdefault(gram, set()).add(experience.id)
                    goals.setdefault(gram, set()).add(goal)
                    seen_here.add(gram)

        all_ids = {e.id for e in experiences}
        candidates: list[PrimitiveCandidate] = []
        for gram, count in occurrences.items():
            if count < self.min_occurrences:
                continue
            support_ids = sorted(support.get(gram, set()))
            others = all_ids - set(support_ids)
            domains = frozenset(d for i in support_ids if (d := domain_of.get(i)) is not None)
            candidates.append(
                PrimitiveCandidate(
                    fragments=gram,
                    occurrences=count,
                    support_ids=support_ids,
                    domains=domains,
                    used_verified=sum(i in verified_ids for i in support_ids),
                    other_total=len(others),
                    other_verified=sum(i in verified_ids for i in others),
                    distinct_goals=len(goals.get(gram, set())),
                )
            )
        # Most-compressive first, so the strongest candidates lead the report.
        candidates.sort(key=lambda c: (-c.compression, c.fragments))
        return candidates

    def promote(
        self, experiences: list[Experience]
    ) -> tuple[list[PrimitiveCandidate], list[PrimitiveCandidate]]:
        promoted: list[PrimitiveCandidate] = []
        rejected: list[PrimitiveCandidate] = []
        for candidate in self.discover(experiences):
            (promoted if candidate.promoted(self.thresholds) else rejected).append(candidate)
        return promoted, rejected

    def discover_candidates(self, memory: MemoryRouter) -> list["Candidate"]:
        """CandidateSource contract (instance form). Delegates to the module
        function so both call styles share one implementation."""
        return discover_primitive_candidates(memory, thresholds=self.thresholds)


def _domain_lookup(memory: MemoryRouter) -> DomainFn:
    """Map an experience to its emergent-domain name via the domain concepts."""
    member_domain: dict[str, str] = {}
    for record in memory.concepts.search(tags=["domain"], limit=200):
        name = record.content.get("name", record.id)
        for member_id in record.content.get("members", []):
            member_domain.setdefault(member_id, name)
    return lambda experience: member_domain.get(experience.id)


def discover_primitives_and_store(
    memory: MemoryRouter, thresholds: dict[str, float] | None = None
) -> list[PrimitiveCandidate]:
    """Discover candidate primitives from stored experiences, promote only those
    that clear the quality gates and one breadth path (generality OR reuse), and
    record every decision in the ledger. Returns the promoted candidates (usually
    none — that is the point)."""
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    engine = PrimitiveDiscoveryEngine(thresholds=thresholds, domain_fn=_domain_lookup(memory))
    promoted, rejected = engine.promote(experiences)
    ledger = Ledger(memory)

    for candidate in promoted:
        memory.concepts.add(
            {
                "level": "primitive",
                "name": candidate.name(),
                "fragments": list(candidate.fragments),
                "occurrences": candidate.occurrences,
                "domains": sorted(candidate.domains),
                "gates": {g.name: g.value for g in candidate.gates(engine.thresholds)},
            },
            tags=["primitive", "discovered"],
            confidence=candidate.predictive,
            record_id=candidate.id,
        )
        for member_id in candidate.support_ids:
            memory.add_edge(candidate.id, member_id, "compiled_from")

    for candidate in [*promoted, *rejected]:
        won = candidate in promoted
        if won:
            path = (
                "generality" if candidate.generality >= engine.thresholds["generality"] else "reuse"
            )
            reason = f"all quality gates passed; breadth via {path}"
        else:
            reason = f"failed gates: {', '.join(candidate.failing_gates(engine.thresholds))}"
        ledger.record_claim(
            subject_id=candidate.id,
            hypothesis=f"the fragment sequence {list(candidate.fragments)} is a reusable primitive",
            experiment=f"six-gate promotion benchmark: {reason}",
            dataset=candidate.support_ids,
            metrics={g.name: round(g.value, 4) for g in candidate.gates(engine.thresholds)},
            decision="adopted" if won else "rejected",
            confidence=candidate.predictive,
            reproducible=True,
            claim_id=f"claim_{candidate.id}_primitive",
        )
    return promoted


def discover_primitive_candidates(
    memory: MemoryRouter, thresholds: dict[str, float] | None = None
) -> list["Candidate"]:
    """CandidateSource contract: emit an EXPERIMENT candidate per candidate
    primitive that clears the six gates, WITHOUT promoting or recording anything.

    Purely observational -- reruns discovery (no store writes) and packages the
    primitive's own *reversibility* claim as a statistical A/B: workflows that
    USE the primitive (treatment = supporters) verify strictly better than those
    that do NOT (baseline = the rest). This is exactly the counterfactual the
    reversibility gate already measures, re-expressed as a runnable experiment.

    The observed groups are unequal-sized. run_experiment computes an equal-n
    two-proportion test on one task list, so we reframe deterministically: n =
    number of supporter workflows; treatment task i carries the real supporter
    verified flag; baseline task i realizes the *other-group* empirical verify
    rate deterministically (the first round(rate*n) tasks succeed). No random
    sampling, so the experiment is reproducible. This is a documented estimator
    of the reversibility gap, not fabricated data -- see GOV-PRM-001.

    Does NOT promote, record, or mutate. The orchestrator submits; the
    ExperimentRunner produces the evidence; the promotion gate decides.
    """
    from cog.science.experiment import EvidenceClass, ExperimentSpec
    from cog.science.pipeline import Candidate

    experiences = [
        Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)
    ]
    engine = PrimitiveDiscoveryEngine(thresholds=thresholds, domain_fn=_domain_lookup(memory))
    promoted, _rejected = engine.promote(experiences)
    out: list[Candidate] = []
    for cand in promoted:
        n = len(cand.support_ids)
        if n == 0:
            continue
        # Real supporter outcomes: used_verified successes out of n.
        treat_outcomes = [i < cand.used_verified for i in range(n)]
        # Deterministic realization of the other-group verify rate at the same n.
        base_rate = cand.baseline_predictive
        base_succ = round(base_rate * n)
        base_outcomes = [i < base_succ for i in range(n)]
        tasks = list(range(n))

        def _treatment(task, _o=treat_outcomes):
            return {"verified": _o[task]}

        def _baseline(task, _o=base_outcomes):
            return {"verified": _o[task]}

        spec = ExperimentSpec(
            subject_id=cand.id,
            hypothesis=(
                f"workflows using {list(cand.fragments)} verify strictly better "
                f"than those that do not (reversibility)"
            ),
            baseline=_baseline,
            treatment=_treatment,
            tasks=tasks,
            evidence_class=EvidenceClass.STATISTICAL,
            baseline_id="without_primitive",
            treatment_id=cand.id,
        )
        out.append(
            Candidate(
                subject_id=cand.id,
                hypothesis=spec.hypothesis,
                kind="experiment",
                producer_spec=spec,
                baseline_id="without_primitive",
                treatment_id=cand.id,
            )
        )
    return out
