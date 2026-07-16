"""Benchmark harness: the seed of Reasoning Economics (Phase 20) and the
Experiment Engine's comparison baseline (Phase 17).

    python -m cog.bench

Runs a fixed task suite through the runtime with a deterministic scripted
adapter and reports success rate, verified rate, mean confidence, mean
latency, and actions per task. Deliberately includes failure cases — the
verification gate is part of what's being measured.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cog import Budget, CallableAdapter, CogRuntime, ScriptedAdapter, Task


@dataclass
class BenchCase:
    name: str
    goal: str
    plan_script: str  # what the scripted "model" answers for this goal
    expected_output: Any = None
    expect_verified: bool = True  # what the gate SHOULD decide
    budget: Budget = field(default_factory=Budget)


@dataclass
class BenchRow:
    case: str
    verified: bool
    gate_correct: bool
    confidence: float
    latency_s: float
    actions: int
    outcome: str


@dataclass
class BenchReport:
    rows: list[BenchRow]

    @property
    def gate_accuracy(self) -> float:
        return sum(r.gate_correct for r in self.rows) / len(self.rows)

    @property
    def verified_rate(self) -> float:
        return sum(r.verified for r in self.rows) / len(self.rows)

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(r.confidence for r in self.rows)

    @property
    def mean_latency_s(self) -> float:
        return statistics.fmean(r.latency_s for r in self.rows)

    @property
    def mean_actions(self) -> float:
        return statistics.fmean(r.actions for r in self.rows)

    def summary(self) -> dict[str, float]:
        return {
            "n": len(self.rows),
            "cases": len(self.rows),
            "gate_accuracy": round(self.gate_accuracy, 4),
            "verified_rate": round(self.verified_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_latency_s": round(self.mean_latency_s, 6),
            "mean_actions": round(self.mean_actions, 2),
        }

    @property
    def deterministic_summary(self) -> dict[str, float]:
        """Metrics that must be byte-identical across runs given a fixed seed.

        ``mean_latency_s`` is deliberately EXCLUDED: it is wall-clock timing
        noise, not a property of the logic under test. Comparing runs on the
        full ``summary()`` would report false non-determinism. Use this for the
        reproducibility guarantee.
        """
        return {
            "n": len(self.rows),
            "cases": len(self.rows),
            "gate_accuracy": round(self.gate_accuracy, 4),
            "verified_rate": round(self.verified_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_actions": round(self.mean_actions, 2),
        }


def _calc(expression: str, description: str = "compute") -> str:
    return f"step: calculator {json.dumps({'expression': expression})} -- {description}"


def default_suite() -> list[BenchCase]:
    """Solvable arithmetic, multi-step chains, and deliberate failure modes."""
    return [
        BenchCase(
            name="arithmetic_simple",
            goal="Compute 2 + 2 * 10",
            plan_script=_calc("2 + 2 * 10"),
            expected_output=22,
        ),
        BenchCase(
            name="arithmetic_precedence",
            goal="Compute (7 - 2) ** 3 / 5",
            plan_script=_calc("(7 - 2) ** 3 / 5"),
            expected_output=25,
        ),
        BenchCase(
            name="multi_step_chain",
            goal="Note the plan then compute 12 * 12",
            plan_script=(
                'step: note {"text": "about to square 12"} -- record intent\n'
                + _calc("12 * 12", "square it")
            ),
            expected_output=144,
        ),
        BenchCase(
            name="compound_interest",
            goal="Compute compound growth 1000 * (1 + 0.05) ** 10",
            plan_script=_calc("1000 * (1 + 0.05) ** 10"),
            expected_output=lambda out: abs(float(out) - 1628.894626777442) < 1e-6,
        ),
        BenchCase(
            name="negative_and_mod",
            goal="Compute -17 % 5",
            plan_script=_calc("-17 % 5"),
            expected_output=3,
        ),
        BenchCase(
            name="text_reverse",
            goal="Reverse the word cog",
            plan_script='step: text {"op": "reverse", "value": "cog"} -- reverse it',
            expected_output="goc",
        ),
        BenchCase(
            name="text_length",
            goal="Measure the length of runtime",
            plan_script='step: text {"op": "length", "value": "runtime"} -- count chars',
            expected_output=7,
        ),
        BenchCase(
            name="json_get",
            goal="Read the port from the config document",
            plan_script=(
                'step: json {"op": "get", "document": "{\\"server\\": {\\"port\\": 8080}}",'
                ' "path": "server.port"} -- extract port'
            ),
            expected_output=8080,
        ),
        BenchCase(
            name="json_keys",
            goal="List the top-level keys of the payload",
            plan_script=(
                'step: json {"op": "keys", "document": "{\\"b\\": 1, \\"a\\": 2}"} -- list keys'
            ),
            expected_output=["a", "b"],
        ),
        BenchCase(
            name="wrong_answer_rejected",
            goal="Compute 6 * 7 but the model plans the wrong expression",
            plan_script=_calc("6 + 7"),
            expected_output=42,
            expect_verified=False,
        ),
        BenchCase(
            name="unknown_tool_rejected",
            goal="Use a tool that does not exist",
            plan_script='step: teleport {"to": "prod"} -- not a real tool',
            expect_verified=False,
        ),
        BenchCase(
            name="unsafe_expression_rejected",
            goal="Try to evaluate malicious code",
            plan_script=_calc("__import__('os').system('true')"),
            expect_verified=False,
        ),
        BenchCase(
            name="empty_plan_rejected",
            goal="Produce a goal the model has no plan for",
            plan_script="I would rather write prose than steps.",
            expect_verified=False,
        ),
        BenchCase(
            name="budget_truncation",
            goal="Plan more steps than the budget allows",
            plan_script="\n".join(_calc(f"{i} + {i}") for i in range(6)),
            budget=Budget(max_actions=3),
            expect_verified=False,
        ),
    ]


def run_bench(
    suite: list[BenchCase] | None = None,
    runtime_factory: Callable[[ScriptedAdapter, Path], CogRuntime] | None = None,
) -> BenchReport:
    suite = suite if suite is not None else default_suite()
    rows: list[BenchRow] = []
    with tempfile.TemporaryDirectory(prefix="cog-bench-") as tmp:
        for i, case in enumerate(suite):
            adapter = ScriptedAdapter(script={case.goal: case.plan_script})
            storage = Path(tmp) / f"case_{i}"
            runtime = (
                runtime_factory(adapter, storage)
                if runtime_factory
                else CogRuntime(adapter, storage_dir=storage)
            )
            task = Task(goal=case.goal, expected_output=case.expected_output, budget=case.budget)
            started = time.monotonic()
            experience = runtime.run(task)
            latency = time.monotonic() - started
            rows.append(
                BenchRow(
                    case=case.name,
                    verified=experience.verified,
                    gate_correct=experience.verified == case.expect_verified,
                    confidence=experience.confidence,
                    latency_s=latency,
                    actions=experience.metrics.actions,
                    outcome=experience.outcome,
                )
            )
            runtime.close()
    return BenchReport(rows=rows)


@dataclass
class MemoryReuseProbe:
    """Does a verified experience improve the NEXT run of a similar goal?

    The core philosophy made measurable: after one verified run, the second
    run's workspace should surface the earlier fact/experience as memories.
    """

    first_run_memories: int
    second_run_memories: int

    @property
    def learning_observed(self) -> bool:
        return self.first_run_memories == 0 and self.second_run_memories > 0


def run_memory_reuse_probe() -> MemoryReuseProbe:
    goal = "Compute 2 + 2 * 10"
    adapter = ScriptedAdapter(script={goal: _calc("2 + 2 * 10")})
    counts: list[int] = []
    with tempfile.TemporaryDirectory(prefix="cog-probe-") as tmp:
        runtime = CogRuntime(adapter, storage_dir=Path(tmp))
        runtime.hooks.on("plan_ready", lambda _e, p: counts.append(len(p["workspace"].memories)))
        runtime.run(Task(goal=goal, expected_output=22))
        runtime.run(Task(goal=goal, expected_output=22))
        runtime.close()
    return MemoryReuseProbe(first_run_memories=counts[0], second_run_memories=counts[1])


@dataclass
class SkillReuseProbe:
    """After two verified model-planned runs + one learning cycle, a NEW goal
    matching the induced template must be solved by skill replay: verified,
    zero model calls. Compression made measurable."""

    skills_compiled: int
    strategy: str
    replay_verified: bool
    model_calls_on_replay: int

    @property
    def reuse_observed(self) -> bool:
        return (
            self.skills_compiled >= 1
            and self.strategy == "skill_replay"
            and self.replay_verified
            and self.model_calls_on_replay == 0
        )


def run_skill_reuse_probe() -> SkillReuseProbe:
    # Triggers anchor on the prompt's "Goal:" line — retrieved memories quote
    # earlier goals in their JSON, and must not re-trigger old plans.
    adapter = ScriptedAdapter(
        script={
            "Goal: Compute 3 + 4": _calc("3 + 4"),
            "Goal: Compute 8 * 9": _calc("8 * 9"),
            # deliberately NO script for the third goal: if the model were
            # consulted it would return an empty plan and fail verification
        }
    )
    with tempfile.TemporaryDirectory(prefix="cog-skill-probe-") as tmp:
        runtime = CogRuntime(adapter, storage_dir=Path(tmp))
        runtime.run(Task(goal="Compute 3 + 4", expected_output=7))
        runtime.run(Task(goal="Compute 8 * 9", expected_output=72))
        report = runtime.learn()
        calls_before = len(adapter.calls)
        experience = runtime.run(Task(goal="Compute 12 - 5", expected_output=7))
        runtime.close()
    return SkillReuseProbe(
        skills_compiled=report.skills_compiled,
        strategy=experience.strategy,
        replay_verified=experience.verified,
        model_calls_on_replay=len(adapter.calls) - calls_before,
    )


@dataclass
class RecoveryProbe:
    """A wrong first plan + one retry: the failure hypothesis must reach the
    replan and the second attempt must verify. Reflection made measurable."""

    verified: bool
    attempts: int

    @property
    def recovery_observed(self) -> bool:
        return self.verified and self.attempts == 2


def run_recovery_probe() -> RecoveryProbe:
    goal = "Compute 6 * 7"
    adapter = ScriptedAdapter(
        script={
            f"Goal: {goal}": _calc("6 + 7"),  # wrong plan on the first attempt
            "Previous attempt failed": _calc("6 * 7"),  # corrected under the hypothesis
        }
    )
    with tempfile.TemporaryDirectory(prefix="cog-recovery-probe-") as tmp:
        runtime = CogRuntime(adapter, storage_dir=Path(tmp))
        experience = runtime.run(Task(goal=goal, expected_output=42, budget=Budget(max_retries=1)))
        runtime.close()
    return RecoveryProbe(verified=experience.verified, attempts=experience.metrics.attempt)


@dataclass
class TransferProbe:
    """Cross-domain transfer at Cog scale: train on the calculator domain,
    then act in the text domain. Reasoning (reflection, genes) must
    transfer; surface skills must NOT falsely fire across domains."""

    no_false_transfer: bool
    recovery_in_new_domain: bool
    genes_span_domains: bool
    domains_discovered: int

    @property
    def transfer_observed(self) -> bool:
        return self.no_false_transfer and self.recovery_in_new_domain and self.genes_span_domains


def run_transfer_probe() -> TransferProbe:
    goal = "Reverse the word laminar"
    adapter = ScriptedAdapter(
        script={
            "Goal: Compute 3 + 4": _calc("3 + 4"),
            "Goal: Compute 8 * 9": _calc("8 * 9"),
            # wrong op first; the corrected plan only appears under the
            # hypothesis (trigger kept longer than the goal trigger so the
            # longest-match rule selects it on the retry prompt)
            f"Goal: {goal}": 'step: text {"op": "upper", "value": "laminar"} -- wrong op',
            "Previous attempt failed: checks ['output'] did not pass": (
                'step: text {"op": "reverse", "value": "laminar"} -- corrected'
            ),
        }
    )
    with tempfile.TemporaryDirectory(prefix="cog-transfer-probe-") as tmp:
        runtime = CogRuntime(adapter, storage_dir=Path(tmp))
        # Train exclusively in the calculator domain, then compile skills.
        runtime.run(Task(goal="Compute 3 + 4", expected_output=7))
        runtime.run(Task(goal="Compute 8 * 9", expected_output=72))
        runtime.learn()

        # Act in the never-seen text domain, with one retry available.
        experience = runtime.run(
            Task(goal=goal, expected_output="ranimal", budget=Budget(max_retries=1))
        )
        no_false_transfer = experience.strategy == "model_plan"  # calc skill stayed put
        recovery = experience.verified and experience.metrics.attempt == 2

        report = runtime.learn()  # gene records now see both domains
        gene = runtime.memory.concepts.get("gene_execute")
        gene_domains = set(gene.content.get("transfer_domains", [])) if gene else set()
        runtime.close()
    return TransferProbe(
        no_false_transfer=no_false_transfer,
        recovery_in_new_domain=recovery,
        genes_span_domains={"calculator", "text"} <= gene_domains,
        domains_discovered=report.domains_discovered,
    )


@dataclass
class CorrectionProbe:
    """Causal credit assignment: after learning from failure→success retries,
    a NEW similar goal is solved on the FIRST attempt (the correction is
    injected preemptively), whereas without corrections it needs the retry."""

    baseline_attempts: int
    corrected_attempts: int

    @property
    def correction_observed(self) -> bool:
        return self.baseline_attempts == 2 and self.corrected_attempts == 1


def _reverse_planner(prompt: str) -> str:
    goal_line = next(line for line in prompt.splitlines() if line.startswith("Goal:"))
    goal = goal_line[len("Goal:") :].strip()
    corrected = "prefer the text" in prompt or "Previous attempt failed" in prompt
    if "reverse" in goal.lower():
        word = goal.split()[-1].strip(".")
        if corrected:
            return f'step: text {{"op": "reverse", "value": "{word}"}} -- reverse'
        return 'step: calculator {"expression": "1 + 1"} -- wrong approach'
    return ""


def run_correction_probe() -> CorrectionProbe:
    new = Task(goal="Kindly reverse bravo", expected_output="ovarb", budget=Budget(max_retries=1))
    with tempfile.TemporaryDirectory(prefix="cog-corr-probe-") as tmp:
        # Baseline: a fresh runtime with no learned corrections pays the retry.
        base = CogRuntime(CallableAdapter(_reverse_planner), storage_dir=Path(tmp) / "base")
        baseline_attempts = base.run(new).metrics.attempt
        base.close()

        # Trained: learn from two failure→success 'reverse' chains, then retry.
        trained = CogRuntime(CallableAdapter(_reverse_planner), storage_dir=Path(tmp) / "trained")
        for goal, expected in [("Reverse the word cog", "goc"), ("Please reverse tango", "ognat")]:
            trained.run(Task(goal=goal, expected_output=expected, budget=Budget(max_retries=1)))
        trained.learn()
        corrected_attempts = trained.run(new).metrics.attempt
        trained.close()
    return CorrectionProbe(
        baseline_attempts=baseline_attempts, corrected_attempts=corrected_attempts
    )


@dataclass
class CalibrationProbe:
    """Metacognition: the self-audit must tell calibrated confidence from
    miscalibrated. A matched set reads as reliable; an overconfident set
    (0.9 stated, 0.5 observed) is flagged and the map corrects it toward 0.5."""

    calibrated_reliable: bool
    miscalibrated_flagged: bool
    corrected_toward_truth: bool

    @property
    def audit_works(self) -> bool:
        return (
            self.calibrated_reliable and self.miscalibrated_flagged and self.corrected_toward_truth
        )


def run_calibration_probe() -> CalibrationProbe:
    from cog.learning.calibration import CalibrationEngine

    engine = CalibrationEngine()
    good = []
    for conf, rate in [(0.05, 0.0), (0.5, 0.5), (0.95, 1.0)]:
        good += [(conf, i < rate * 10) for i in range(10)]
    good_report = engine.evaluate(good)

    bad = [(0.9, i % 2 == 0) for i in range(20)]  # stated 0.9, observed 0.5
    bad_report = engine.evaluate(bad)

    return CalibrationProbe(
        calibrated_reliable=good_report.reliable,
        miscalibrated_flagged=not bad_report.reliable,
        corrected_toward_truth=abs(bad_report.calibrated(0.9) - 0.5) < 0.05,
    )


@dataclass
class CuriosityProbe:
    """Intrinsic motivation: given a compiled-but-uncertain skill, Cog proposes
    its own experiment, runs it, and reduces the uncertainty — no external task."""

    proposed: bool
    strategy: str
    uncertainty_reduced: bool

    @property
    def curiosity_observed(self) -> bool:
        return self.proposed and self.strategy == "skill_replay" and self.uncertainty_reduced


def _calc_planner(prompt: str) -> str:
    goal_line = next(line for line in prompt.splitlines() if line.startswith("Goal:"))
    goal = goal_line[len("Goal:") :].strip()
    if goal.startswith("Compute "):
        return f'step: calculator {{"expression": "{goal[len("Compute ") :]}"}} -- compute'
    return ""


def run_curiosity_probe() -> CuriosityProbe:
    with tempfile.TemporaryDirectory(prefix="cog-curio-probe-") as tmp:
        runtime = CogRuntime(CallableAdapter(_calc_planner), storage_dir=Path(tmp))
        runtime.run(Task(goal="Compute 3 + 4", expected_output=7))
        runtime.run(Task(goal="Compute 8 * 9", expected_output=72))
        runtime.learn()
        (skill,) = runtime.memory.skills.search(limit=5)
        runtime.memory.skills.add(
            skill.content, tags=skill.tags, confidence=0.8, record_id=skill.id
        )
        outcome = runtime.explore()
        runtime.close()
    if outcome is None:
        return CuriosityProbe(proposed=False, strategy="", uncertainty_reduced=False)
    return CuriosityProbe(
        proposed=True,
        strategy=outcome.experience.strategy,
        uncertainty_reduced=outcome.uncertainty_reduced,
    )


@dataclass
class BeliefProbe:
    """Belief revision: after a verified answer, a drifted fact contradicts it.
    Cog must detect the self-contradiction, resolve it by weight of evidence,
    and stop retrieving the loser as trustworthy context."""

    detected: bool
    resolved: int
    winner_output: object
    superseded_removed_from_retrieval: bool

    @property
    def revision_observed(self) -> bool:
        return (
            self.detected
            and self.resolved == 1
            and self.winner_output == 7
            and self.superseded_removed_from_retrieval
        )


def run_belief_probe() -> BeliefProbe:
    with tempfile.TemporaryDirectory(prefix="cog-belief-probe-") as tmp:
        runtime = CogRuntime(CallableAdapter(_calc_planner), storage_dir=Path(tmp))
        # Two verified confirmations of the true answer (support = 2)...
        runtime.run(Task(goal="Compute 3 + 4", expected_output=7))
        runtime.run(Task(goal="Compute 3 + 4", expected_output=7))
        # ...then a drifted fact asserting a wrong answer, more confident but
        # supported only once — exactly what the gate cannot veto on its own.
        runtime.memory.facts.add(
            {"statement": "drift", "goal": "Compute 3 + 4", "output": 99},
            tags=["derived"],
            confidence=0.99,
        )
        detected = len(runtime.beliefs()) == 1
        report = runtime.learn()
        retrieved = runtime.memory.retrieve("Compute 3 + 4", kinds=("fact",), limit=10)
        outputs = {r.content["output"] for r in retrieved}
        runtime.close()
    return BeliefProbe(
        detected=detected,
        resolved=report.contradictions_resolved,
        winner_output=7 if outputs == {7} else next(iter(outputs), None),
        superseded_removed_from_retrieval=(outputs == {7}),
    )


@dataclass
class CorroborationProbe:
    """Verifying without ground truth: two independent methods (a replayed skill
    and a fresh model plan) either agree — corroborating the answer above any
    single method — or disagree, catching a silently drifted skill that the gate,
    with no declared expectation, could not have caught on its own."""

    agree_corroborated: bool
    agree_confidence: float
    agree_max_prior: float
    drift_caught: bool

    @property
    def corroboration_observed(self) -> bool:
        return (
            self.agree_corroborated
            and self.agree_confidence > self.agree_max_prior
            and self.drift_caught
        )


def run_corroboration_probe() -> CorroborationProbe:
    skill = {
        "name": "replay_calculator",
        "goal_template": "Compute {p0}",
        "goal_regex": r"Compute\ (.+?)",
        "parameters": ["p0"],
        "steps": [{"tool": "calculator", "args": {"expression": "{p0}"}}],
        "uses": 0,
    }
    with tempfile.TemporaryDirectory(prefix="cog-corrob-ok-") as tmp:
        runtime = CogRuntime(CallableAdapter(_calc_planner), storage_dir=Path(tmp))
        runtime.memory.skills.add(skill, tags=["compiled"], confidence=0.9)
        agree = runtime.corroborate(Task(goal="Compute 5 + 6"))  # no expected_output
        runtime.close()

    drifted = dict(skill, steps=[{"tool": "calculator", "args": {"expression": "{p0} + 100"}}])
    with tempfile.TemporaryDirectory(prefix="cog-corrob-drift-") as tmp:
        runtime = CogRuntime(CallableAdapter(_calc_planner), storage_dir=Path(tmp))
        runtime.memory.skills.add(drifted, tags=["compiled"], confidence=0.9)
        drift = runtime.corroborate(Task(goal="Compute 2 + 2"))
        runtime.close()

    return CorroborationProbe(
        agree_corroborated=agree.corroborated,
        agree_confidence=agree.confidence,
        agree_max_prior=max(agree.priors, default=1.0),
        drift_caught=(not drift.corroborated and len(drift.methods) == 2),
    )


@dataclass
class CompetitionProbe:
    """Representation competition: several theories explain the same real logs;
    the field is scored on prediction and parsimony, one survives, and every
    competitor (winner + losers) is filed in the Theory Ledger."""

    field_size: int
    survived: bool
    winner_is_parsimonious: bool
    ledger_adopted: int
    ledger_rejected: int

    @property
    def competition_observed(self) -> bool:
        return (
            self.field_size >= 2
            and self.survived
            and self.winner_is_parsimonious
            and self.ledger_adopted == 1
            and self.ledger_rejected == self.field_size - 1
        )


def run_competition_probe() -> CompetitionProbe:
    from cog.learning.representation_competition import run_competition_and_store
    from cog.science.ledger import Ledger

    with tempfile.TemporaryDirectory(prefix="cog-compete-probe-") as tmp:
        runtime = CogRuntime(CallableAdapter(_calc_planner), storage_dir=Path(tmp))
        for i in range(1, 9):
            runtime.run(Task(goal=f"Compute {i} + {i}", expected_output=2 * i))
        competitions = run_competition_and_store(runtime.memory)
        ledger = Ledger(runtime.memory)
        adopted = len(ledger.claims.search(tags=["adopted"], limit=99))
        rejected = len(ledger.claims.search(tags=["rejected"], limit=99))
        competition = competitions[0] if competitions else None
        survivor = competition.survivor if competition else None
        runtime.close()

    field_size = len(competition.theories) if competition else 0
    # the calculator cluster's redundant features make the shortest discriminating
    # theory (flow alone) the parsimonious winner.
    parsimonious = survivor is not None and all(
        len(survivor.definition) <= len(t.definition) for t in competition.theories
    )
    return CompetitionProbe(
        field_size=field_size,
        survived=survivor is not None,
        winner_is_parsimonious=parsimonious,
        ledger_adopted=adopted,
        ledger_rejected=rejected,
    )


@dataclass
class PrimitiveProbe:
    """Primitive discovery, two breadth paths. A fragment that recurs across two
    domains promotes via *generality*; the same fragment confined to one domain
    but reused across many distinct goals promotes via *reuse*; confined to one
    domain AND few goals it is rejected on *breadth*. Strict in every direction —
    it earns a promotion two ways, and still withholds one."""

    generality_path_promoted: bool
    reuse_path_promoted: bool
    reuse_winner_generality: int
    breadth_rejected: bool
    breadth_failing_gate: str

    @property
    def discovery_observed(self) -> bool:
        return (
            self.generality_path_promoted
            and self.reuse_path_promoted
            and self.reuse_winner_generality == 1  # one domain — carried purely by reuse
            and self.breadth_rejected
            and self.breadth_failing_gate == "breadth"
        )


@dataclass
class OrganizationProbe:
    """The Organizational Runtime: on a task family where one reasoning structure
    verifies more often than another, comparison ranks the better one first, and
    evolution adopts an improved organization from a proven primitive — but only
    because the verified-rate strictly rises."""

    comparison_picks_better: bool
    evolution_accepted: bool
    incumbent_rate: float
    evolved_rate: float
    weak_proposal_rejected: bool

    @property
    def organization_observed(self) -> bool:
        return (
            self.comparison_picks_better
            and self.evolution_accepted
            and self.evolved_rate > self.incumbent_rate
            and self.weak_proposal_rejected
        )


def run_organization_probe() -> OrganizationProbe:
    from cog.learning.organizations import (
        Organization,
        OrganizationComparator,
        OrganizationEvolver,
    )

    def exp(flow, verified):
        return SimpleNamespace(execution=[{"tool": t} for t in flow], verified=verified)

    # A task family: 'direct' (calculator) verified 2/5; 'record-then-compute'
    # (note, calculator) verified 5/5 — the structure matters.
    family = [exp(["calculator"], i < 2) for i in range(5)]
    family += [exp(["note", "calculator"], True) for _ in range(5)]

    direct = Organization("direct", ("calculator",))
    record = Organization("record-then-compute", ("note", "calculator"))

    comparator = OrganizationComparator()
    better = comparator.compare(direct, record, family)

    # evolution: from the incumbent 'direct' plus a proven primitive, adopt only
    # a strict verified-rate improvement.
    good = OrganizationEvolver().evolve(direct, family, primitives=[("note", "calculator")])
    weak = OrganizationEvolver().evolve(record, family, primitives=[("calculator",)])

    return OrganizationProbe(
        comparison_picks_better=better == record,
        evolution_accepted=good.accepted,
        incumbent_rate=good.incumbent_rate,
        evolved_rate=good.candidate_rate,
        weak_proposal_rejected=not weak.accepted,
    )


def run_primitive_probe() -> PrimitiveProbe:
    from cog.learning.primitives import DEFAULT_THRESHOLDS, PrimitiveDiscoveryEngine

    def corpus(second_domain: str, goals: list[str]):
        members = [
            SimpleNamespace(
                id=f"c{i}", execution=[], verified=True, _f=["observe", "gather"], goal=goals[i]
            )
            for i in range(4)
        ] + [
            SimpleNamespace(
                id=f"t{i}", execution=[], verified=True, _f=["observe", "gather"], goal=goals[4 + i]
            )
            for i in range(4)
        ]
        domains = {f"c{i}": "calc" for i in range(4)}
        domains.update({f"t{i}": second_domain for i in range(4)})
        domains.update({f"n{i}": "calc" for i in range(4)})
        noise = [
            SimpleNamespace(id=f"n{i}", execution=[], verified=i < 2, _f=["solo"], goal=f"n{i}")
            for i in range(4)
        ]
        return members + noise, domains

    def engine(domains):
        return PrimitiveDiscoveryEngine(
            flow_fn=lambda e: e._f, domain_fn=lambda e: domains.get(e.id)
        )

    def find(corpus_domains, want_promoted: bool):
        corp, doms = corpus_domains
        promoted, rejected = engine(doms).promote(corp)
        pool = promoted if want_promoted else rejected
        return next((c for c in pool if c.fragments == ("observe", "gather")), None)

    distinct = [f"g{i}" for i in range(8)]  # 8 distinct goals
    few = ["a", "b"] * 4  # only 2 distinct goals

    generality_win = find(corpus("text", distinct), want_promoted=True)  # 2 domains
    reuse_win = find(corpus("calc", distinct), want_promoted=True)  # 1 domain, reuse 8
    breadth_loss = find(corpus("calc", few), want_promoted=False)  # 1 domain, reuse 2

    return PrimitiveProbe(
        generality_path_promoted=generality_win is not None and generality_win.generality >= 2,
        reuse_path_promoted=reuse_win is not None and reuse_win.reuse >= 5,
        reuse_winner_generality=reuse_win.generality if reuse_win else -1,
        breadth_rejected=breadth_loss is not None,
        breadth_failing_gate=(
            breadth_loss.failing_gates(DEFAULT_THRESHOLDS)[0]
            if breadth_loss and breadth_loss.failing_gates(DEFAULT_THRESHOLDS)
            else ""
        ),
    )


def main() -> None:
    report = run_bench()
    width = max(len(r.case) for r in report.rows)
    print(f"{'case':<{width}}  verified  gate_ok  confidence  latency_s  actions")
    for r in report.rows:
        print(
            f"{r.case:<{width}}  {str(r.verified):<8}  {str(r.gate_correct):<7}"
            f"  {r.confidence:>10.3f}  {r.latency_s:>9.4f}  {r.actions:>7}"
        )
    memory_probe = run_memory_reuse_probe()
    print(
        f"\nmemory reuse probe: run1 saw {memory_probe.first_run_memories} memories,"
        f" run2 saw {memory_probe.second_run_memories}"
        f" -> learning_observed={memory_probe.learning_observed}"
    )
    skill_probe = run_skill_reuse_probe()
    print(
        f"skill reuse probe: {skill_probe.skills_compiled} skill(s) compiled,"
        f" new goal solved via {skill_probe.strategy} with"
        f" {skill_probe.model_calls_on_replay} model calls"
        f" -> reuse_observed={skill_probe.reuse_observed}"
    )
    recovery_probe = run_recovery_probe()
    print(
        f"recovery probe: verified={recovery_probe.verified} in"
        f" {recovery_probe.attempts} attempts"
        f" -> recovery_observed={recovery_probe.recovery_observed}"
    )
    transfer_probe = run_transfer_probe()
    print(
        f"transfer probe: no_false_transfer={transfer_probe.no_false_transfer},"
        f" recovery_in_new_domain={transfer_probe.recovery_in_new_domain},"
        f" genes_span_domains={transfer_probe.genes_span_domains},"
        f" domains={transfer_probe.domains_discovered}"
        f" -> transfer_observed={transfer_probe.transfer_observed}"
    )
    correction_probe = run_correction_probe()
    print(
        f"correction probe: baseline needed {correction_probe.baseline_attempts} attempts,"
        f" after learning {correction_probe.corrected_attempts}"
        f" -> correction_observed={correction_probe.correction_observed}"
    )
    calibration_probe = run_calibration_probe()
    print(
        f"calibration probe: calibrated_reliable={calibration_probe.calibrated_reliable},"
        f" miscalibrated_flagged={calibration_probe.miscalibrated_flagged},"
        f" corrected={calibration_probe.corrected_toward_truth}"
        f" -> audit_works={calibration_probe.audit_works}"
    )
    curiosity_probe = run_curiosity_probe()
    print(
        f"curiosity probe: proposed={curiosity_probe.proposed} via"
        f" {curiosity_probe.strategy}, uncertainty_reduced={curiosity_probe.uncertainty_reduced}"
        f" -> curiosity_observed={curiosity_probe.curiosity_observed}"
    )
    belief_probe = run_belief_probe()
    print(
        f"belief probe: detected={belief_probe.detected},"
        f" resolved={belief_probe.resolved},"
        f" winner={belief_probe.winner_output!r},"
        f" superseded_hidden={belief_probe.superseded_removed_from_retrieval}"
        f" -> revision_observed={belief_probe.revision_observed}"
    )
    corroboration_probe = run_corroboration_probe()
    print(
        f"corroboration probe: agree_corroborated={corroboration_probe.agree_corroborated}"
        f" (conf {corroboration_probe.agree_confidence} >"
        f" prior {corroboration_probe.agree_max_prior}),"
        f" drift_caught={corroboration_probe.drift_caught}"
        f" -> corroboration_observed={corroboration_probe.corroboration_observed}"
    )
    competition_probe = run_competition_probe()
    print(
        f"competition probe: {competition_probe.field_size} theories competed,"
        f" survived={competition_probe.survived} (parsimonious"
        f" ={competition_probe.winner_is_parsimonious}),"
        f" ledger adopted/rejected={competition_probe.ledger_adopted}/"
        f"{competition_probe.ledger_rejected}"
        f" -> competition_observed={competition_probe.competition_observed}"
    )
    primitive_probe = run_primitive_probe()
    print(
        f"primitive probe: generality-path promoted={primitive_probe.generality_path_promoted},"
        f" reuse-path promoted={primitive_probe.reuse_path_promoted}"
        f" (generality={primitive_probe.reuse_winner_generality}),"
        f" low-breadth rejected={primitive_probe.breadth_rejected}"
        f" on '{primitive_probe.breadth_failing_gate}'"
        f" -> discovery_observed={primitive_probe.discovery_observed}"
    )
    organization_probe = run_organization_probe()
    print(
        f"organization probe: comparison picks better={organization_probe.comparison_picks_better},"
        f" evolution {organization_probe.incumbent_rate:.2f}->{organization_probe.evolved_rate:.2f}"
        f" accepted={organization_probe.evolution_accepted},"
        f" weak proposal rejected={organization_probe.weak_proposal_rejected}"
        f" -> organization_observed={organization_probe.organization_observed}"
    )
    summary = report.summary() | {
        "learning_observed": memory_probe.learning_observed,
        "skill_reuse_observed": skill_probe.reuse_observed,
        "recovery_observed": recovery_probe.recovery_observed,
        "transfer_observed": transfer_probe.transfer_observed,
        "correction_observed": correction_probe.correction_observed,
        "calibration_audit_works": calibration_probe.audit_works,
        "curiosity_observed": curiosity_probe.curiosity_observed,
        "belief_revision_observed": belief_probe.revision_observed,
        "corroboration_observed": corroboration_probe.corroboration_observed,
        "competition_observed": competition_probe.competition_observed,
        "primitive_discovery_observed": primitive_probe.discovery_observed,
        "organization_observed": organization_probe.organization_observed,
    }
    print("\nsummary:", json.dumps(summary, indent=2))
    # Determinism guarantee: this block is byte-identical across runs for a
    # fixed seed. It EXCLUDES mean_latency_s (wall-clock noise). Compare runs
    # on this, not on the full summary, to assert reproducibility.
    det = report.deterministic_summary | {
        "learning_observed": memory_probe.learning_observed,
        "skill_reuse_observed": skill_probe.reuse_observed,
        "recovery_observed": recovery_probe.recovery_observed,
        "transfer_observed": transfer_probe.transfer_observed,
        "correction_observed": correction_probe.correction_observed,
        "calibration_audit_works": calibration_probe.audit_works,
        "curiosity_observed": curiosity_probe.curiosity_observed,
        "belief_revision_observed": belief_probe.revision_observed,
        "corroboration_observed": corroboration_probe.corroboration_observed,
        "competition_observed": competition_probe.competition_observed,
        "primitive_discovery_observed": primitive_probe.discovery_observed,
        "organization_observed": organization_probe.organization_observed,
    }
    print("deterministic_summary:", json.dumps(det, indent=2))
    if report.gate_accuracy < 1.0:
        raise SystemExit("benchmark regression: verification gate made a wrong call")
    if not memory_probe.learning_observed:
        raise SystemExit("benchmark regression: verified experience did not shape the next run")
    if not skill_probe.reuse_observed:
        raise SystemExit("benchmark regression: compiled skill was not reused for a new goal")
    if not recovery_probe.recovery_observed:
        raise SystemExit("benchmark regression: retry with failure hypothesis did not recover")
    if not transfer_probe.transfer_observed:
        raise SystemExit("benchmark regression: reasoning did not transfer across domains")
    if not correction_probe.correction_observed:
        raise SystemExit("benchmark regression: learned correction did not prevent the retry")
    if not calibration_probe.audit_works:
        raise SystemExit(
            "benchmark regression: confidence self-audit failed to detect miscalibration"
        )
    if not curiosity_probe.curiosity_observed:
        raise SystemExit(
            "benchmark regression: self-directed exploration did not reduce uncertainty"
        )
    if not belief_probe.revision_observed:
        raise SystemExit(
            "benchmark regression: belief revision did not reconcile a self-contradiction"
        )
    if not corroboration_probe.corroboration_observed:
        raise SystemExit(
            "benchmark regression: corroboration did not cross-verify without ground truth"
        )
    if not competition_probe.competition_observed:
        raise SystemExit(
            "benchmark regression: representation competition did not select a surviving theory"
        )
    if not primitive_probe.discovery_observed:
        raise SystemExit(
            "benchmark regression: primitive discovery's six-gate benchmark is not strict both ways"
        )
    if not organization_probe.organization_observed:
        raise SystemExit(
            "benchmark regression: organizational comparison/evolution did not track outcomes"
        )


if __name__ == "__main__":
    main()
