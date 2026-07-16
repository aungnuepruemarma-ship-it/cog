"""Phase 0: CogRuntime — the fast loop.

Task → Workspace → Plan (skill replay or model) → Execute → Verify →
Memory (gated) → Experience. On failed verification, the loop retries
(within ``budget.max_retries``) with the failure hypothesis appended to a
fresh workspace — reflection as evidence, not vibes. Every attempt becomes
its own Experience, linked by ``retry_of`` edges.
"""

from __future__ import annotations

from pathlib import Path

from cog.execution.executor import Executor
from cog.execution.planner import Planner
from cog.execution.router import ToolRouter
from cog.execution.strategist import PlanningStrategist
from cog.execution.tools import Tool, default_tools
from cog.experience.emitter import ExperienceEmitter
from cog.experience.graph import Edge, ExperienceGraph
from cog.experience.record import Experience
from cog.memory.router import MemoryRouter
from cog.runtime.adapter import ModelAdapter
from cog.runtime.hooks import HookBus
from cog.runtime.session import Session
from cog.runtime.task import Task
from cog.runtime.context import ExecutionContext
from cog.runtime.trace import ExecutionTrace, snapshot_environment
from cog.economics.strategy import ModelTier
from cog.verification.checks import Check
from cog.verification.pipeline import VerificationPipeline
from cog.workspace.builder import WorkspaceBuilder


class CogRuntime:
    def __init__(
        self,
        adapter: ModelAdapter,
        storage_dir: Path | str,
        tools: list[Tool] | None = None,
        checks: list[Check] | None = None,
        verification_threshold: float = 0.7,
        model_ladder: list[ModelTier] | None = None,
        ordering_mode: str = "planner",
    ) -> None:
        storage = Path(storage_dir)
        # ordering_mode selects the opt-in step-ordering heuristic. "planner"
        # (default) = no change to historical behavior. "dependency_aware"
        # enables the topological ordering plug-in (execution/ordering.py)
        # under controlled rollout (an active/experimental policy).
        self._ordering_mode = ordering_mode
        self.adapter = adapter
        self._init_tools = tools
        self._init_checks = checks
        self._verification_threshold = verification_threshold
        # Layer 4 (FrugalGPT): ordered cheapest-first model tiers. The
        # strategist picks the cheapest tier meeting the threshold; the runtime
        # plans with that tier's adapter. Falls back to ``adapter`` if no ladder.
        self.model_ladder = list(model_ladder) if model_ladder else []
        self._tiers_by_name = {t.name: t for t in self.model_ladder}
        self.session = Session(storage_dir=storage)
        self.hooks = HookBus()
        self.memory = MemoryRouter(storage)
        self.graph = ExperienceGraph(self.memory)
        self.router = ToolRouter(tools if tools is not None else default_tools(storage))
        self.planner = Planner(adapter, self.router)
        self.executor = Executor(self.router, self.hooks)
        self.pipeline = VerificationPipeline(checks, threshold=verification_threshold)
        self.workspace_builder = WorkspaceBuilder(self.memory)
        self.strategist = PlanningStrategist(
            self.planner, self.memory, required_accuracy=verification_threshold,
            ladder=self.model_ladder,
        )

    def run(self, task: Task, policy_context=None) -> Experience:
        self.session.task_ids.append(task.id)
        self.hooks.emit("task_start", {"task": task})

        # Policy injection: an active PolicyContext shapes the planner's plan
        # (hint, not authority). The executor still runs whatever the plan says.
        saved_adapter = None
        if policy_context is not None and getattr(policy_context, "policies", None):
            from cog.learning.policy.runtime import PolicyAwareAdapter
            saved_adapter = self.planner.adapter
            self.planner.adapter = PolicyAwareAdapter(saved_adapter, policy_context)
            self.hooks.emit("policy_injected", {"context": policy_context.to_dict()})

        try:
            experience = self._run_attempts(task)
        finally:
            if saved_adapter is not None:
                self.planner.adapter = saved_adapter
        return experience

    def _run_attempts(self, task: Task) -> Experience:

        # Prior experiences with overlapping goals, captured before this task
        # adds its own records — they become similar_goal edges.
        prior_similar = self.memory.experiences.search(query=task.goal, limit=3)

        max_attempts = 1 + max(0, task.budget.max_retries)
        experience: Experience | None = None
        failure_note: str | None = None
        force_model = False
        emitter = ExperienceEmitter()

        for attempt in range(1, max_attempts + 1):
            workspace = self.workspace_builder.build(task)
            if failure_note is not None:
                workspace.hypotheses.append(failure_note)

            plan, strategy, skill_record, chosen_tier = self.strategist.plan(
                workspace, force_model=force_model
            )
            # Layer 4: if a model tier was selected, plan with that tier's
            # adapter (cheapest tier meeting the verification threshold).
            if chosen_tier is not None:
                self.planner.adapter = chosen_tier.adapter
            self.hooks.emit(
                "plan_ready", {"plan": plan, "workspace": workspace, "strategy": strategy}
            )

            # Factual trace for this attempt; env snapshot enables replay.
            trace = ExecutionTrace(
                task_id=task.id,
                seed=task.seed,
                task_version=task.version,
                environment_snapshot=snapshot_environment(workspace),
            )

            # Opt-in step reordering for controlled rollout. Only active when a
            # policy enables a non-default ordering_mode. The deps convention:
            # a step may declare dependencies via args["_deps"] (list of step
            # ids like "s0"). When no deps are declared the sort is stable.
            reorder = None
            if self._ordering_mode != "planner":
                from cog.execution.ordering import order_with_mode

                def _reorder(steps):
                    return order_with_mode(
                        steps,
                        self._ordering_mode,
                        deps_fn=lambda s: list(s.deps),
                    ).steps

                reorder = _reorder

            execution = self.executor.run(
                plan, task.budget, trace=trace, workspace=workspace,
                context=ExecutionContext(task_id=task.id, workspace=workspace),
                reorder=reorder,
            )

            report = self.pipeline.verify(task, workspace, execution)
            workspace.confidence = report.confidence
            self.hooks.emit("verified", {"report": report})

            previous = experience
            experience = emitter.emit(
                task,
                workspace,
                plan,
                trace,
                execution.log,
                report,
                strategy=strategy,
                attempt=attempt,
                domain=task.domain,
                difficulty=task.difficulty,
            )

            # The gate: only verified experiences write to trusted memory.
            facts = self.memory.write_from_experience(experience)
            links = [Edge(src=experience.id, dst=f.id, kind="produced_fact") for f in facts]
            if skill_record is not None:
                links.append(Edge(src=experience.id, dst=skill_record.id, kind="used_skill"))
            links += [Edge(src=experience.id, dst=r.id, kind="similar_goal") for r in prior_similar]
            if previous is not None:
                links.append(Edge(src=experience.id, dst=previous.id, kind="retry_of"))
            self.graph.record(experience, links)

            # Evidence updates: skill evolution + strategy economics.
            if skill_record is not None:
                self.strategist.update_skill(skill_record, report.verified)
            self.strategist.record_outcome(strategy, report.verified)
            if chosen_tier is not None:
                self.strategist.record_tier_outcome(chosen_tier.name, report.verified)

            if report.verified:
                break

            error_note = (
                f"errors: {execution.log.errors[0].error}"
                if execution.log.errors
                else f"output was {execution.output!r}"
            )
            failure_note = (
                f"Previous attempt failed: checks {report.required_failures} "
                f"did not pass; {error_note}"
            )
            # A failed replay means the induced program is wrong for this
            # goal — go back to the model on the next attempt.
            force_model = strategy == "skill_replay"

        self.hooks.emit("task_end", {"experience": experience})
        self.memory.conn.commit()  # single transactional commit for the whole run
        return experience

    def learn(self):
        """Run one learning cycle (hourly-loop work) over this runtime's memory."""
        from cog.learning.loops import run_learning_cycle

        return run_learning_cycle(self.memory)

    def reliability(self):
        """Metacognitive self-audit: is this runtime's confidence calibrated?
        Returns a CalibrationReport (Brier, ECE, reliability diagram) over the
        accumulated experience history."""
        from cog.learning.calibration import evaluate_memory

        return evaluate_memory(self.memory)

    def corroborate(self, task: Task):
        """Cross-verify a goal with no ground truth by solving it two
        structurally independent ways — replaying an induced skill and asking
        the model to plan from scratch — and checking whether they agree.
        Returns a Corroboration; ``corroborated`` is True only when >=2 methods
        produced the same answer, and a disagreement is surfaced as untrusted
        rather than silently believed. Read-only: it does not write to memory."""
        from cog.verification.corroboration import Corroboration

        result = Corroboration(goal=task.goal)

        # Method 1: the induced program, if one matches — independent of the model.
        ws_skill = self.workspace_builder.build(task)
        plan, strategy, skill, _tier = self.strategist.plan(ws_skill, force_model=False)
        if strategy == "skill_replay" and skill is not None:
            execution = self.executor.run(plan, task.budget)
            if not execution.log.errors:
                result.methods.append("skill_replay")
                result.outputs.append(execution.output)
                result.priors.append(skill.confidence)

        # Method 2: a fresh model plan — independent of the induced program.
        ws_model = self.workspace_builder.build(task)
        model_plan, _strategy, _skill, _tier2 = self.strategist.plan(ws_model, force_model=True)
        execution = self.executor.run(model_plan, task.budget)
        if not execution.log.errors:
            result.methods.append("model_plan")
            result.outputs.append(execution.output)
            result.priors.append(self.strategist._model_accuracy())

        return result

    def beliefs(self):
        """Belief revision: detect goals this runtime's memory answers more than
        one way. Returns the current Contradictions (read-only detection); the
        learning cycle is what actually resolves them and supersedes the losers."""
        from cog.learning.beliefs import BeliefRevisionEngine

        return BeliefRevisionEngine().detect(self.memory)

    def explore(self):
        """Intrinsic motivation: run the single most informative experiment
        Cog can propose about itself, and fold the result back in. Picks the
        highest-uncertainty target with a synthesizable, self-verifiable probe
        (currently: uncertain calculator skills), runs it, and reports how much
        the target's confidence moved. Returns an ExplorationOutcome, or None
        when nothing runnable is uncertain enough."""
        from cog.learning.curiosity import CuriosityEngine, ExplorationOutcome

        proposals = CuriosityEngine().propose(self.memory)
        proposal = next((p for p in proposals if p.suggested_goal is not None), None)
        if proposal is None:
            return None
        before = self.memory.skills.get(proposal.target_id)
        confidence_before = before.confidence if before else 0.0
        experience = self.run(
            Task(goal=proposal.suggested_goal, expected_output=proposal.expected_output)
        )
        after = self.memory.skills.get(proposal.target_id)
        confidence_after = after.confidence if after else confidence_before
        return ExplorationOutcome(
            proposal=proposal,
            experience=experience,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
        )

    def fork(self, storage_dir: Path | str) -> CogRuntime:
        """A child runtime with a full copy of this runtime's accumulated
        memory. Experiments run on the fork without touching the parent —
        the ExperimentEngine can A/B "same history, different config"."""
        import sqlite3

        destination = Path(storage_dir)
        destination.mkdir(parents=True, exist_ok=True)
        child_conn = sqlite3.connect(destination / "memory.db")
        try:
            self.memory.conn.backup(child_conn)
        finally:
            child_conn.close()
        return CogRuntime(
            self.adapter,
            storage_dir=destination,
            tools=self._init_tools,
            checks=self._init_checks,
            verification_threshold=self._verification_threshold,
        )

    def close(self) -> None:
        self.memory.close()
