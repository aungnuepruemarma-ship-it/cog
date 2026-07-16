"""Phase 5: the Experience — Cog's atomic unit of learning.

Goal → Workspace → Reasoning → Execution → Verification → Metrics → Outcome.
Failures are recorded with the same fidelity as successes; only successes
are trusted (see the verification gate).

Schema v0.1 adds the four structural pieces the deterministic evidence layer
needs *before* any policy engine exists:

  * belief_state / reality_delta  -- the "belief vs reality" gap
  * failure                       -- category / error_signature / root_cause
  * resolution                    -- what was attempted and whether it worked
  * causal_graph                  -- failure_node -> caused_by links
  * replay                        -- seed / environment_snapshot / task_version

Everything here is produced by the runtime (executor + verifier), never by an
LLM. Policy extraction depends on this being accurate first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cog._util import json_safe, new_id, utc_now
from cog.execution.executor import ExecutionResult
from cog.execution.planner import Plan
from cog.runtime.task import Task
from cog.verification.pipeline import VerificationReport
from cog.workspace.workspace import TaskWorkspace


# --------------------------------------------------------------------------- #
# Structural sub-records (schema v0.1)
# --------------------------------------------------------------------------- #
@dataclass
class BeliefState:
    """What Cog assumed to be true before acting.

    The gap between these assumptions and what actually happened is the
    single most valuable learning signal: it teaches 'my assumption was
    wrong', not merely 'the command failed'.
    """

    assumptions: list[str] = field(default_factory=list)
    # Conditions Cog believed held (e.g. "port 8080 free", "python 3.11").
    known_conditions: list[str] = field(default_factory=list)


@dataclass
class RealityDelta:
    """What the world did that violated the belief state."""

    unexpected_conditions: list[str] = field(default_factory=list)
    # Free-text description of the surprise, if any.
    note: str | None = None


@dataclass
class FailureInfo:
    """Structured classification of a failure (empty when outcome==success)."""

    category: str | None = None  # e.g. "dependency_failure"
    error_signature: str | None = None  # e.g. "missing_package"
    root_cause: str | None = None  # filled by analysis, not at record time
    failed_step: str | None = None  # step id that errored


@dataclass
class Resolution:
    """What was done about the failure, and whether it worked."""

    attempted: bool = False
    action_taken: str | None = None
    result: str | None = None  # "success" | "failure" | "partial" | None


@dataclass
class CausalGraph:
    """Minimal causal link: a failure node and what caused it.

    A normal log says 'Step 5 failed'. A cognitive record says
    'Step 5 failed because <caused_by>'.
    """

    failure_node: str | None = None
    caused_by: list[str] = field(default_factory=list)


@dataclass
class ReplayInfo:
    """Everything needed to reproduce this experience.

    Without replay, policy validation is unreliable.
    """

    seed: int | None = None
    environment_snapshot: str | None = None  # hash of the env at run time
    task_version: str = "v1"


@dataclass
class ExperienceContext:
    """Static context snapshot: environment + initial state at plan time."""

    os: str = "linux"
    runtime: str = "native"
    available_tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    initial_files: list[str] = field(default_factory=list)
    initial_services: list[str] = field(default_factory=list)


@dataclass
class ExperienceMetrics:
    """Raw material for Reasoning Economics (Phase 20)."""

    duration_s: float = 0.0
    actions: int = 0
    errors: int = 0
    budget_max_actions: int = 0
    confidence: float = 0.0
    attempt: int = 1  # which planning attempt produced this experience
    model_calls: int = 0  # 0 when the plan came from a compiled skill


# --------------------------------------------------------------------------- #
# The Experience record
# --------------------------------------------------------------------------- #
@dataclass
class Experience:
    id: str
    task_id: str
    goal: str
    purpose: str
    domain: str = "unspecified"
    difficulty: str = "unspecified"
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    context: ExperienceContext = field(default_factory=ExperienceContext)
    belief: BeliefState = field(default_factory=BeliefState)
    reality_delta: RealityDelta = field(default_factory=RealityDelta)
    workspace: dict[str, Any] = field(default_factory=dict)  # snapshot at plan time
    reasoning: dict[str, Any] = field(default_factory=dict)  # planner prompt + raw plan
    execution: list[dict[str, Any]] = field(default_factory=list)  # the full action log
    verification: dict[str, Any] = field(default_factory=dict)  # VerificationReport.to_dict()
    metrics: ExperienceMetrics = field(default_factory=ExperienceMetrics)
    failure: FailureInfo = field(default_factory=FailureInfo)
    resolution: Resolution = field(default_factory=Resolution)
    causal: CausalGraph = field(default_factory=CausalGraph)
    replay: ReplayInfo = field(default_factory=ReplayInfo)
    outcome: str = "failure"  # "success" | "failure"
    output: Any = None
    created_at: str = field(default_factory=utc_now)
    # ---- governance / attribution fields (governance-v0.2, additive) ---- #
    session_id: str | None = None      # identifies the learning session/run
    agent_id: str | None = None        # which agent produced this experience
    environment_id: str | None = None  # identifies the environment snapshot
    timestamp: str = field(default_factory=utc_now)  # record time (captured, not inferred)

    # ---- derived properties (deterministic, no LLM) ---- #
    @property
    def verified(self) -> bool:
        return bool(self.verification.get("verified"))

    @property
    def confidence(self) -> float:
        return float(self.verification.get("confidence", 0.0))

    @property
    def strategy(self) -> str:
        return str(self.reasoning.get("strategy", "model_plan"))

    @property
    def failed(self) -> bool:
        return self.outcome != "success"

    # ---- serialization ---- #
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = json_safe(data["output"])
        data["workspace"] = json_safe(data["workspace"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experience:
        payload = dict(data)
        metrics = payload.pop("metrics", {}) or {}
        context = payload.pop("context", {}) or {}
        belief = payload.pop("belief", {}) or {}
        reality_delta = payload.pop("reality_delta", {}) or {}
        failure = payload.pop("failure", {}) or {}
        resolution = payload.pop("resolution", {}) or {}
        causal = payload.pop("causal", {}) or {}
        replay = payload.pop("replay", {}) or {}
        return cls(
            metrics=ExperienceMetrics(**metrics),
            context=ExperienceContext(**context),
            belief=BeliefState(**belief),
            reality_delta=RealityDelta(**reality_delta),
            failure=FailureInfo(**failure),
            resolution=Resolution(**resolution),
            causal=CausalGraph(**causal),
            replay=ReplayInfo(**replay),
            **payload,
        )

    # ---- validation: schema + invariants, deterministic, no LLM ---- #
    def validate(self) -> list[str]:
        """Return a list of problems; empty list means the record is valid.

        This is the gate that keeps the evidence layer clean. It is pure
        deterministic checking — no model call — so it can run on every emit.
        """
        problems: list[str] = []

        # Identity / required fields
        if not self.id:
            problems.append("id is empty")
        if not self.task_id:
            problems.append("task_id is empty")
        if not self.goal:
            problems.append("goal is empty")

        # outcome is one of the two allowed values
        if self.outcome not in ("success", "failure"):
            problems.append(f"outcome must be 'success'|'failure', got {self.outcome!r}")

        # outcome must agree with the verification gate
        if self.outcome == "success" and not self.verified:
            problems.append("outcome=success but verification.verified is False")
        if self.outcome == "failure" and self.verified:
            problems.append("outcome=failure but verification.verified is True")

        # failures must carry classification; successes must not
        if self.failed:
            if self.failure.category is None:
                problems.append("failure outcome without failure.category")
            if self.failure.error_signature is None:
                problems.append("failure outcome without failure.error_signature")
        else:
            if self.failure.category is not None:
                problems.append("success outcome carries a failure.category")

        # execution trace completeness
        if not self.execution:
            problems.append("execution log is empty")

        # belief vs reality consistency
        # Phase A captures reality deltas from verification mismatches even
        # before BeliefState (Phase B) is populated, so a delta without
        # declared assumptions is allowed. We only flag a contradiction when
        # beliefs WERE declared but the gap references none of them.
        if self.reality_delta.unexpected_conditions and self.belief.assumptions:
            # Beliefs exist; the delta should still be coherent (no hard
            # invariant to enforce yet — kept as a soft, non-blocking note).
            pass

        # replay reproducibility: a failure we want to learn from needs a snapshot
        if self.failed and self.replay.environment_snapshot is None:
            problems.append(
                "failure experience without replay.environment_snapshot "
                "(cannot be reproduced for shadow validation)"
            )

        # confidence bounds
        if not (0.0 <= self.confidence <= 1.0):
            problems.append(f"confidence out of [0,1]: {self.confidence}")

        return problems

    def is_valid(self) -> bool:
        return not self.validate()

    # ---- builder from the live runtime trace (deterministic) ---- #
    @classmethod
    def from_run(
        cls,
        task: Task,
        workspace: TaskWorkspace,
        plan: Plan,
        execution: ExecutionResult,
        report: VerificationReport,
        strategy: str = "model_plan",
        attempt: int = 1,
        *,
        domain: str = "unspecified",
        difficulty: str = "unspecified",
        belief: BeliefState | None = None,
        reality_delta: RealityDelta | None = None,
        failure: FailureInfo | None = None,
        resolution: Resolution | None = None,
        causal: CausalGraph | None = None,
        replay: ReplayInfo | None = None,
    ) -> Experience:
        """Construct an Experience purely from runtime objects.

        No LLM is involved. The optional structured fields (belief, failure,
        causal, replay, ...) default to empty unless the caller — e.g. the
        executor or verifier — populates them from observed facts.
        """
        errors = execution.log.errors
        failed_step = errors[0].tool if errors else None
        metrics = ExperienceMetrics(
            duration_s=round(execution.duration_s, 6),
            actions=len(execution.log),
            errors=len(errors),
            budget_max_actions=task.budget.max_actions,
            confidence=report.confidence,
            attempt=attempt,
            model_calls=1 if strategy == "model_plan" else 0,
        )
        return cls(
            id=new_id("exp"),
            task_id=task.id,
            goal=task.goal,
            purpose=task.purpose,
            domain=domain,
            difficulty=difficulty,
            constraints=list(task.constraints),
            success_criteria=list(task.success_criteria),
            context=ExperienceContext(
                available_tools=list(getattr(workspace, "available_tools", [])),
                constraints=list(task.constraints),
            ),
            belief=belief or BeliefState(),
            reality_delta=reality_delta or RealityDelta(),
            workspace=workspace.persist_snapshot(),
            reasoning={
                "prompt": plan.prompt,
                "raw_plan": plan.raw,
                "rejected": plan.rejected,
                "strategy": strategy,
            },
            execution=execution.log.to_dicts(),
            verification=report.to_dict(),
            metrics=metrics,
            failure=failure
            or FailureInfo(
                category=None if report.verified else "unverified_outcome",
                error_signature=errors[0].error if errors else None,
                failed_step=failed_step,
            ),
            resolution=resolution or Resolution(),
            causal=causal or CausalGraph(),
            replay=replay or ReplayInfo(),
            outcome="success" if report.verified else "failure",
            output=json_safe(execution.output),
            created_at=utc_now(),
        )
