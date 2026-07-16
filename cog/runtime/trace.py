"""Phase 2.5: the ExecutionTrace — the runtime's factual record of one run.

The executor populates this object as it runs. It knows NOTHING about database
formats or Experience schemas; it only collects observed facts:

  * what each step tried (tool/args)
  * what it expected vs what it observed
  * which steps failed and the raw error
  * any recovery attempts

Later, the ExperienceEmitter converts a populated ExecutionTrace (+ the
VerificationReport + task/workspace/plan) into an Experience. Runtime stays
ignorant of storage; the emitter owns the schema mapping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cog._util import utc_now


@dataclass
class StepTrace:
    step_id: str
    tool: str
    args: dict[str, Any]
    # What the planner/operator believed would hold after this step.
    expected_state: dict[str, Any] = field(default_factory=dict)
    # What was actually observed (exit code, output, error).
    observed_state: dict[str, Any] = field(default_factory=dict)
    tool_output: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    # Operational assumptions in force at this step (kept factual, not LLM thoughts).
    assumptions: list[str] = field(default_factory=list)


@dataclass
class RecoveryAttempt:
    step_id: str
    action_taken: str
    result: str  # "success" | "failure" | "partial"
    detail: str = ""


@dataclass
class ExecutionTrace:
    task_id: str
    started_at: str = field(default_factory=utc_now)
    seed: int | None = None
    task_version: str = "v1"
    # Snapshot of the environment at task start (file hashes, env vars, docker state).
    environment_snapshot: str | None = None
    steps: list[StepTrace] = field(default_factory=list)
    recovery_attempts: list[RecoveryAttempt] = field(default_factory=list)
    # Set by the emitter once verification runs; mirrors VerificationReport.verified.
    notes: list[str] = field(default_factory=list)

    def add_step(self, step: StepTrace) -> None:
        self.steps.append(step)

    def add_recovery(self, attempt: RecoveryAttempt) -> None:
        self.recovery_attempts.append(attempt)

    def failed_steps(self) -> list[StepTrace]:
        return [s for s in self.steps if s.error is not None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_environment(workspace: Any, extra: dict[str, Any] | None = None) -> str:
    """Deterministic environment hash for ReplayInfo.

    Hashes a stable, JSON-serialisable view of the run environment so the
    same starting conditions produce the same hash (reproducibility).
    """
    view: dict[str, Any] = {
        "task_id": getattr(workspace, "task_id", None),
        "goal": getattr(workspace, "goal", None),
        "available_tools": getattr(workspace, "available_tools", []),
        "constraints": getattr(workspace, "constraints", []),
        "extra": extra or {},
    }
    payload = json.dumps(view, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]
