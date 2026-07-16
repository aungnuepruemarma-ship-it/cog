"""Phase 3: pluggable checks.

Each check inspects the task + workspace + execution and returns a scored
result. ``required`` checks can veto verification; others only shape the
confidence score (via ``weight``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cog.execution.executor import ExecutionResult
from cog.runtime.task import Task
from cog.workspace.workspace import TaskWorkspace


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float  # in [0, 1]
    details: str = ""


@runtime_checkable
class Check(Protocol):
    name: str
    required: bool
    weight: float

    def run(
        self, task: Task, workspace: TaskWorkspace, execution: ExecutionResult
    ) -> CheckResult: ...


class NoErrorsCheck:
    """Execution finished every step without an error."""

    name = "no_errors"
    required = True
    weight = 1.0

    def run(self, task: Task, workspace: TaskWorkspace, execution: ExecutionResult) -> CheckResult:
        total = len(execution.log)
        if total == 0:
            return CheckResult(self.name, passed=False, score=0.0, details="no actions executed")
        clean = total - len(execution.log.errors)
        score = clean / total if execution.completed else min(clean / total, 0.5)
        details = "; ".join(r.error for r in execution.log.errors) or "all actions succeeded"
        return CheckResult(self.name, passed=execution.completed, score=score, details=details)


class OutputCheck:
    """Final output matches the task's declared expectation, if any."""

    name = "output"
    required = True
    weight = 1.0

    def run(self, task: Task, workspace: TaskWorkspace, execution: ExecutionResult) -> CheckResult:
        expected = task.expected_output
        if expected is None:
            return CheckResult(self.name, passed=True, score=1.0, details="no expectation declared")
        output = execution.output
        if callable(expected):
            ok = bool(expected(output))
            detail = f"predicate({output!r}) -> {ok}"
        else:
            ok = output == expected or str(output) == str(expected)
            detail = f"expected {expected!r}, got {output!r}"
        return CheckResult(self.name, passed=ok, score=1.0 if ok else 0.0, details=detail)


class BudgetCheck:
    """Scores budget discipline; never vetoes (Phase 20 signal)."""

    name = "budget"
    required = False
    weight = 0.5

    def run(self, task: Task, workspace: TaskWorkspace, execution: ExecutionResult) -> CheckResult:
        if execution.truncated_by_budget:
            return CheckResult(self.name, passed=True, score=0.0, details="budget exhausted")
        used = len(execution.log) / max(task.budget.max_actions, 1)
        score = 1.0 - 0.5 * min(used, 1.0)  # cheap solutions score higher
        return CheckResult(
            self.name, passed=True, score=score, details=f"{len(execution.log)} actions used"
        )


def default_checks() -> list[Check]:
    return [NoErrorsCheck(), OutputCheck(), BudgetCheck()]
