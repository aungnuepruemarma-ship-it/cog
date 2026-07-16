"""Phase 3: the verification pipeline.

Execution → checks → confidence score. ``verified`` is True only when every
required check passed AND the weighted confidence clears the threshold.
Nothing enters long-term memory without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cog.execution.executor import ExecutionResult
from cog.runtime.task import Task
from cog.verification.checks import Check, CheckResult, default_checks
from cog.workspace.workspace import TaskWorkspace


@dataclass
class VerificationReport:
    results: list[CheckResult]
    confidence: float
    verified: bool
    threshold: float
    required_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "verified": self.verified,
            "threshold": self.threshold,
            "required_failures": self.required_failures,
            "results": [asdict(r) for r in self.results],
        }


class VerificationPipeline:
    def __init__(self, checks: list[Check] | None = None, threshold: float = 0.7) -> None:
        self.checks = checks if checks is not None else default_checks()
        self.threshold = threshold

    def verify(
        self, task: Task, workspace: TaskWorkspace, execution: ExecutionResult
    ) -> VerificationReport:
        results: list[CheckResult] = []
        required_failures: list[str] = []
        weight_total = 0.0
        weighted_score = 0.0

        for check in self.checks:
            result = check.run(task, workspace, execution)
            results.append(result)
            weight_total += check.weight
            weighted_score += check.weight * max(0.0, min(1.0, result.score))
            if check.required and not result.passed:
                required_failures.append(check.name)

        confidence = weighted_score / weight_total if weight_total else 0.0
        verified = not required_failures and confidence >= self.threshold
        return VerificationReport(
            results=results,
            confidence=round(confidence, 4),
            verified=verified,
            threshold=self.threshold,
            required_failures=required_failures,
        )
