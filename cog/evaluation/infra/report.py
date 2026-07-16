"""Evaluation infrastructure: common report structure.

Every suite (correctness, capability, future ones) returns the SAME
EvaluationReport shape. That common interface is what keeps CI, dashboards,
and future benchmark runners simple.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cog.evaluation.infra.manifest import Manifest
from cog.evaluation.infra.metrics import MetricResult


@dataclass
class EvaluationReport:
    suite_name: str
    version: str
    metrics: list[MetricResult] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "version": self.version,
            "metrics": [m.to_dict() for m in self.metrics],
            "manifest": self.manifest,
            "artifacts": self.artifacts,
            "duration_seconds": self.duration_seconds,
            "passed": self.passed,
        }

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # ---- text rendering ---- #
    def render(self) -> str:
        lines = []
        lines.append(f"{self.suite_name} ({self.version})")
        lines.append("-" * 56)
        for m in self.metrics:
            mark = "OK " if m.passed else "FAIL"
            val = m.value
            if isinstance(val, float):
                vs = f"{val:.4g}"
            else:
                vs = str(val)
            lines.append(f"  [{mark}] {m.name:28s} {vs:>10s}  {m.target}")
        lines.append("-" * 56)
        lines.append(f"  Passed: {self.passed}")
        return "\n".join(lines)
