"""Evaluation infrastructure: common suite interface + runner.

Every suite (Correctness, Capability, future benchmarks) inherits EvaluationSuite
and returns an EvaluationReport. The runner handles timing, manifest assembly,
and artifact recording so individual suites stay small and focused.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cog.evaluation.infra.manifest import Manifest
from cog.evaluation.infra.metrics import MetricResult, MetricRegistry
from cog.evaluation.infra.report import EvaluationReport


class EvaluationSuite(ABC):
    """Common interface. Subclasses implement _run() returning metric results."""

    name: str = "suite"
    version: str = "v1.0.0"

    def __init__(self, seed: int = 42, config: dict[str, Any] | None = None,
                 artifact_root: str | Path | None = None) -> None:
        self.seed = seed
        self.config = config or {}
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.metrics = MetricRegistry()
        self._register_metrics()

    def _register_metrics(self) -> None:
        """Override to register suite-specific metrics in self.metrics."""

    @abstractmethod
    def _run(self) -> tuple[list[MetricResult], dict[str, str]]:
        """Return (metric_results, artifact_paths)."""

    def run(self) -> EvaluationReport:
        t0 = time.monotonic()
        results, artifacts = self._run()
        duration = time.monotonic() - t0
        passed = all(r.passed for r in results)
        manifest = Manifest(
            eval_suite_version=self.version,
            experiment_id=f"exp_{self.seed}",
            seed=self.seed,
            configuration=self.config,
            artifacts=artifacts,
            results={"passed": passed, "duration_seconds": round(duration, 3)},
        ).to_dict()
        return EvaluationReport(
            suite_name=self.name,
            version=self.version,
            metrics=results,
            manifest=manifest,
            artifacts=artifacts,
            duration_seconds=round(duration, 3),
            passed=passed,
        )


class SuiteRunner:
    """Runs one or more suites and formats a combined report."""

    def __init__(self, suites: list[EvaluationSuite]) -> None:
        self.suites = suites

    def run_all(self) -> list[EvaluationReport]:
        return [s.run() for s in self.suites]

    def print(self) -> bool:
        all_ok = True
        for rep in self.run_all():
            print(rep.render())
            print()
            all_ok = all_ok and rep.passed
        verdict = "PASS" if all_ok else "FAIL"
        print("=" * 56)
        print(f"  OVERALL VERDICT: {verdict}")
        print("=" * 56)
        return all_ok
