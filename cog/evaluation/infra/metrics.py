"""Evaluation infrastructure: metric registry.

A small, explicit metric framework. Correctness metrics are threshold/bool
gates; capability metrics are scored numbers. Every suite returns the same
report shape, so CI/dashboards stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MetricResult:
    name: str
    value: float | bool | None
    passed: bool
    detail: str = ""
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "detail": self.detail,
            "target": self.target,
        }


class Metric:
    def compute(self, value: float) -> MetricResult:
        raise NotImplementedError


class ThresholdMetric(Metric):
    """Pass if value is within [min, max] (either bound optional)."""

    def __init__(self, name: str, min_value: float | None = None,
                 max_value: float | None = None, target: str = "") -> None:
        self.name = name
        self.min = min_value
        self.max = max_value
        self.target = target or _fmt_bounds(min_value, max_value)

    def compute(self, value: float) -> MetricResult:
        ok = True
        if self.min is not None:
            ok = ok and value >= self.min
        if self.max is not None:
            ok = ok and value <= self.max
        return MetricResult(self.name, value, bool(ok),
                             detail=f"value={value:g} target={self.target}",
                             target=self.target)


class ContinuousMetric(Metric):
    """Scored metric (no pass/fail); records value + optional target for display."""

    def __init__(self, name: str, target: str = "", higher_is_better: bool = True) -> None:
        self.name = name
        self.target = target
        self.higher_is_better = higher_is_better

    def compute(self, value: float) -> MetricResult:
        return MetricResult(self.name, value, True,
                             detail=f"value={value:g} target={self.target}",
                             target=self.target)


class BooleanMetric(Metric):
    def __init__(self, name: str, target: str = "must be True") -> None:
        self.name = name
        self.target = target

    def compute(self, value: bool) -> MetricResult:
        return MetricResult(self.name, value, bool(value),
                             detail=f"value={value}", target=self.target)


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        self._metrics[metric.name] = metric

    def compute(self, name: str, value: float | bool) -> MetricResult:
        m = self._metrics[name]
        return m.compute(value)  # type: ignore[arg-type]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics


def _fmt_bounds(lo: float | None, hi: float | None) -> str:
    if lo is not None and hi is not None:
        return f"[{lo:g}, {hi:g}]"
    if lo is not None:
        return f">= {lo:g}"
    if hi is not None:
        return f"<= {hi:g}"
    return "any"
