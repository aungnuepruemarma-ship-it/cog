"""Epistemic validation suite: the constitution test for Cog's learning.

Run:  python -m cog.evaluation.epistemic_suite.harness
"""

from cog.evaluation.epistemic_suite.harness import EpistemicTestHarness, main
from cog.evaluation.epistemic_suite.report import TestResult, evaluate_gate

__all__ = ["EpistemicTestHarness", "main", "TestResult", "evaluate_gate"]
