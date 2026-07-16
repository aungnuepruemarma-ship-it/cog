"""Cog evaluation framework.

Layered, versioned evaluation:
  infra/       -- harness, metrics, stats, manifest, report, baselines, generators
  learning/    -- belief-layer (epistemic) evaluation
  runtime/     -- policy-layer (behavioral) evaluation via real A/B
  correctness/ -- must-pass architecture invariants
  capability/  -- aggregates learning + runtime into system metrics
"""

from cog.evaluation.infra.harness import EvaluationSuite, SuiteRunner
from cog.evaluation.infra.manifest import Manifest

__all__ = ["EvaluationSuite", "SuiteRunner", "Manifest"]
