"""Experiment-specific planners for controlled A/B comparison.

These are SEPARATE from the runtime's generic Planner (cog.execution.planner).
Baseline and treatment planners live here so the control/treatment split is
auditable and each can be frozen independently for experiments like Exp3.
"""

from cog.experiment.planners.flat import FlatPlanner

__all__ = ["FlatPlanner"]
