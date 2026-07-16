"""Evaluation infrastructure: baselines.

Only ONE baseline for v0.1, as agreed: the no-learning runtime. It is the SAME
CogRuntime the learned system uses, differing only in that no policy is injected
(policy_context=None). That isolates the value added by learning. No artificial
"learning_enabled=False" flag is needed -- v0.1 changes runtime behavior purely
through policy injection, so "no policy" IS "no learning effect".
"""

from __future__ import annotations

from pathlib import Path

from cog.experiment.runtime_ab import _solve_with_policy  # same path the learned arm uses
from cog.runtime.task import Task


class NoLearningRuntime:
    """Baseline: identical runtime, no learned policy injected."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage = Path(storage_dir)

    def solve(self, task: Task) -> bool:
        return _solve_with_policy(task, self._storage, policy_context=None)
