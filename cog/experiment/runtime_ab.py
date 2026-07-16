"""Phase 3 -> Runtime: the real A/B learning laboratory.

This is where the cognitive loop is PROVEN, not assumed:

    Same task battery
          /            \\
    Control (no policy)   Treatment (active policy injected)
          \\            /
            Compare metrics

Both arms run through the REAL CogRuntime: planner -> executor -> verifier ->
emitter. The policy is injected exactly as in production (PolicyContext ->
PolicyAwareAdapter prepends the policy's required pre-step). If the policy is
causal, the treatment arm fails less; if it isn't, the experiment says so.

Causal dependency (real, not a benchmark artifact): the docker_build tool
answers the honest question "given this execution context, can I run?" by
checking context.metadata["preflight_done"], which dep_preflight sets. This uses
the ExecutionContext flowing through executor -> router -> tool. The same
context later supports causal graphs, replay, and safety checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cog.experiment.ab import Variant, run_experiment
from cog.execution.router import Tool
from cog.execution.tools import ToolSpec
from cog.runtime.adapter import ScriptedAdapter
from cog.runtime.core import CogRuntime
from cog.runtime.task import Task


def _demo_tools() -> list[Tool]:
    """Tools whose behavior depends on real execution context, not hidden state."""

    def dep_preflight(**kwargs: Any) -> str:
        context = kwargs.get("context")
        if context is not None:
            context.set("preflight_done", True)
        return "preflight ok"

    def docker_build(**kwargs: Any) -> str:
        context = kwargs.get("context")
        if context is None or not context.get("preflight_done", False):
            raise RuntimeError("missing package: run dependency preflight first")
        return "image built"

    return [
        ToolSpec(name="dep_preflight", description="Check dependencies before build",
                 run=dep_preflight),
        ToolSpec(name="docker_build", description="Build a docker image",
                 run=docker_build),
    ]


def _solve_with_policy(task: Task, storage_dir: Path, policy_context=None) -> bool:
    adapter = ScriptedAdapter(
        script={task.goal: "step: docker_build {} -- build image"},
        default="step: docker_build {} -- build image",
    )
    rt = CogRuntime(adapter, storage_dir=storage_dir, tools=_demo_tools(),
                    verification_threshold=0.7)
    try:
        exp = rt.run(task, policy_context=policy_context)
        return bool(exp and exp.verification.get("verified", False))
    except Exception:
        return False


def run_runtime_ab(policy_context, tasks: list[Task], storage_dir: Path, seed: int = 0):
    """Run the real A/B: control (no policy) vs treatment (policy injected)."""
    def control_solve(task: Task) -> bool:
        return _solve_with_policy(task, storage_dir, policy_context=None)

    def treatment_solve(task: Task) -> bool:
        return _solve_with_policy(task, storage_dir, policy_context=policy_context)

    return run_experiment(
        Variant(id="control", solve=control_solve),
        Variant(id="treatment", solve=treatment_solve),
        tasks,
        seed=seed,
    )
