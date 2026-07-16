"""
execution/ordering.py -- opt-in dependency-aware step ordering.

This is the REAL runtime plug-in for the heuristic validated by
EXP-DISCOVERY-001 ("dependency-aware prioritization improves multi-step
task completion", Cohen's h=1.95, p<0.001). It is OPT-IN: the executor
only reorders when a reorder callable is supplied (set by the runtime from
an active, versioned policy). Default behavior is unchanged.

Design:
- Pure function: `dependency_aware_order(steps, deps_fn)` returns a NEW
  list of steps topologically sorted by `deps_fn(step) -> list[str]`
  (each dep is a step-id reference). Steps with no deps keep a stable
  relative order.
- Uses Kahn's algorithm. On a cycle it falls back to the original order
  (safe: never silently drops or deadlocks steps).
- No modification to PlanStep required: dependencies are supplied
  externally (policy/experiment config or a plan annotation), keeping the
  change minimal and reviewable.

Maturity: this is a CONTROLLED-ROLLOUT component, not default behavior.
It is only active when an EXPERIMENTAL/VALIDATED/ACTIVE policy enables it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from cog.execution.planner import PlanStep


@dataclass
class OrderingResult:
    steps: list[PlanStep]
    reordered: bool = False
    cycle_detected: bool = False
    notes: list[str] = field(default_factory=list)


def dependency_aware_order(
    steps: list[PlanStep],
    deps_fn: Callable[[PlanStep], list[str]],
    id_fn: Callable[[PlanStep, int], str] | None = None,
) -> OrderingResult:
    """Topologically sort `steps` so every step runs after its dependencies.

    Args:
        steps:   the planner-produced steps (unchanged input).
        deps_fn: maps a step to the list of step-ids it depends on.
        id_fn:   maps (step, index) -> a stable id used for dep references.
                 Defaults to f"s{index}" to match the executor's trace ids.

    Returns OrderingResult with the reordered steps. On a cycle, returns the
    original order with cycle_detected=True (safe fallback).
    """
    n = len(steps)
    id_of = id_fn or (lambda s, i: f"s{i}")
    ids = [id_of(s, i) for i, s in enumerate(steps)]
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    # Build dependency graph.
    indeg = {cid: 0 for cid in ids}
    adj: dict[str, list[str]] = {cid: [] for cid in ids}
    for i, s in enumerate(steps):
        for dep in deps_fn(s):
            if dep in id_to_idx and dep != ids[i]:
                adj[dep].append(ids[i])
                indeg[ids[i]] += 1

    # Kahn's algorithm, preserving original relative order among ready nodes.
    ready = deque([cid for cid in ids if indeg[cid] == 0])
    order_ids: list[str] = []
    while ready:
        cid = ready.popleft()
        order_ids.append(cid)
        for m in adj[cid]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)

    if len(order_ids) != n:
        # Cycle detected: fall back to original order (never drop steps).
        return OrderingResult(
            steps=list(steps), reordered=False, cycle_detected=True,
            notes=["cycle detected in dependency graph; used original order"],
        )

    ordered = [steps[id_to_idx[cid]] for cid in order_ids]
    reordered = order_ids != ids
    return OrderingResult(steps=ordered, reordered=reordered, cycle_detected=False,
                          notes=[] if reordered else ["already in valid order"])


# Preset mode selectors so the runtime can pick by policy without importing internals.
def order_with_mode(
    steps: list[PlanStep],
    mode: str,
    deps_fn: Callable[[PlanStep], list[str]] | None = None,
) -> OrderingResult:
    """Dispatch by ordering mode. 'planner' = no change (default)."""
    if mode in (None, "planner", "none"):
        return OrderingResult(steps=list(steps), reordered=False)
    if mode == "dependency_aware":
        if deps_fn is None:
            # No deps declared -> topological sort is a no-op (stable order).
            return OrderingResult(steps=list(steps), reordered=False,
                                  notes=["dependency_aware with no deps_fn: stable order"])
        return dependency_aware_order(steps, deps_fn)
    raise ValueError(f"unknown ordering mode: {mode!r}")
