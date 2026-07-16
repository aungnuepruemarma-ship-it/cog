"""Shared evaluation infrastructure: deterministic synthetic experience generators.

Produce REAL Experience records (the same schema the runtime emits) so suites
validate the actual Belief Engine, not a parallel mock. Deterministic given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cog.experience.record import (
    CausalGraph,
    Experience,
    ExperienceContext,
    ExperienceMetrics,
    FailureInfo,
    RealityDelta,
    ReplayInfo,
)
from cog._util import new_id


def make_experience(
    *, tool: str, domain: str, with_preflight: bool, failed: bool,
    category: str = "dependency_failure", seed: int = 0,
) -> Experience:
    steps = []
    if with_preflight:
        steps.append({"index": 0, "tool": "dep_preflight", "args": {},
                      "result": "ok", "error": None, "duration_s": 1.0})
    steps.append({"index": len(steps), "tool": tool, "args": {},
                  "result": None if failed else "ok",
                  "error": "missing package" if failed else None, "duration_s": 1.0})
    return Experience(
        id=new_id("exp"),
        task_id=f"t{seed}",
        goal="deploy",
        purpose="",
        domain=domain,
        difficulty="medium",
        constraints=[],
        success_criteria=[],
        context=ExperienceContext(),
        reality_delta=RealityDelta() if not failed else RealityDelta(unexpected_conditions=["pkg missing"]),
        workspace={},
        reasoning={},
        execution=steps,
        verification={"verified": not failed, "confidence": 0.0 if failed else 0.95},
        metrics=ExperienceMetrics(),
        failure=FailureInfo(category=category, error_signature=("MISSING_X" if failed else ""),
                            failed_step=tool) if failed else FailureInfo(),
        causal=CausalGraph(failure_node=tool if failed else None,
                           caused_by=[category] if failed else []),
        replay=ReplayInfo(environment_snapshot=f"sha256:{seed:064x}"),
        outcome="failure" if failed else "success",
    )


def gen_block(n: int, *, tool: str, domain: str, with_preflight: bool, failed: bool,
              category: str = "dependency_failure", start: int = 0) -> list[Experience]:
    return [make_experience(tool=tool, domain=domain, with_preflight=with_preflight,
                            failed=failed, category=category, seed=start + i)
            for i in range(n)]


# ---- hidden causal truth (NEVER stored inside Experience) ---- #
@dataclass(frozen=True)
class HiddenTruth:
    root_cause: str
    effective_interventions: frozenset[str]


@dataclass
class GeneratedDataset:
    experiences: list[Experience]
    labels: dict[str, HiddenTruth]  # keyed by experience id; empty if include_labels=False


def generate_dataset(
    *,
    n_train: int = 800,
    n_eval: int = 200,
    domain: str = "software",
    tool: str = "docker_build",
    hidden_cause: str = "missing_dependency",
    effective_interventions: frozenset[str] | None = None,
    preflight_helps: bool = True,
    seed: int = 0,
    include_labels: bool = True,
) -> GeneratedDataset:
    """Generate a train/eval dataset with hidden causal truth kept separate.

    - preflight_helps=True  -> preflight-absent cases fail; preflight-present
      succeed (a belief "no-preflight -> failure" is TRUE).
    - preflight_helps=False -> preflight does NOT help (both groups fail); a
      belief claiming preflight reduces failure is FALSE.

    The split is deterministic (first n_train for training, remainder for eval)
    so calibration is exactly reproducible. Labels are only attached when
    include_labels=True; otherwise labels is empty (false_belief_rate -> None).
    """
    eff = effective_interventions or frozenset({"dep_preflight", "install_dependencies"})
    experiences: list[Experience] = []
    labels: dict[str, HiddenTruth] = {}

    def add(with_preflight: bool, failed: bool, idx: int) -> None:
        exp = make_experience(tool=tool, domain=domain, with_preflight=with_preflight,
                              failed=failed, seed=seed + idx)
        experiences.append(exp)
        if include_labels:
            labels[exp.id] = HiddenTruth(root_cause=hidden_cause,
                                         effective_interventions=eff)

    for i in range(n_train):
        # 60% failures without preflight; 40% with preflight (success if it helps)
        if i % 5 < 3:
            add(with_preflight=False, failed=True, idx=i)
        else:
            add(with_preflight=True, failed=(not preflight_helps), idx=i)
    for j in range(n_eval):
        if j % 5 < 3:
            add(with_preflight=False, failed=True, idx=n_train + j)
        else:
            add(with_preflight=True, failed=(not preflight_helps), idx=n_train + j)

    return GeneratedDataset(experiences=experiences, labels=labels)

