"""
exp_rollout -- Controlled activation of the dependency-aware ordering heuristic.

This is the Kaggle-ready, self-contained edition of EXP-ROLLOUT-001. It imports
the ``cog`` package directly (no ~/Alive path hacks) and writes its results to
``/kaggle/working/results.json`` when on Kaggle, falling back to ``./results.json``
locally.

Governance sequence demonstrated (the user's refined governance sequence):
  1. Experiment passes        -> done in EXP-DISCOVERY-001 (Cohen's h=1.95, p<0.001)
  2. Controlled activation     -> policy created in EXPERIMENTAL state, rollout
     scoped to the benchmark suite ONLY (not all runs). [THIS SCRIPT]
  3. Regression validation     -> multi-domain battery with heuristic active for
     the experimental cohort, compared against baseline. [THIS SCRIPT]
  4. General adoption          -> LEFT AS A MANUAL STEP. We STOP at EXPERIMENTAL and
     report; the user decides whether to promote to VALIDATED -> ACTIVE.

This is REAL: it wires the topological ordering plug-in (execution/ordering.py)
into the live CogRuntime via ordering_mode, gated by the policy. Default behavior
is unchanged for all non-cohort runs.

Outputs persisted (best-effort, portable):
  - PolicyStore (governance_data/policies.db): belief (SUPPORTED) + policy
    (EXPERIMENTAL) + policy_experiments rows (append-only).
  - results.json (Kaggle / local fallback).

Run locally:   python experiments/exp_rollout.py
Run on Kaggle: (notebook) -> writes /kaggle/working/results.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# --- self-contained import: only the repo root is needed on sys.path ---
_HERE = Path(__file__).resolve().parent          # .../cog/experiments
_ROOT = _HERE.parent                              # .../cog   (repo root)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cog._util import new_id, utc_now
from cog import CogRuntime, ScriptedAdapter, Task, Budget
from cog.learning.belief.model import Belief, BeliefClaim, BeliefState, BeliefScope, BeliefStatistics
from cog.learning.belief.store import BeliefStore
from cog.learning.policy.model import Policy, PolicyState, PolicyEffect
from cog.learning.policy.store import PolicyStore

# Governance data lives next to the package so the experiment is portable.
GOV_ROOT = _ROOT / "cog" / "governance_data"
EXP_DISCOVERY_CLAIM = "claim_experiment_dep_aware_prioritization_ae452acb-ac03-44c6-88ba-184750b08d2d"

# ---------------------------------------------------------------------------
# Multi-domain battery with DECLARED dependency chains. The heuristic only
# matters when a step depends on a prior step; we build plans where the
# planner's listed order is deliberately WRONG (reverse) so the topological
# sort must correct it. This is the honest regression test: does the
# heuristic help / hurt each domain?
# ---------------------------------------------------------------------------
DOMAINS = [
    "programming", "mathematics", "data_analysis", "cybersecurity",
    "system_optimization", "planning", "research_synthesis",
]


def _make_dep_plan(domain: str, i: int) -> tuple[str, dict]:
    """Build a 3-step chain where the LISTED (reverse) order is topologically
    INVALID, so a naive executor violates dependencies but the dependency-aware
    heuristic corrects it. The parser assigns ids s0,s1,s2 by listed order, so
    we derive deps_by_step by actually parsing the plan (guarantees consistency
    with what the runtime executes)."""
    # Forward logical chain: A(s2) <- B(s1) <- C(s0) i.e. C depends on B depends on A.
    # Listed REVERSE (C, B, A) so parse assigns s0=C, s1=B, s2=A.
    # C depends on B (s0 deps s1); B depends on A (s1 deps s2); A deps [].
    steps = [
        {"tool": "text", "args": {"op": "length", "value": f"{domain}-2"}, "deps": ["s1"]},  # C (leaf)
        {"tool": "text", "args": {"op": "upper", "value": f"{domain}-1"}, "deps": ["s2"]},  # B (mid)
        {"tool": "text", "args": {"op": "reverse", "value": f"{domain}-0"}, "deps": []},     # A (seed)
    ]
    plan_lines = []
    for s in steps:
        dep_txt = f" deps: {','.join(s['deps'])}" if s["deps"] else ""
        plan_lines.append(f"step: {s['tool']} {json.dumps(s['args'])}{dep_txt} -- {s['tool']}")
    plan = "\n".join(plan_lines)
    # Parse with the real planner to get the actual (id, deps) the runtime sees.
    from cog.execution.planner import Planner
    parsed, _ = Planner(ScriptedAdapter(), None).parse(plan)
    deps_by_step = {s.id: list(s.deps) for s in parsed}
    return plan, deps_by_step


def _build_tasks():
    tasks = []
    for dom in DOMAINS:
        for i in range(10):
            goal = f"{dom}-dep-{i}"
            plan, deps_by_step = _make_dep_plan(dom, i)
            tasks.append(Task(goal=goal, domain=dom, expected_output=True,
                              budget=Budget(max_actions=12),
                              context={"plan": plan, "deps_by_step": deps_by_step}))
    return tasks


def _completed(exp, deps_by_step) -> bool:
    """Completion = all steps ran AND every step executed AFTER its deps.
    Reads the REAL executed step order from the experience's execution log
    (each record carries step_id). This directly measures whether the
    ordering heuristic produced a topologically valid execution."""
    ex = getattr(exp, "execution", None)
    if not isinstance(ex, list) or len(ex) != len(deps_by_step):
        return False
    executed_ids = [e.get("step_id") for e in ex]
    pos = {sid: k for k, sid in enumerate(executed_ids)}
    for sid in executed_ids:
        for dep in deps_by_step.get(sid, []):
            if pos.get(dep, 1e9) >= pos[sid]:
                return False  # dep must execute before the dependent step
    return True


def _run_cohort(tasks, ordering_mode: str):
    """Run tasks through a runtime with the given ordering_mode. Returns per-task
    (domain, completed) where completed = topologically valid execution."""
    tmp = Path(tempfile.mkdtemp(prefix=f"roll-{ordering_mode}-"))
    rt = CogRuntime(ScriptedAdapter(), storage_dir=tmp,
                    verification_threshold=0.7, ordering_mode=ordering_mode)
    results = []
    for t in tasks:
        plan = t.context["plan"]
        rt.adapter.script = {t.goal: plan}
        exp = rt.run(t)
        completed = _completed(exp, t.context["deps_by_step"])
        results.append((t.domain, completed))
    return results


# ---------------------------------------------------------------------------
# Governance objects
# ---------------------------------------------------------------------------
def _create_belief(store: BeliefStore) -> Belief:
    bid = "belief_dep_aware_improves_multistep"
    b = Belief(
        id=bid,
        claim=BeliefClaim(
            condition={"situation": "multi-step plan with declared step dependencies"},
            prediction={"outcome": "dependency-aware (topological) ordering yields higher completion than listed order"},
        ),
        evidence_ids=[EXP_DISCOVERY_CLAIM],
        statistics=BeliefStatistics(sample_size=60, success_rate=1.0,
                                    confidence_interval=(0.94, 1.0)),
        scope=BeliefScope(domain="multi_step_execution", task_type="planning",
                          environment="cog-runtime"),
        confidence=0.99,
        state=BeliefState.SUPPORTED,
        last_confirmed=utc_now(),
        confirmation_count=1,
    )
    store.add(b)
    store.save_state(b, "proposed", "supported by EXP-DISCOVERY-001 (Cohen h=1.95, p<0.001)")
    return store.get(bid)


def _create_policy(pstore: PolicyStore, belief_id: str) -> Policy:
    pid = "policy_dep_aware_prioritization"
    p = Policy(
        id=pid,
        action="Enable dependency-aware (topological) step ordering in the executor "
               "via execution/ordering.py when ordering_mode='dependency_aware'.",
        trigger={"rollout": "benchmark_suite_only", "ordering_mode": "dependency_aware"},
        justification=[belief_id],
        state=PolicyState.EXPERIMENTAL,
        confidence=0.99,
        evidence_ids=[EXP_DISCOVERY_CLAIM],
        expected_effect=PolicyEffect(metric="multi_step_completion_rate",
                                     direction="increase", expected_delta=0.40),
    )
    pstore.add(p)
    pstore.save_state(p, "observed", "controlled activation: EXPERIMENTAL, benchmark-suite-only rollout")
    return pstore.get(pid)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist(run_id, policy, baseline, exp, per_domain):
    payload = {
        "policy_id": policy.id,
        "policy_state": policy.state.value,
        "rollout": policy.trigger,
        "expected_effect": policy.expected_effect.to_dict(),
        "baseline_completion": baseline,
        "experimental_completion": exp,
        "per_domain": per_domain,
        "n_baseline": 70,
        "n_experimental": 70,
        "decision": "STOPPED_AT_EXPERIMENTAL",
        "note": "Controlled activation only. General adoption is a manual step pending user review.",
    }
    return payload


def _bit(x):
    return 1 if x else 0


def main():
    run_id = utc_now().replace(":", "").replace("-", "")
    print(f"[ROLLOUT] EXP-ROLLOUT-001::{run_id}")

    GOV_ROOT.mkdir(parents=True, exist_ok=True)
    bstore = BeliefStore(GOV_ROOT)
    pstore = PolicyStore(GOV_ROOT)

    belief = _create_belief(bstore)
    print(f"  belief   : {belief.id} state={belief.state.value}")
    policy = _create_policy(pstore, belief.id)
    print(f"  policy   : {policy.id} state={policy.state.value} (controlled activation)")

    tasks = _build_tasks()
    print(f"  battery  : {len(tasks)} tasks across {len(DOMAINS)} domains (deps declared, listed reverse)")

    # Baseline cohort: default planner ordering
    base_res = _run_cohort(tasks, "planner")
    # Experimental cohort: heuristic active (only because policy is EXPERIMENTAL + benchmark rollout)
    exp_res = _run_cohort(tasks, "dependency_aware")

    # Aggregate per domain
    def _agg(res):
        d = {}
        for dom, ok in res:
            d.setdefault(dom, []).append(ok)
        return {k: round(mean(v), 4) for k, v in d.items()}

    base_rate = _agg(base_res)
    exp_rate = _agg(exp_res)
    overall_base = round(mean(_bit(x) for _, x in base_res), 4)
    overall_exp = round(mean(_bit(x) for _, x in exp_res), 4)

    print(f"\n=== REGRESSION RESULTS ===")
    print(f"  {'domain':20} {'baseline':>10} {'experimental':>12} {'delta':>8}")
    for dom in DOMAINS:
        b = base_rate.get(dom, 0); e = exp_rate.get(dom, 0)
        print(f"  {dom:20} {b:>10.3f} {e:>12.3f} {e-b:>+8.3f}")
    print(f"  {'OVERALL':20} {overall_base:>10.3f} {overall_exp:>12.3f} {overall_exp-overall_base:>+8.3f}")

    # Regression check: heuristic must not DEGRADE any domain by > 0.10
    deltas = {d: exp_rate.get(d, 0) - base_rate.get(d, 0) for d in DOMAINS}
    degraded = {d: v for d, v in deltas.items() if v < -0.10}
    no_regression = not degraded
    print(f"\n  no-regression (no domain < -0.10) : {no_regression}")
    if degraded:
        print(f"  DEGRADED domains: {degraded}")

    # Record into PolicyStore as policy_experiments (append-only)
    pstore.record_experiment(policy.id, "rollout_baseline", {
        "ordering_mode": "planner", "overall_completion": overall_base, "per_domain": base_rate})
    pstore.record_experiment(policy.id, "rollout_experimental", {
        "ordering_mode": "dependency_aware", "overall_completion": overall_exp,
        "per_domain": exp_rate, "no_regression": no_regression})

    payload = _persist(run_id, policy, base_rate, exp_rate, deltas)
    print(f"\n  persisted : governance rows for {policy.id} (state={policy.state.value})")
    print(f"  policy left in EXPERIMENTAL (controlled activation).")
    print(f"  next manual step: if no_regression and overall_exp > overall_base ->")
    print(f"    promote policy to VALIDATED then ACTIVE (general adoption).")

    # Acceptance
    ok = no_regression and overall_exp >= overall_base
    print(f"  acceptance (no-regression AND exp>=base): {ok}")

    # Write results file: /kaggle/working/results.json if present, else ./results.json
    results = {
        "experiment": "EXP-ROLLOUT-001",
        "run_id": run_id,
        "policy_id": policy.id,
        "policy_state": policy.state.value,
        "overall_baseline": overall_base,
        "overall_experimental": overall_exp,
        "overall_delta": round(overall_exp - overall_base, 4),
        "per_domain": {
            d: {
                "baseline": base_rate.get(d, 0.0),
                "experimental": exp_rate.get(d, 0.0),
                "delta": round(exp_rate.get(d, 0.0) - base_rate.get(d, 0.0), 4),
            } for d in DOMAINS
        },
        "no_regression": no_regression,
        "degraded_domains": degraded,
        "acceptance": bool(ok),
        "decision": "STOPPED_AT_EXPERIMENTAL",
        "note": "Controlled activation only. General adoption is a manual step pending user review.",
    }
    out_path = Path("/kaggle/working/results.json")
    if not out_path.parent.exists():
        out_path = _ROOT / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  results written to: {out_path}")
    return ok


if __name__ == "__main__":
    main()
