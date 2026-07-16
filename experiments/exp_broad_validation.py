"""
exp_broad_validation -- Broader validation campaign for the dependency-aware
ordering heuristic (policy_dep_aware_prioritization, currently VALIDATED, NOT
ACTIVE).

This is the Kaggle-ready, self-contained edition. It imports the ``cog``
package directly (no ~/Alive path hacks) and writes its results to
``/kaggle/working/results.json`` when run on Kaggle, falling back to
``./results.json`` locally.

Purpose (per project governance): gather the broader evidence required before
promoting VALIDATED -> ACTIVE. Does NOT change default behavior; the policy
stays in its current rollout. The script PRINTS a GO/NO-GO for ACTIVE but does
NOT auto-promote.

Campaign design:
  - Task mix across 7 domains:
      dependency-heavy    (reverse-listed dep plans)        ~40%
      dependency-free     (sequential, no deps)             ~30%
      partially-ordered   (some deps, some not)             ~20%
      noisy/real-world     (arg noise + malformed lines)    ~10%
  - Scale: --n (default 300). Split across domains.
  - Seeds: --seeds (default 3). Each seed re-runs both cohorts; aggregated.
  - Soak: --soak N runs one long single-seed battery with the policy enabled.
  - Rollback verification: re-run the fixed 70-task probe with
    ordering_mode='planner' and assert baseline completion restores cleanly.

Outputs: per-seed + aggregated completion per domain and per task-type,
bootstrap CI (2000 resamples, seed 1234) on the overall delta, no-regression
boolean, and (when a writable data dir exists) a policy_experiments row. The
script writes a JSON results file for Kaggle download.

Run locally:   python experiments/exp_broad_validation.py --n 300 --seeds 3
Run on Kaggle: bash scripts/kaggle_run.sh   (writes /kaggle/working/results.json)
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path
from statistics import mean

# --- self-contained import: only the repo root is needed on sys.path ---
_HERE = Path(__file__).resolve().parent          # .../cog/experiments
_ROOT = _HERE.parent                              # .../cog   (repo root)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cog._util import new_id, utc_now
from cog import CogRuntime, ScriptedAdapter, Task, Budget
from cog.execution.planner import Planner
from cog.learning.policy.store import PolicyStore

# Governance data lives next to the package so the experiment is portable.
GOV_ROOT = _ROOT / "cog" / "governance_data"
POLICY_ID = "policy_dep_aware_prioritization"
DOMAINS = ["programming", "mathematics", "data_analysis", "cybersecurity",
           "system_optimization", "planning", "research_synthesis"]


# ---- task generators (each returns (plan_text, deps_by_step)) ----
def _gen_dep_heavy(domain, rng):
    steps = [
        {"tool": "text", "args": {"op": "length", "value": f"{domain}-2"}, "deps": ["s1"]},
        {"tool": "text", "args": {"op": "upper", "value": f"{domain}-1"}, "deps": ["s2"]},
        {"tool": "text", "args": {"op": "reverse", "value": f"{domain}-0"}, "deps": []},
    ]
    lines = [f"step: {s['tool']} {json.dumps(s['args'])}{' deps: '+','.join(s['deps']) if s['deps'] else ''} -- {s['tool']}"
             for s in steps]
    plan = "\n".join(lines)
    parsed, _ = Planner(ScriptedAdapter(), None).parse(plan)
    return plan, {s.id: list(s.deps) for s in parsed}


def _gen_dep_free(domain, rng):
    steps = [
        {"tool": "text", "args": {"op": "reverse", "value": f"{domain}-0"}, "deps": []},
        {"tool": "text", "args": {"op": "upper", "value": f"{domain}-1"}, "deps": []},
        {"tool": "text", "args": {"op": "length", "value": f"{domain}-2"}, "deps": []},
    ]
    lines = [f"step: {s['tool']} {json.dumps(s['args'])} -- {s['tool']}" for s in steps]
    plan = "\n".join(lines)
    parsed, _ = Planner(ScriptedAdapter(), None).parse(plan)
    return plan, {s.id: list(s.deps) for s in parsed}


def _gen_partial(domain, rng):
    steps = [
        {"tool": "text", "args": {"op": "reverse", "value": f"{domain}-0"}, "deps": []},
        {"tool": "text", "args": {"op": "upper", "value": f"{domain}-1"}, "deps": ["s0"]},
        {"tool": "text", "args": {"op": "length", "value": f"{domain}-2"}, "deps": []},
    ]
    lines = [f"step: {s['tool']} {json.dumps(s['args'])}{' deps: '+','.join(s['deps']) if s['deps'] else ''} -- {s['tool']}"
             for s in steps]
    plan = "\n".join(lines)
    parsed, _ = Planner(ScriptedAdapter(), None).parse(plan)
    return plan, {s.id: list(s.deps) for s in parsed}


def _gen_noisy(domain, rng):
    # valid 2-step plan + one malformed line that must land in 'rejected' (not crash)
    steps = [
        {"tool": "text", "args": {"op": "upper", "value": f"{domain}-0"}, "deps": []},
        {"tool": "text", "args": {"op": "length", "value": f"{domain}-1"}, "deps": []},
    ]
    lines = [f"step: {s['tool']} {json.dumps(s['args'])} -- {s['tool']}" for s in steps]
    noise = rng.choice(["step: not_a_tool {\"bad\":1}", "step: text oops", "garbage line"])
    plan = "\n".join(lines + [noise])
    parsed, _ = Planner(ScriptedAdapter(), None).parse(plan)
    # noisy plans: completion = all PARSED steps ran in valid order (deps empty here)
    return plan, {s.id: list(s.deps) for s in parsed}


GENERATORS = {
    "dep_heavy": _gen_dep_heavy,
    "dep_free": _gen_dep_free,
    "partial": _gen_partial,
    "noisy": _gen_noisy,
}


def _build_tasks(n, seed):
    rng = random.Random(seed)
    per_domain = max(1, n // len(DOMAINS))
    mix = (["dep_heavy"] * 4 + ["dep_free"] * 3 + ["partial"] * 2 + ["noisy"] * 1)
    tasks = []
    for dom in DOMAINS:
        for i in range(per_domain):
            kind = rng.choice(mix)
            plan, deps = GENERATORS[kind](dom, rng)
            tasks.append((Task(goal=f"{dom}-{kind}-{i}", domain=dom, expected_output=True,
                               budget=Budget(max_actions=12),
                               context={"plan": plan, "deps_by_step": deps}),
                         kind))
    return tasks


def _completed(exp, deps_by_step):
    ex = getattr(exp, "execution", None)
    if not isinstance(ex, list) or len(ex) != len(deps_by_step):
        return False
    ids = [e.get("step_id") for e in ex]
    pos = {sid: k for k, sid in enumerate(ids)}
    for sid in ids:
        for dep in deps_by_step.get(sid, []):
            if pos.get(dep, 1e9) >= pos[sid]:
                return False
    return True


def _run_cohort(tasks, ordering_mode):
    tmp = Path(tempfile.mkdtemp(prefix=f"bv-{ordering_mode}-"))
    rt = CogRuntime(ScriptedAdapter(), storage_dir=tmp, verification_threshold=0.7,
                    ordering_mode=ordering_mode)
    out = []
    for t, kind in tasks:
        rt.adapter.script = {t.goal: t.context["plan"]}
        exp = rt.run(t)
        out.append((t.domain, kind, _completed(exp, t.context["deps_by_step"])))
    return out


def _bootstrap_delta(base_rates, exp_rates, seed=1234, n=2000):
    """Bootstrap CI for the mean delta (exp - base) across all task results."""
    rng = random.Random(seed)
    deltas = [e - b for b, e in zip(base_rates, exp_rates)]
    if not deltas:
        return (0.0, 0.0, 0.0)
    obs = mean(deltas)
    bs = []
    for _ in range(n):
        sample = [rng.choice(deltas) for _ in deltas]
        bs.append(mean(sample))
    bs.sort()
    lo = bs[int(0.025 * len(bs))]
    hi = bs[int(0.975 * len(bs))]
    return (obs, lo, hi)


def _campaign(n, seeds):
    all_base, all_exp = [], []
    per_domain_base, per_domain_exp = {d: [] for d in DOMAINS}, {d: [] for d in DOMAINS}
    per_kind_base, per_kind_exp = {k: [] for k in GENERATORS}, {k: [] for k in GENERATORS}
    for seed in seeds:
        tasks = _build_tasks(n, seed)
        base = _run_cohort(tasks, "planner")
        exp = _run_cohort(tasks, "dependency_aware")
        for (db, kb, cb), (_, ke, ce) in zip(base, exp):
            all_base.append(1 if cb else 0); all_exp.append(1 if ce else 0)
            per_domain_base[db].append(1 if cb else 0); per_domain_exp[db].append(1 if ce else 0)
            per_kind_base[kb].append(1 if cb else 0); per_kind_exp[ke].append(1 if ce else 0)
    return all_base, all_exp, per_domain_base, per_domain_exp, per_kind_base, per_kind_exp


def _rollback_probe():
    """Re-run the fixed 70-task reverse-listed battery with ordering_mode='planner'
    and assert baseline completion is its known value (proves archived default restores)."""
    tasks = []
    for dom in DOMAINS:
        for i in range(10):
            plan, deps = _gen_dep_heavy(dom, random.Random(i))
            tasks.append((Task(goal=f"rb-{dom}-{i}", domain=dom, expected_output=True,
                               budget=Budget(max_actions=12),
                               context={"plan": plan, "deps_by_step": deps}), "dep_heavy"))
    res = _run_cohort(tasks, "planner")
    rate = mean(1 if c else 0 for _, _, c in res)
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--soak", type=int, default=0, help="long single-seed run with policy enabled")
    args = ap.parse_args()

    print(f"[BROAD-VAL] n={args.n} seeds={args.seeds}")
    all_base, all_exp, pdb_, pde_, pkb_, pke_ = _campaign(args.n, args.seeds)

    overall_base = mean(all_base); overall_exp = mean(all_exp)
    obs, lo, hi = _bootstrap_delta(all_base, all_exp)
    print(f"\n=== AGGREGATE (n={len(all_base)}) ===")
    print(f"  baseline : {overall_base:.3f}")
    print(f"  exp      : {overall_exp:.3f}")
    print(f"  delta    : {obs:+.3f}  CI95[{lo:+.3f}, {hi:+.3f}]")

    print(f"\n  {'domain':18} {'base':>7} {'exp':>7} {'delta':>7}")
    for d in DOMAINS:
        b = mean(pdb_[d]) if pdb_[d] else 0; e = mean(pde_[d]) if pde_[d] else 0
        print(f"  {d:18} {b:7.3f} {e:7.3f} {e-b:+7.3f}")
    print(f"  {'task-type':18} {'base':>7} {'exp':>7} {'delta':>7}")
    for k in GENERATORS:
        b = mean(pkb_[k]) if pkb_[k] else 0; e = mean(pke_[k]) if pke_[k] else 0
        print(f"  {k:18} {b:7.3f} {e:7.3f} {e-b:+7.3f}")

    # Regression: any domain or task-type dropping below baseline - 0.10
    regressions = []
    for d in DOMAINS:
        b = mean(pdb_[d]) if pdb_[d] else 0; e = mean(pde_[d]) if pde_[d] else 0
        if e < b - 0.10: regressions.append(f"domain:{d} ({e-b:+.3f})")
    for k in GENERATORS:
        b = mean(pkb_[k]) if pkb_[k] else 0; e = mean(pke_[k]) if pke_[k] else 0
        if e < b - 0.10: regressions.append(f"kind:{k} ({e-b:+.3f})")
    no_regression = not regressions
    print(f"\n  no-regression: {no_regression}")
    if regressions: print(f"  REGRESSIONS: {regressions}")

    # Rollback verification
    rb = _rollback_probe()
    rollback_ok = abs(rb - 0.0) < 0.001  # planner default must restore known baseline
    print(f"  rollback probe (planner default): {rb:.3f}  clean={rollback_ok}")

    # Soak (optional)
    soak_ok = None
    if args.soak:
        soak_tasks = _build_tasks(args.soak, 999)
        soak = _run_cohort(soak_tasks, "dependency_aware")
        soak_ok = mean(1 if c else 0 for _, _, c in soak)
        print(f"  soak ({args.soak} tasks, policy enabled): {soak_ok:.3f}")

    # GO/NO-GO for ACTIVE (does NOT auto-promote)
    go = no_regression and rollback_ok and (overall_exp >= overall_base)
    print(f"\n  ACTIVE GO/NO-GO: {'GO' if go else 'NO-GO'} (not auto-promoted)")

    # Build the results payload
    results = {
        "policy_id": POLICY_ID,
        "n": len(all_base),
        "seeds": list(args.seeds),
        "overall_base": round(overall_base, 4),
        "overall_exp": round(overall_exp, 4),
        "delta": round(obs, 4),
        "delta_ci95": [round(lo, 4), round(hi, 4)],
        "per_domain": {
            d: {
                "base": round(mean(pdb_[d]), 4) if pdb_[d] else 0.0,
                "exp": round(mean(pde_[d]), 4) if pde_[d] else 0.0,
            } for d in DOMAINS
        },
        "per_kind": {
            k: {
                "base": round(mean(pkb_[k]), 4) if pkb_[k] else 0.0,
                "exp": round(mean(pke_[k]), 4) if pke_[k] else 0.0,
            } for k in GENERATORS
        },
        "no_regression": no_regression,
        "regressions": regressions,
        "rollback_clean": rollback_ok,
        "soak_rate": round(soak_ok, 4) if soak_ok is not None else None,
        "active_go": bool(go),
        "note": "Broader campaign; ACTIVE withheld pending review.",
    }

    # Persist to governance store (best-effort; writable dir on Kaggle/local)
    try:
        GOV_ROOT.mkdir(parents=True, exist_ok=True)
        pstore = PolicyStore(GOV_ROOT)
        pstore.record_experiment(POLICY_ID, "broad_validation", results)
        print(f"  persisted: governance row for {POLICY_ID}")
    except Exception as exc:  # pragma: no cover - storage is optional on Kaggle
        print(f"  [warn] governance persist skipped: {exc}")

    # Write results file: /kaggle/working/results.json if present, else ./results.json
    out_path = Path("/kaggle/working/results.json")
    if not out_path.parent.exists():
        out_path = _ROOT / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  results written to: {out_path}")


if __name__ == "__main__":
    main()
