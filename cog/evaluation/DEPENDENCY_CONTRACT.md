# Evaluation Layer — Dependency Contract

This contract lists every external `cog.*` API the evaluation layer
(`cog/evaluation/`) imports, and the required symbol/behavior. It is the
interface the evaluation commit must build against once the underlying engine
layers are committed.

## Import surface (exact, from grep of cog/evaluation)

### Tracked in HEAD (no action needed)
- `cog._util`
- `cog.experience.store` (ExperienceStore: add/get/filter/replay/transitions)
- `cog.experiment.runtime_ab` (run_runtime_ab)
- `cog.learning.belief.{contradiction,engine,lifecycle,model,store,synthesis}`
- `cog.learning.policy.{lifecycle,model,runtime,selector,store}`
- `cog.learning.stats` (proportion_ci, wald_ci, cohens_h, bootstrap_ci)
- `cog.evaluation.*` (internal — infra/learning/runtime/capability/correctness/epistemic_suite)

### REQUIRES UNCOMMITTED ENGINE-CHANGES (BLOCKING — see COORD_EVAL_COMMIT.md)
- `cog.experience.record`:
  - `Experience` dataclass, must accept `failure: FailureInfo` kwarg
  - `FailureInfo` dataclass with `.category` and `.error_signature` fields
  - `Experience.validate() -> list[str]` (returns problems; store.add refuses on non-empty)
- `cog.runtime.core`:
  - `CogRuntime.run(task, policy_context=None)` — `policy_context` param is
    working-tree only (HEAD: `run(self, task)`)
- `cog.runtime.task`:
  - `Task` (working-tree 5-line addition)

NOTE: `cog.experiment.ab` is NOT a blocking dependency. The eval layer imports
`proportion_ci` from `cog.learning.stats` (tracked in HEAD) and `run_runtime_ab`
from `cog.experiment.runtime_ab` (tracked). `compare_proportions` lives inside
`ab.py` but is NOT imported by the eval layer, so `ab.py` need not be committed
for the evaluation commit to build. (This corrects an earlier draft that listed
`ab.py` as blocking.)

## Behavioral contracts the eval layer assumes (must hold post-commit)
1. `ExperienceStore.add(exp)` raises if `exp.validate()` is non-empty.
2. `run_runtime_ab(tasks, policy_context)` returns `(ExperimentReport, cost)`;
   control = `policy_context=None`, treatment = active policies.
3. `BeliefEngine.run(min_evidence=...)` returns `list[BeliefCase]` with
   `.belief.state` and `.belief.claim.condition`.
4. `generate_dataset(seed=...)` is deterministic (same seed -> byte-identical).
5. `PolicyContext` built from active `Policy` objects; `Policy.effect` is
   `PolicyEffect(metric, direction, expected_delta)`.

## Versioning
- Evaluation suite version: v1.0.0 (capability suite reports this)
- Manifest version: 1 (cog/evaluation/infra/manifest.py)
- Dataset version: v1 (single-domain synthetic; v2/v3 tiers are separate)
