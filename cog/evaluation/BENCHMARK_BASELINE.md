# Cog Evaluation Framework — Baseline (v0.1)

This document is the committed baseline for the evaluation framework. It records
the architecture, the behavioral contracts, and the 12/12 suite result at the
time of the evaluation-only commit. It is the stable target for review / cherry-pick.

## Scope of this commit

Evaluation-only snapshot. Contains:

- `cog/evaluation/infra/`      — harness, metrics, stats, manifest, generators, baselines
- `cog/evaluation/learning/`   — belief-layer (epistemic) evaluation
- `cog/evaluation/runtime/`    — policy-layer (behavioral) evaluation
- `cog/evaluation/capability/` — aggregate (knowledge_efficiency)
- `cog/evaluation/correctness/`— architecture-invariant gates
- `cog/evaluation/versions/`   — v2_adversarial, v3_real_traces benchmark tiers
- `cog/evaluation/epistemic_suite/` — prior v0.1 epistemic battery (unchanged here)

NOT included (deliberately): `cog/runtime/**`, `cog/execution/**`,
`cog/planner/**`, `cog/learning/belief/**`, `cog/learning/policy/**`. Those are
sibling-owned and remain untracked. The evaluation tree imports from them at
runtime but does not modify them.

## Architectural principle

Beliefs and policies are SEPARATE layers with separate ground truth:

- **Beliefs** are predictive statements about the world. Evaluated against
  OBSERVATIONS only. Metrics: accuracy / precision / recall / calibration (ECE) /
  retention-under-benign-noise. `false_belief_rate` is `None` in v0.1 because the
  dataset carries no `world_state` labels — it is NOT fabricated.
- **Policies** are prescriptive interventions. Evaluated against the hidden
  `effective_interventions` labels only. Metrics: policy_lift (+ Wilson 95% CI) /
  policy_precision / policy_recall.

No benchmark reads hidden labels during belief evaluation. Train/eval leakage is
forbidden: beliefs are synthesized on the train split and evaluated only on the
held-out eval split.

## Behavioral contracts (every versioned tier)

- `generate()` is deterministic for a fixed seed (same seed => identical dataset).
- No train/eval leakage (noise injected only into the eval split).
- Hidden labels are never read during belief evaluation.
- Each tier exposes: `generate()`, `run()`, `smoke_test()`, `manifest()`.

## Metric definitions

| Metric | Meaning |
|--------|---------|
| `false_active_rate` (v2) | Active beliefs that contradict the (noise-preserving) eval pattern. Gate ≤ 0.05 |
| `replay_determinism` | Store round-trip reproduces experiences identically. Gate ≥ 0.95 |
| `scope_leakage` | Active belief with no concrete domain. Gate == 0 |
| `cross_domain_scope_leakage` (v3) | Belief spanning multiple domains. Gate == 0 |
| `multi_domain_false_active` (v3) | False belief across mixed domains. Gate ≤ 0.05 |
| `belief_retention_rate` | Fraction of active belief CONDITIONS still ACTIVE after benign noise (5% flips preserving the pattern). Stability metric. |
| `policy_lift` | treatment_success − baseline_success (real runtime A/B). |
| `policy_lift_ci` | Wilson 95% CI on the lift. |
| `knowledge_efficiency` | policy_lift / active_beliefs (UNAVAILABLE when no runtime data / no active beliefs). |

## Retention metric — interpretation note

`belief_retention_rate` measures STABILITY under benign stochastic noise, NOT
contradiction handling. Direct contradiction (pattern reversal) is a separate
property, verified by the correctness suite's `contradiction_detection` gate.
The two are deliberately kept separate: a learner should survive noise but drop
on genuine reversal.

## Known architectural question (deferred to v0.2)

The `BeliefEngine` is stateless across runs: `run()` re-synthesizes candidates
from the experience store and re-quarantines every belief. ACTIVE beliefs are not
carried forward. Whether to add persistent ACTIVE-belief memory (incremental
revalidation, aging, dependency tracking) is a dedicated v0.2 design proposal,
not bundled with evaluation work.

## Baseline result (commit time)

12/12 suites PASS:

- cog.experience.test_record_store
- cog.experience.test_emitter
- cog.experience.test_runtime_wiring
- cog.learning.belief.tests.test_belief_engine
- cog.learning.policy.test_policy_lifecycle
- cog.experiment.test_runtime_ab
- cog.experience.benchmark_evidence
- cog.evaluation.epistemic_suite.harness  (7/7, all gates)
- cog.evaluation.correctness.suite        (6/6 gates)
- cog.evaluation.capability.run           (learning + runtime + capability)
- cog.evaluation.versions.v2_adversarial.suite
- cog.evaluation.versions.v3_real_traces.suite

Representative capability numbers (single-domain smoke, controlled scenario):

- Learning: active_beliefs=1, accuracy/precision/recall=1.0, calibration_ece=0.02,
  belief_retention_rate=1.0, false_belief_rate=None
- Runtime: baseline_success=0.0, treatment_success=1.0, policy_lift=1.0
  (CI 0.96–1.04), policy_precision=1.0, policy_recall=0.5
- Capability: knowledge_efficiency=1.0

CAVEAT: these numbers are from a controlled synthetic single-domain scenario
(preflight prevents a dependency failure). They prove the framework is sound and
the pipeline is measurable, NOT that Cog generalizes. Generalization requires the
v2/v3 tiers run on broader / real corpora.
