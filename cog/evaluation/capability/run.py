"""Smoke runner for the capability suite: builds a labeled dataset, runs the
belief engine, then evaluates learning + runtime + aggregate capability.

Run: python -m cog.evaluation.capability.run
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from cog.evaluation.infra.generators import generate_dataset, GeneratedDataset
from cog.evaluation.learning.learning import evaluate_learning
from cog.evaluation.runtime.runtime import evaluate_runtime
from cog.evaluation.capability.suite import CapabilitySuite, CapabilityReport
from cog.experience.store import ExperienceStore
from cog.learning.belief.engine import BeliefEngine
from cog.learning.belief.model import BeliefState
from cog.learning.belief.store import BeliefStore
from cog.learning.policy.model import Policy, PolicyEffect, PolicyState
from cog.learning.policy.store import PolicyStore


def _active_policies_from_dataset(dataset: GeneratedDataset) -> list[Policy]:
    """Run the belief engine on the train split, then synthesize a 1:1 policy
    per ACTIVE belief so the runtime benchmark has something to inject."""
    split = len(dataset.experiences) - len(dataset.experiences) // 5
    train = dataset.experiences[:split]
    db = Path(mkdtemp(prefix="cap_bel_")) / "beliefs.db"
    store = ExperienceStore(db.parent / "train_exp")
    for e in train:
        store.add(e)
    bstore = BeliefStore(db)
    engine = BeliefEngine(bstore, store)
    cases = engine.run(min_evidence=10)
    active = [c.belief for c in cases if c.belief.state == BeliefState.ACTIVE]

    # Derive a policy per active belief (prescriptive intervention).
    pstore = PolicyStore(Path(mkdtemp(prefix="cap_pol_")) / "policies.db")
    policies: list[Policy] = []
    for i, b in enumerate(active):
        p = Policy(
            id=f"pol_from_{b.id}",
            action="dep_preflight",  # the effective intervention per the labels
            trigger={"task_type": b.scope.task_type, "domain": b.scope.domain},
            justification=[b.id],
            state=PolicyState.ACTIVE,
            confidence=0.9,
            evidence_ids=list(b.evidence_ids),
            expected_effect=PolicyEffect(metric="docker_build_failure_rate",
                                          direction="decrease", expected_delta=0.3),
            created_at="t", last_validated="t",
        )
        pstore.add(p)
        policies.append(p)
    return policies


def main() -> None:
    dataset = generate_dataset(n_train=800, n_eval=200, seed=42, include_labels=True)
    policies = _active_policies_from_dataset(dataset)

    learning = evaluate_learning(dataset)
    runtime = evaluate_runtime(dataset, policies)
    suite = CapabilitySuite(dataset, policies, seed=42)
    rep = suite.run()
    cap = suite.last_report

    print("=== LEARNING REPORT ===")
    for k, v in learning.to_dict().items():
        print(f"  {k:22s}: {v}")
    print("\n=== RUNTIME REPORT ===")
    for k, v in runtime.to_dict().items():
        print(f"  {k:22s}: {v}")
    print("\n=== CAPABILITY REPORT ===")
    print(f"  knowledge_efficiency : {cap.knowledge_efficiency}")
    print("\n=== SUITE ===")
    print(rep.render())
    print("PASSED:", rep.passed)


if __name__ == "__main__":
    main()
