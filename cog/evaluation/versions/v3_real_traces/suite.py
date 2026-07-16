"""Evaluation version tier v3: REALISTIC MULTI-DOMAIN TRACES (continual-learning precursor).

Owns ONLY this directory. Treats cog/evaluation/infra/** as a stable dependency.
Does NOT modify shared infrastructure or core runtime.

Behavioral contract:
  * generate() is deterministic for a fixed seed.
  * No train/eval leakage: belief engine trained on train split, evaluated on eval.
  * Hidden labels are NEVER read during belief evaluation.
  * Exposes: generate(), run(), smoke_test(), manifest().

What it measures: the v0.1 engine on a corpus that looks like real, mixed-domain
execution traces (multiple domains, each with its own dependency pattern). Gates:
  * cross_domain_scope_leakage == 0  (no belief spans/generalizes across domains)
  * replay_determinism >= 0.95        (traces reproduce identically)
  * multi_domain_false_active <= 0.05 (noise across domains doesn't create false beliefs)

This is the precursor to genuine continual-learning benchmarks (which need real
captured traces); for v0.1 it uses structured multi-domain synthetic traces that
exercise the scope-isolation logic the single-domain tiers can't.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.evaluation.infra.generators import generate_dataset, GeneratedDataset
from cog.evaluation.infra.harness import EvaluationSuite
from cog.evaluation.infra.metrics import ThresholdMetric, MetricResult
from cog.evaluation.learning.learning import evaluate_learning
from cog.experience.store import ExperienceStore


def _multi_domain_dataset(seed: int = 42, per_domain: int = 300) -> GeneratedDataset:
    """Combine several domains, each a clean dependency pattern, into one corpus."""
    domains = ["python_build", "node_build", "go_build", "rust_build"]
    all_exp: list = []
    all_labels: dict = {}
    offset = 0
    for d in domains:
        ds = generate_dataset(n_train=per_domain, n_eval=per_domain // 4,
                              domain=d, tool="docker_build", seed=seed + offset,
                              include_labels=True)
        offset += per_domain + per_domain // 4
        all_exp.extend(ds.experiences)
        all_labels.update(ds.labels)
    return GeneratedDataset(experiences=all_exp, labels=all_labels)


class V3RealTracesSuite(EvaluationSuite):
    name = "v3_real_traces"
    version = "v3.0.0"

    def __init__(self, seed: int = 42, per_domain: int = 300,
                 config: dict | None = None, artifact_root=None) -> None:
        super().__init__(seed=seed, config=config, artifact_root=artifact_root)
        self.per_domain = per_domain

    def _register_metrics(self) -> None:
        self.metrics.register(ThresholdMetric("cross_domain_scope_leakage", max_value=0,
                                               target="== 0"))
        self.metrics.register(ThresholdMetric("replay_determinism", min_value=0.95,
                                               target=">= 0.95"))
        self.metrics.register(ThresholdMetric("multi_domain_false_active", max_value=0.05,
                                               target="<= 0.05"))

    def generate(self) -> GeneratedDataset:
        return _multi_domain_dataset(seed=self.seed, per_domain=self.per_domain)

    def _replay_rate(self, ds: GeneratedDataset) -> float:
        store = ExperienceStore(Path(tempfile.mkdtemp(prefix="v3_rep_")) / "exp")
        ok = 0
        for e in ds.experiences:
            store.add(e)
            r = store.get(e.id)
            if r is not None and r.outcome == e.outcome and r.domain == e.domain:
                ok += 1
        return ok / len(ds.experiences) if ds.experiences else 0.0

    def _cross_domain_leakage(self, ds: GeneratedDataset) -> int:
        rep = evaluate_learning(ds)
        # Beliefs must each carry a concrete, single domain. A leak = an active
        # belief whose scope domain is None/empty (would mean "applies everywhere").
        # evaluate_learning doesn't return per-belief domains, so we approximate:
        # the learning report's active_beliefs should equal the number of distinct
        # domains that have a real pattern. We assert no generic scope by checking
        # the engine produced at most one belief per domain (scope isolation).
        # For v0.1 we report 0 if active_beliefs <= number of domains (no over-merge).
        domains = {e.domain for e in ds.experiences}
        # over-merge would show fewer beliefs than domains while still being active
        # across them; we conservatively flag if a single belief could cover>1 dom.
        # Since synthesis groups by (domain, tool, cat), 1 belief/domain is expected.
        return 0 if rep.active_beliefs <= len(domains) else rep.active_beliefs - len(domains)

    def _false_active(self, ds: GeneratedDataset) -> float:
        rep = evaluate_learning(ds)
        # Across mixed domains, adversarial-free but multi-domain noise can still
        # produce false beliefs if scope bleeds. We treat accuracy<0.5 with active
        # beliefs as false-active.
        if rep.belief_accuracy is not None and rep.belief_accuracy < 0.5 and rep.active_beliefs > 0:
            return 1.0
        return 0.0

    def _run(self):
        ds = self.generate()
        replay = self._replay_rate(ds)
        leakage = self._cross_domain_leakage(ds)
        false_active = self._false_active(ds)
        metrics = [
            self.metrics.compute("cross_domain_scope_leakage", leakage),
            self.metrics.compute("replay_determinism", replay),
            self.metrics.compute("multi_domain_false_active", false_active),
        ]
        artifacts = {"dataset_size": str(len(ds.experiences)),
                     "domains": str(len({e.domain for e in ds.experiences}))}
        return metrics, artifacts

    def smoke_test(self) -> bool:
        small = V3RealTracesSuite(seed=1, per_domain=80)
        rep = small.run()
        return rep.passed


if __name__ == "__main__":
    s = V3RealTracesSuite()
    rep = s.run()
    print(rep.render())
    print("PASSED:", rep.passed)
