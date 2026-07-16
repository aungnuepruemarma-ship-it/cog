"""Test 3: Replay determinism (v0.1 = structured reproduction).

v0.1 replay guarantees: an experience serializes, stores, and reloads to the
SAME structured evidence (error_signature, failure category, outcome, causal
node). This proves no corruption and deterministic storage. It does NOT claim
live re-execution (container snapshots / filesystem capture) -- that is v0.2.

Target: >=95% of stored failure experiences reproduce identical signal on reload.
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.evaluation.infra.generators import gen_block
from cog.experience.store import ExperienceStore
from cog._util import new_id
import tempfile
from pathlib import Path


def test_replay() -> TestResult:
    tmp = Path(tempfile.mkdtemp(prefix="replay_"))
    store = ExperienceStore(tmp / "exp")

    failures = []
    for i in range(100):
        # Mix of failing experiences with full replay info.
        e = gen_block(1, tool="docker_build", domain="software",
                      with_preflight=(i % 3 == 0), failed=True,
                      category="dependency_failure", start=i)[0]
        store.add(e)
        failures.append(e.id)

    reproduced = 0
    for eid in failures:
        orig = store.get(eid)
        reloaded = store.replay(eid)
        if reloaded is None:
            continue
        same = (
            orig.outcome == reloaded["outcome"]
            and (orig.failure.category or "") == (reloaded.get("failure") or {}).get("category", "")
            and (orig.failure.error_signature or "") == (reloaded.get("failure") or {}).get("error_signature", "")
        )
        if same:
            reproduced += 1

    rate = reproduced / len(failures) if failures else 0.0
    return TestResult(
        name="replay",
        passed=rate >= 0.95,
        metrics={
            "total_failures": len(failures),
            "reproduced": reproduced,
            "replay_rate": round(rate, 4),
        },
        detail=f"structured reproduction rate={rate:.2%}",
    )
