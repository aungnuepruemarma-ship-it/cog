"""Runner for the v4 multi-domain governance stress suite.

Persists a reproducible report (manifest + JSON + Markdown) under
cog/evaluation/versions/v4_stress/artifacts/v4_stress_<timestamp>/.

This stays entirely within cog/evaluation/ (evaluation territory). It does NOT
edit core runtime, policy, or experiment modules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cog.evaluation.versions.v4_stress.suite import V4StressSuite

_HERE = Path(__file__).resolve().parent


def run_and_persist(seed: int = 42, per_domain_train: int = 80,
                    per_domain_eval: int = 20, cycles: int = 20) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _HERE / "artifacts" / f"v4_stress_{ts}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    suite = V4StressSuite(seed=seed, per_domain_train=per_domain_train,
                          per_domain_eval=per_domain_eval, cycles=cycles,
                          artifact_root=artifact_dir)
    report = suite.run()

    # JSON artifact
    json_path = artifact_dir / "report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

    # Markdown artifact
    md_path = artifact_dir / "report.md"
    md_path.write_text(_to_markdown(report, seed, per_domain_train, per_domain_eval, cycles))

    # Manifest (seed + config + artifact pointers)
    manifest = {
        "suite": report.suite_name,
        "version": report.version,
        "seed": seed,
        "config": {"per_domain_train": per_domain_train,
                   "per_domain_eval": per_domain_eval, "cycles": cycles},
        "generated_at": ts,
        "artifacts": {"json": str(json_path), "markdown": str(md_path)},
        "passed": report.passed,
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(report.render())
    print(f"\nArtifacts written to: {artifact_dir}")
    print("PASSED:", report.passed)
    return report.to_dict()


def _to_markdown(report, seed, pdt, pde, cycles) -> str:
    lines = [
        f"# Cog v4 Governance Stress Report",
        f"",
        f"- Suite: {report.suite_name} ({report.version})",
        f"- Seed: {seed}",
        f"- Corpus: {pdt + pde} per domain x 7 domains",
        f"- Cycles: {cycles}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value | Target | Passed |",
        f"|--------|-------|--------|--------|",
    ]
    for m in report.metrics:
        lines.append(f"| {m.name} | {m.value} | {m.target} | {m.passed} |")
    lines += [
        f"",
        f"## Verdict: {'PASS' if report.passed else 'FAIL'}",
        f"",
        f"## Reproducibility",
        f"- Re-run: `python -m cog.evaluation.versions.v4_stress.run`",
        f"- Deterministic for fixed seed.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_and_persist()
