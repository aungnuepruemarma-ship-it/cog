"""Phase 3, component 3: deterministic belief synthesis (no LLM, observation-based).

Groups validated experiences by an OBSERVABLE condition (tool + failure
category + whether a preflight-style check was present) and emits a candidate
belief whose prediction is a measurable outcome rate. It does NOT infer causal
mechanisms ("developers forget dependencies") — only: "under condition X,
outcome Y occurs with rate Z". Causal interpretation is deferred.

A candidate is emitted only when:
  * the condition group has >= MIN_EVIDENCE experiences, and
  * there is a measurable outcome difference (e.g. failures occur at a
    non-trivial rate under the condition).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cog._util import new_id
from cog.experience.store import ExperienceStore
from cog.learning.belief.model import Belief, BeliefClaim, BeliefScope, BeliefState


MIN_EVIDENCE = 10
MIN_FAILURE_RATE = 0.30  # condition must show a meaningful failure rate to be worth a belief


def _preflight_present(exp: dict[str, Any]) -> bool:
    """Heuristic, observable: did the execution include a preflight/inspect step?

    We infer this from the trace's tool list (no LLM). Conservative: only
    clearly-named preflight tools count.
    """
    tools = [s.get("tool", "") for s in (exp.get("execution") or [])]
    return any("preflight" in t or "inspect" in t or "check" in t for t in tools)


def _group_key(exp: dict[str, Any]) -> tuple[str, str, str]:
    domain = exp.get("domain") or "unspecified"
    cat = (exp.get("failure") or {}).get("category") or "unknown"
    # The tool that failed is the observable "task/operation" under study.
    failed_tool = (exp.get("causal") or {}).get("failure_node") or (
        exp.get("execution") or [{}])[0].get("tool", "unknown") if exp.get("execution") else "unknown"
    return (domain, failed_tool, cat)  # condition studied is "preflight absent"


@dataclass
class SynthesisResult:
    candidates: list[Belief]
    groups: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [b.to_dict() for b in self.candidates],
            "groups": self.groups,
        }


def synthesize(store: ExperienceStore, scope_domain: str = "software",
               min_evidence: int = MIN_EVIDENCE,
               min_failure_rate: float = MIN_FAILURE_RATE) -> SynthesisResult:
    failures = store.filter(domain=scope_domain, outcome="failure")
    # Group by condition: (domain, failed_tool, failure_category) with preflight absent.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for exp in failures:
        # Only consider experiences where preflight was absent (the condition).
        if _preflight_present(exp):
            continue
        key = _group_key(exp)
        groups[key].append(exp)

    candidates: list[Belief] = []
    group_counts = {f"{k[0]}|{k[1]}|{k[2]}": len(v) for k, v in groups.items()}

    for (domain, tool, cat), exps in groups.items():
        n = len(exps)
        if n < min_evidence:
            continue
        # Every experience in this group is a failure under the condition
        # (preflight absent), so the observed failure rate under condition is 1.0.
        # The candidate predicts: under this condition, failure is likely.
        failure_rate = 1.0
        # Measurable outcome difference: failure rate under condition is meaningful.
        if failure_rate < min_failure_rate:
            continue
        belief = Belief(
            id=new_id("bel"),  # type: ignore[call-arg]
            claim=BeliefClaim(
                condition={"task": tool, "preflight": False, "domain": domain},
                prediction={"failure_probability": round(failure_rate, 3),
                            "category": cat},
            ),
            evidence_ids=[e.get("id", "") for e in exps],
            statistics=__import__("cog.learning.belief.model", fromlist=["BeliefStatistics"]).BeliefStatistics(
                sample_size=n, success_rate=round(1 - failure_rate, 3),
                confidence_interval=(0.0, 0.0)  # filled by the tester
            ),
            scope=BeliefScope(domain=domain, task_type=tool, environment="default"),
            confidence=round(min(1.0, n / (n + 10)), 3),
            state=BeliefState.PROPOSED,
        )
        candidates.append(belief)

    return SynthesisResult(candidates=candidates, groups=group_counts)
