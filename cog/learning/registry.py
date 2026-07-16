"""Phase 1C: the PolicyRegistry — durable, queryable policy store.

Policies live as JSONL (one line per policy, latest wins on id) so they are
human-inspectable and replayable. The registry also indexes by trigger for
fast selector lookup. No vector DB, no LLM.

The registry does NOT decide transitions — that is the PolicyLifecycle's job.
The registry only persists state and answers 'which policy matches this
context?'.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from cog.learning.lifecycle import PolicyLifecycle
from cog.learning.policy import Policy, PolicyStatus, PolicyTrigger


class PolicyRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_path = self.root / "policies.jsonl"
        self._policies: dict[str, Policy] = {}
        self.lifecycle = PolicyLifecycle()
        self._load()

    # ---- persistence ---- #
    def _load(self) -> None:
        if not self.store_path.exists():
            return
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Policy.from_dict(json.loads(line))
            self._policies[p.id] = p

    def _persist(self, policy: Policy) -> None:
        # Rewrite the whole file: small N, clarity over append-amend bookkeeping.
        with self.store_path.open("w", encoding="utf-8") as fh:
            for p in self._policies.values():
                fh.write(json.dumps(p.to_dict()) + "\n")

    # ---- writes ---- #
    def add(self, policy: Policy) -> None:
        if not policy.is_valid():
            raise ValueError(f"refusing invalid policy {policy.id}: {policy.validate()}")
        self._policies[policy.id] = policy
        self._persist(policy)

    def save(self, policy: Policy) -> None:
        """Persist after a lifecycle transition mutated it."""
        self._persist(policy)

    def get(self, policy_id: str) -> Policy | None:
        return self._policies.get(policy_id)

    def all(self) -> list[Policy]:
        return list(self._policies.values())

    def by_status(self, status: PolicyStatus) -> list[Policy]:
        return [p for p in self._policies.values() if p.status == status]

    # ---- selection (deterministic, no vector retrieval) ---- #
    def select(self, *,
               tool: str | None = None,
               failure_category: str | None = None,
               domain: str | None = None,
               operation: str | None = None) -> list[Policy]:
        """Return ACTIVE policies whose trigger matches, exact-match first.

        This is the deterministic selector the review described: a compile
        pass over executable knowledge, not RAG. Exact matches (all trigger
        fields set and matching) rank above partial matches.
        """
        scored: list[tuple[int, Policy]] = []
        for p in self._policies.values():
            if p.status != PolicyStatus.ACTIVE:
                continue
            if not p.trigger.matches(tool=tool, failure_category=failure_category,
                                     domain=domain, operation=operation):
                continue
            specificity = sum(1 for v in asdict(p.trigger).values() if v is not None)
            scored.append((specificity, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def transition_log(self) -> list[dict[str, Any]]:
        return self.lifecycle.history()
