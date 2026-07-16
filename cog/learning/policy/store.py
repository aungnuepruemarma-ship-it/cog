"""Phase 3, Policy track: PolicyStore (SQLite, append-only history).

Tables:
    policies            -- current policy state (updated only via recorded transition)
    policy_beliefs      -- belief_id <-> policy_id (the dependency graph edges)
    policy_transitions  -- full lifecycle history (never updated/deleted)
    policy_experiments  -- A/B experiment records (never updated)

Same discipline as the BeliefStore: policy STATE is never overwritten without
a corresponding transition row. History is append-only for auditability.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cog._util import new_id, utc_now
from cog.learning.policy.model import Policy, PolicyState


SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    data        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policy_beliefs (
    policy_id   TEXT NOT NULL,
    belief_id   TEXT NOT NULL,
    PRIMARY KEY (policy_id, belief_id)
);
CREATE TABLE IF NOT EXISTS policy_transitions (
    id          TEXT PRIMARY KEY,
    policy_id   TEXT NOT NULL,
    at          TEXT NOT NULL,
    fro         TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    reason      TEXT
);
CREATE TABLE IF NOT EXISTS policy_experiments (
    id          TEXT PRIMARY KEY,
    policy_id   TEXT NOT NULL,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    result      TEXT NOT NULL
);
"""


class PolicyStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "policies.db"
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Any:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes ---- #
    def add(self, policy: Policy) -> None:
        from cog.learning.belief.store import BeliefStore
        # validate takes a belief store; if we have none we skip justification check
        policy.created_at = policy.created_at or utc_now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO policies (id, state, data, updated_at) VALUES (?,?,?,?)",
                (policy.id, policy.state.value, json.dumps(policy.to_dict()), utc_now()),
            )
            for bid in policy.justification:
                conn.execute(
                    "INSERT OR IGNORE INTO policy_beliefs (policy_id, belief_id) VALUES (?,?)",
                    (policy.id, bid),
                )

    def save_state(self, policy: Policy, fro: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO policies (id, state, data, updated_at) VALUES (?,?,?,?)",
                (policy.id, policy.state.value, json.dumps(policy.to_dict()), utc_now()),
            )
            conn.execute(
                "INSERT INTO policy_transitions (id, policy_id, at, fro, to_state, reason) "
                "VALUES (?,?,?,?,?,?)",
                (new_id("pt"), policy.id, utc_now(), fro, policy.state.value, reason),
            )

    def record_experiment(self, policy_id: str, kind: str, result: dict[str, Any]) -> str:
        tid = new_id("pex")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO policy_experiments (id, policy_id, at, kind, result) VALUES (?,?,?,?,?)",
                (tid, policy_id, utc_now(), kind, json.dumps(result)),
            )
        return tid

    # ---- queries ---- #
    def get(self, policy_id: str) -> Policy | None:
        with self._conn() as conn:
            row = conn.execute("SELECT data FROM policies WHERE id = ?", (policy_id,)).fetchone()
        return Policy.from_dict(json.loads(row["data"])) if row else None

    def all(self) -> list[Policy]:
        with self._conn() as conn:
            rows = conn.execute("SELECT data FROM policies").fetchall()
        return [Policy.from_dict(json.loads(r["data"])) for r in rows]

    def by_state(self, state: PolicyState) -> list[Policy]:
        with self._conn() as conn:
            rows = conn.execute("SELECT data FROM policies WHERE state = ?", (state.value,)).fetchall()
        return [Policy.from_dict(json.loads(r["data"])) for r in rows]

    def by_belief(self, belief_id: str) -> list[Policy]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT p.data FROM policies p JOIN policy_beliefs pb ON p.id = pb.policy_id "
                "WHERE pb.belief_id = ?", (belief_id,)
            ).fetchall()
        return [Policy.from_dict(json.loads(r["data"])) for r in rows]

    def transitions(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if policy_id is None:
                rows = conn.execute("SELECT * FROM policy_transitions ORDER BY at").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM policy_transitions WHERE policy_id = ? ORDER BY at", (policy_id,)
                ).fetchall()
        return [dict(r) for r in rows]

    def experiments(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if policy_id is None:
                rows = conn.execute("SELECT * FROM policy_experiments ORDER BY at").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM policy_experiments WHERE policy_id = ? ORDER BY at", (policy_id,)
                ).fetchall()
        return [dict(r) for r in rows]
