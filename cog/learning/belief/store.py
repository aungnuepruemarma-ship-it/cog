"""Phase 3, component 2: BeliefStore (SQLite, append-only history).

Tables:
    beliefs            -- current belief state (one row per id; updated only
                          via recorded transitions, never silently overwritten)
    belief_evidence    -- experience_id <-> belief_id links
    belief_transitions -- full lifecycle history (never updated/deleted)
    belief_tests       -- discriminator experiment records (never updated)

Design rule: belief STATE is never overwritten without a corresponding
transition row. History is append-only so we can audit how any belief moved.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cog._util import new_id, utc_now
from cog.learning.belief.model import Belief, BeliefState


SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    data        TEXT NOT NULL,   -- full Belief.to_dict() JSON
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS belief_evidence (
    belief_id   TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (belief_id, experience_id)
);
CREATE TABLE IF NOT EXISTS belief_transitions (
    id          TEXT PRIMARY KEY,
    belief_id   TEXT NOT NULL,
    at          TEXT NOT NULL,
    fro         TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    reason      TEXT
);
CREATE TABLE IF NOT EXISTS belief_tests (
    id          TEXT PRIMARY KEY,
    belief_id   TEXT NOT NULL,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- 'discriminator'
    result      TEXT NOT NULL    -- JSON of the experiment outcome
);
"""


class BeliefStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "beliefs.db"
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

    # ---- writes (each state change is a transition row) ---- #
    def add(self, belief: Belief) -> None:
        if not belief.is_valid():
            raise ValueError(f"refusing invalid belief {belief.id}: {belief.validate()}")
        belief.created_at = belief.created_at or utc_now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO beliefs (id, state, data, updated_at) VALUES (?,?,?,?)",
                (belief.id, belief.state.value, json.dumps(belief.to_dict()), utc_now()),
            )
            for eid in belief.evidence_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO belief_evidence (belief_id, experience_id) VALUES (?,?)",
                    (belief.id, eid),
                )

    def save_state(self, belief: Belief, fro: str, reason: str) -> None:
        """Persist a belief after a lifecycle move; records the transition."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO beliefs (id, state, data, updated_at) VALUES (?,?,?,?)",
                (belief.id, belief.state.value, json.dumps(belief.to_dict()), utc_now()),
            )
            conn.execute(
                "INSERT INTO belief_transitions (id, belief_id, at, fro, to_state, reason) "
                "VALUES (?,?,?,?,?,?)",
                (new_id("bt"), belief.id, utc_now(), fro, belief.state.value, reason),
            )

    def record_test(self, belief_id: str, kind: str, result: dict[str, Any]) -> str:
        tid = new_id("btx")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO belief_tests (id, belief_id, at, kind, result) VALUES (?,?,?,?,?)",
                (tid, belief_id, utc_now(), kind, json.dumps(result)),
            )
        return tid

    # ---- reads ---- #
    def get(self, belief_id: str) -> Belief | None:
        with self._conn() as conn:
            row = conn.execute("SELECT data FROM beliefs WHERE id = ?", (belief_id,)).fetchone()
        return Belief.from_dict(json.loads(row["data"])) if row else None

    def all(self) -> list[Belief]:
        with self._conn() as conn:
            rows = conn.execute("SELECT data FROM beliefs").fetchall()
        return [Belief.from_dict(json.loads(r["data"])) for r in rows]

    def by_state(self, state: BeliefState) -> list[Belief]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM beliefs WHERE state = ?", (state.value,)
            ).fetchall()
        return [Belief.from_dict(json.loads(r["data"])) for r in rows]

    def transitions(self, belief_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if belief_id is None:
                rows = conn.execute(
                    "SELECT * FROM belief_transitions ORDER BY at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM belief_transitions WHERE belief_id = ? ORDER BY at",
                    (belief_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def tests(self, belief_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if belief_id is None:
                rows = conn.execute("SELECT * FROM belief_tests ORDER BY at").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM belief_tests WHERE belief_id = ? ORDER BY at",
                    (belief_id,),
                ).fetchall()
        return [dict(r) for r in rows]
