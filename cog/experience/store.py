"""Experience store — deterministic, queryable evidence layer (schema v0.1).

For v0.1 we deliberately do NOT use a vector database. SQLite is enough for
the three things the learning system needs:

  * filtering   -- "all software dependency failures"
  * statistics  -- success rate by domain, failure-category counts
  * replay      -- pull a structured experience back exactly as recorded

The on-disk layout mirrors the user's proposed tree:

    learning/
    ├── experiences/        (JSONL mirror, one line per experience)
    │   └── exp_*.jsonl
    └── experience.db      (SQLite, the source of truth for queries)

Every query returns plain dicts so downstream policy analysis can consume
them without importing the dataclasses.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from cog._util import new_id
from cog.experience.record import Experience

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id                      TEXT PRIMARY KEY,
    task_id                 TEXT NOT NULL,
    goal                    TEXT NOT NULL,
    domain                  TEXT NOT NULL DEFAULT 'unspecified',
    difficulty              TEXT NOT NULL DEFAULT 'unspecified',
    outcome                 TEXT NOT NULL,
    verified                INTEGER NOT NULL,
    confidence              REAL NOT NULL,
    failure_category        TEXT,
    error_signature         TEXT,
    created_at              TEXT NOT NULL,
    data                    TEXT NOT NULL  -- full Experience.to_dict() JSON
);
CREATE INDEX IF NOT EXISTS idx_domain        ON experiences(domain);
CREATE INDEX IF NOT EXISTS idx_failure_cat   ON experiences(failure_category);
CREATE INDEX IF NOT EXISTS idx_outcome       ON experiences(outcome);
CREATE INDEX IF NOT EXISTS idx_created       ON experiences(created_at);
"""


class ExperienceStore:
    """SQLite-backed store with a JSONL mirror for human inspection."""

    def __init__(self, root: str | Path, *, jsonl_mirror: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "experience.db"
        self.exp_dir = self.root / "experiences"
        self.jsonl_mirror = jsonl_mirror
        if self.jsonl_mirror:
            self.exp_dir.mkdir(exist_ok=True)
        self._init_db()

    # ----- lifecycle ----- #
    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- writes ----- #
    def add(self, exp: Experience) -> str:
        if not exp.is_valid():
            raise ValueError(
                f"refusing to store invalid experience {exp.id}: {exp.validate()}"
            )
        data = exp.to_dict()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiences
                (id, task_id, goal, domain, difficulty, outcome, verified,
                 confidence, failure_category, error_signature, created_at, data)
                VALUES (:id, :task_id, :goal, :domain, :difficulty, :outcome,
                        :verified, :confidence, :failure_category,
                        :error_signature, :created_at, :data)
                """,
                {
                    "id": data["id"],
                    "task_id": data["task_id"],
                    "goal": data["goal"],
                    "domain": data["domain"],
                    "difficulty": data["difficulty"],
                    "outcome": data["outcome"],
                    "verified": int(data["verification"].get("verified", False)),
                    "confidence": data["verification"].get("confidence", 0.0),
                    "failure_category": (data["failure"] or {}).get("category"),
                    "error_signature": (data["failure"] or {}).get("error_signature"),
                    "created_at": data["created_at"],
                    "data": json.dumps(data),
                },
            )
        if self.jsonl_mirror:
            with (self.exp_dir / f"{exp.id}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
        return exp.id

    # ----- reads ----- #
    def get(self, experience_id: str) -> Experience | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        if row is None:
            return None
        return Experience.from_dict(json.loads(row["data"]))

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    # ----- the canonical query from the spec ----- #
    def failures_by(self, failure_type: str, domain: str) -> list[dict[str, Any]]:
        """Exact query requested in the review:

            SELECT * FROM experiences
            WHERE failure_type='dependency_failure'
            AND domain='software';

        Mapped onto our column names (failure_category / domain).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT data FROM experiences
                WHERE failure_category = ? AND domain = ?
                ORDER BY created_at
                """,
                (failure_type, domain),
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ----- filtering / statistics / replay ----- #
    def filter(
        self,
        *,
        domain: str | None = None,
        outcome: str | None = None,
        verified: bool | None = None,
        failure_category: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        if verified is not None:
            clauses.append("verified = ?")
            params.append(int(verified))
        if failure_category is not None:
            clauses.append("failure_category = ?")
            params.append(failure_category)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT data FROM experiences {where} ORDER BY created_at", params
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregates the learning system needs before any policy work."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            by_outcome = dict(
                conn.execute(
                    "SELECT outcome, COUNT(*) FROM experiences GROUP BY outcome"
                ).fetchall()
            )
            by_domain = dict(
                conn.execute(
                    "SELECT domain, COUNT(*) FROM experiences GROUP BY domain"
                ).fetchall()
            )
            by_failure = dict(
                conn.execute(
                    "SELECT failure_category, COUNT(*) FROM experiences "
                    "WHERE failure_category IS NOT NULL GROUP BY failure_category"
                ).fetchall()
            )
            verified_ok = conn.execute(
                "SELECT COUNT(*) FROM experiences WHERE verified = 1"
            ).fetchone()[0]
        return {
            "total": total,
            "by_outcome": by_outcome,
            "by_domain": by_domain,
            "by_failure_category": by_failure,
            "verified_count": verified_ok,
            "verified_rate": (verified_ok / total) if total else 0.0,
        }

    def replay(self, experience_id: str) -> dict[str, Any] | None:
        """Return the structured evidence for one run, verbatim.

        Replay capability is the foundation for: EXP experiments, policy
        shadow validation, skill discovery, and future HTN method learning.
        """
        exp = self.get(experience_id)
        if exp is None:
            return None
        return exp.to_dict()
