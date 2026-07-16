"""Governance-v0.2: tiered decay for beliefs (consolidation pass).

Decay only LOWERS a belief's privilege; it NEVER deletes knowledge. Deletion
is a separate, gated governance action (retire()). The purpose is to keep
ACTIVE beliefs honest: an unused, unconfirmed, or heavily-contradicted belief
should lose its privilege over time, but a very high-confidence belief should
require MORE evidence to move, not be impossible to change.

Tiers (COG_LEARNING_GOVERNANCE.md section 4):
    confidence > 0.95 : extremely slow decay
    0.70 - 0.95       : normal decay
    < 0.70            : aggressive decay

Pure function: given a belief + clock params, returns the suggested target
state. It does not mutate the belief and does not touch the DB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cog.learning.belief.model import Belief, BeliefState

# decay tuning (all additive / configurable; defaults chosen for clarity)
STALE_HORIZON_DAYS_NORMAL = 30.0
STALE_HORIZON_DAYS_SLOW = 180.0
STALE_HORIZON_DAYS_AGGRESSIVE = 7.0
AGGRESSIVE_CONFIRM_FLOOR = 3        # below this, aggressive decay demotes hard
CONTRADICTION_FAST_TRACK = 3        # above this, fast-track to CHALLENGED


def _parse(ts: str | None) -> datetime:
    if not ts:
        # no usage record -> treat as maximally stale (epoch)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _days_since(ts: str | None, now: datetime) -> float:
    return (now - _parse(ts)).total_seconds() / 86400.0


def tiered_decay(
    belief: Belief,
    now: datetime | None = None,
    *,
    stale_horizon_normal: float = STALE_HORIZON_DAYS_NORMAL,
    stale_horizon_slow: float = STALE_HORIZON_DAYS_SLOW,
    stale_horizon_aggressive: float = STALE_HORIZON_DAYS_AGGRESSIVE,
    confirm_floor: int = AGGRESSIVE_CONFIRM_FLOOR,
    contradiction_fast_track: int = CONTRADICTION_FAST_TRACK,
) -> BeliefState:
    """Return the suggested (lower-privilege) state for a belief.

    Never returns RETIRED. The caller decides whether to apply the transition.
    """
    now = now or datetime.now(timezone.utc)
    conf = belief.confidence
    stale_days = _days_since(belief.last_used, now)
    confirmations = belief.confirmation_count

    # Fast-track heavily-contradicted beliefs toward CHALLENGED (still not deletion).
    if belief.contradiction_count >= contradiction_fast_track:
        return BeliefState.CHALLENGED

    # Gating: only ACTIVE/SUPPORTED beliefs can be demoted by decay.
    if belief.state not in (BeliefState.ACTIVE, BeliefState.SUPPORTED):
        return belief.state  # leave as-is

    if conf > 0.95:
        # extremely slow: only demote if very stale AND weakly confirmed
        if stale_days > stale_horizon_slow and confirmations < confirm_floor:
            return BeliefState.TESTING
        return belief.state
    elif conf >= 0.70:
        # normal: demote to TESTING if stale
        if stale_days > stale_horizon_normal:
            return BeliefState.TESTING
        return belief.state
    else:
        # aggressive: demote if stale; hard-demote (CHALLENGED) if also unconfirmed
        if stale_days > stale_horizon_aggressive:
            if confirmations < confirm_floor:
                return BeliefState.CHALLENGED
            return BeliefState.TESTING
        return belief.state
