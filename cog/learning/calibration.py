"""Metacognition: does Cog's confidence mean what it says?

Cog attaches a confidence score to every experience, and downstream engines
treat it as a probability of success (skill benchmark scores, representation
predictive value, primitive scoring). But a score is only a *probability* if
it is **calibrated** — if the experiences Cog rated 0.9 actually succeed about
90% of the time. Nothing measured that, until now.

This engine builds a reliability diagram over the experience history and
reports three standard proper measures:

- **Brier score** — mean squared error between confidence and outcome
  (0 is perfect, lower is better);
- **ECE** (expected calibration error) — the average gap between stated
  confidence and observed accuracy across bins (0 is perfectly calibrated);
- a **calibration map** — each confidence bin's *observed* accuracy, so a raw
  heuristic score can be corrected into an empirical probability.

The report is stored and recorded as a Scientific-Ledger claim (a reproducible
self-audit); ``calibrated_confidence`` applies the latest map. This is Cog
auditing the trustworthiness of its own confidence — and correcting it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from cog.experience.record import Experience
from cog.memory.router import MemoryRouter

CALIBRATION_RECORD_ID = "concept_calibration"
_RELIABLE_ECE = 0.1  # at or below this, confidence is treated as trustworthy


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float  # observed verified-rate of experiences in this bin


@dataclass
class CalibrationReport:
    n: int
    brier: float
    ece: float
    bins: list[CalibrationBin] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.n > 0 and self.ece <= _RELIABLE_ECE

    def calibrated(self, raw: float) -> float:
        """Correct a raw confidence into the empirical success rate its bin
        exhibited. Falls back to the nearest populated bin, then to raw."""
        raw = max(0.0, min(1.0, raw))
        for b in self.bins:
            if b.lo <= raw < b.hi or (raw == 1.0 and b.hi == 1.0):
                return b.accuracy
        if not self.bins:
            return raw
        nearest = min(self.bins, key=lambda b: abs((b.lo + b.hi) / 2 - raw))
        return nearest.accuracy

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier,
            "ece": self.ece,
            "bins": [asdict(b) for b in self.bins],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationReport:
        return cls(
            n=data["n"],
            brier=data["brier"],
            ece=data["ece"],
            bins=[CalibrationBin(**b) for b in data.get("bins", [])],
        )


class CalibrationEngine:
    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def evaluate(self, pairs: list[tuple[float, bool]]) -> CalibrationReport:
        """pairs: (confidence in [0,1], verified). Builds the reliability diagram."""
        if not pairs:
            return CalibrationReport(n=0, brier=0.0, ece=0.0)
        n = len(pairs)
        brier = sum((c - int(y)) ** 2 for c, y in pairs) / n

        width = 1.0 / self.n_bins
        bins: list[CalibrationBin] = []
        ece = 0.0
        for i in range(self.n_bins):
            lo, hi = i * width, (i + 1) * width
            members = [
                (c, y) for c, y in pairs if (lo <= c < hi) or (i == self.n_bins - 1 and c == 1.0)
            ]
            if not members:
                continue
            count = len(members)
            mean_conf = sum(c for c, _ in members) / count
            accuracy = sum(int(y) for _, y in members) / count
            bins.append(
                CalibrationBin(
                    round(lo, 4), round(hi, 4), count, round(mean_conf, 4), round(accuracy, 4)
                )
            )
            ece += (count / n) * abs(mean_conf - accuracy)

        return CalibrationReport(n=n, brier=round(brier, 4), ece=round(ece, 4), bins=bins)


def evaluate_memory(memory: MemoryRouter, n_bins: int = 10) -> CalibrationReport:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    pairs = [(e.confidence, e.verified) for e in experiences]
    return CalibrationEngine(n_bins=n_bins).evaluate(pairs)


def store_calibration(memory: MemoryRouter, n_bins: int = 10) -> CalibrationReport:
    """Compute the reliability report, persist it, and file a ledger claim —
    a reproducible self-audit of confidence trustworthiness."""
    report = evaluate_memory(memory, n_bins=n_bins)
    if report.n == 0:
        return report
    memory.concepts.add(
        {"level": "calibration", **report.to_dict()},
        tags=["calibration"],
        confidence=max(0.0, 1.0 - report.ece),
        record_id=CALIBRATION_RECORD_ID,
    )
    from cog.science.ledger import Ledger

    Ledger(memory).record_claim(
        subject_id=CALIBRATION_RECORD_ID,
        hypothesis=f"Cog's confidence is calibrated (ECE {report.ece:.3f} ≤ {_RELIABLE_ECE})",
        experiment=f"reliability diagram over {report.n} experiences, {n_bins} bins",
        dataset=[],
        metrics={"brier": report.brier, "ece": report.ece, "n": report.n},
        decision="adopted" if report.reliable else "rejected",
        confidence=max(0.0, 1.0 - report.ece),
        reproducible=True,  # deterministic given the experience set
        claim_id="claim_calibration",
    )
    return report


def load_calibration(memory: MemoryRouter) -> CalibrationReport | None:
    record = memory.concepts.get(CALIBRATION_RECORD_ID)
    return CalibrationReport.from_dict(record.content) if record else None


def calibrated_confidence(memory: MemoryRouter, raw: float) -> float:
    """Correct a raw confidence using the latest stored calibration map."""
    report = load_calibration(memory)
    return report.calibrated(raw) if report else raw
