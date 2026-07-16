"""Phase 9: the Compression Engine's first implemented move — skill
subsumption.

The compiler naturally produces over-specific skills early (two identical
runs of "Compute 5 + 5" compile to an exact-replay skill) and general ones
later ("Compute {p0}" once varied goals arrive). When a parameterized skill
provably covers an exact skill — the goal matches AND replaying the general
skill on that goal produces the *identical* step sequence — the exact skill
is compressed away. Fewer artifacts, same verified capability, and the
`compresses` edge records the merge in the knowledge graph.

The capability check is what keeps this honest: compression that cannot
prove behavioral equality does not happen.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cog.learning.skill_compiler import instantiate_plan
from cog.memory.base import MemoryRecord
from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger
from cog.science.verification import FormalVerifier

if TYPE_CHECKING:
    from cog.science.pipeline import Candidate

_INACTIVE = {"retired", "compressed"}


class CompressionEngine:
    def __init__(self, memory: MemoryRouter) -> None:
        self.memory = memory

    def compress_skills(self) -> list[str]:
        """Fold exact skills into general skills that provably cover them."""
        active = [r for r in self.memory.skills.search(limit=500) if not _INACTIVE & set(r.tags)]
        actions: list[str] = []
        for specific in active:
            if specific.content.get("parameters"):
                continue  # only exact skills are folded (provable equality)
            for general in active:
                if general.id == specific.id or not general.content.get("parameters"):
                    continue
                if not self._covers(general, specific):
                    continue
                self.memory.skills.add(
                    specific.content,
                    tags=[*specific.tags, "compressed"],
                    confidence=specific.confidence,
                    record_id=specific.id,
                )
                self.memory.add_edge(general.id, specific.id, "compresses")
                ledger = Ledger(self.memory)
                # Formal evidence: behavioral-equality is a DETERMINISTIC PROOF,
                # not a stochastic experiment. Route it through the FormalVerifier
                # (the authorized producer of formal_verification claims), then
                # promote via the single sanctioned path. This keeps proofs and
                # experiments as distinct evidence classes -- the promotion gate
                # consumes either without caring how the evidence was produced.
                verifier = FormalVerifier(ledger)
                proof_claim = verifier.verify(
                    subject_id=specific.id,
                    hypothesis=(
                        f"{general.content.get('name', general.id)} reproduces"
                        f" {specific.content.get('name', specific.id)} exactly on its goal"
                    ),
                    proof=lambda: True,  # equality already established by _covers above
                    treatment_id=specific.id,
                    baseline_id=general.id,
                    dataset=[general.id],
                    claim_id=f"exp_{specific.id}_compressed",
                )
                ledger.promote_claim(
                    subject_id=specific.id,
                    hypothesis=(
                        f"{general.content.get('name', general.id)} reproduces"
                        f" {specific.content.get('name', specific.id)} exactly on its goal"
                    ),
                    experiment="behavioral-equality replay check (goal match + identical steps)",
                    dataset=[general.id],
                    metrics={"goal": specific.content.get("goal_template", "")},
                    confidence=1.0,
                    experiment_id=proof_claim.id,
                    baseline_id=general.id,
                    treatment_id=specific.id,
                    claim_id=f"claim_{specific.id}_compressed",
                )
                actions.append(
                    f"compressed {specific.content.get('name', specific.id)}"
                    f" [{specific.content.get('goal_template', '')!r}] into"
                    f" {general.content.get('name', general.id)}"
                    f" [{general.content.get('goal_template', '')!r}]"
                )
                break
        return actions

    def _covers(self, general: MemoryRecord, specific: MemoryRecord) -> bool:
        """True iff replaying `general` on `specific`'s goal reproduces
        `specific`'s exact step sequence — behavioral equality, not vibes."""
        goal = specific.content.get("goal_template", "")
        matched = re.fullmatch(general.content.get("goal_regex", ""), goal)
        if not matched:
            return False
        bound = dict(zip(general.content.get("parameters", []), matched.groups(), strict=False))
        replayed = instantiate_plan(general, bound)
        general_steps = [(step.tool, step.args) for step in replayed.steps]
        specific_steps = [
            (step["tool"], step["args"]) for step in specific.content.get("steps", [])
        ]
        return general_steps == specific_steps

    def discover_candidates(self, memory: MemoryRouter) -> list["Candidate"]:
        """CandidateSource contract: emit a FORMAL promotion candidate for every
        provable compression, WITHOUT mutating skills or promoting anything.

        This is purely observational -- it owns the domain knowledge (which
        general skill provably covers which exact skill, via `_covers`) and
        packages it as a ready-to-submit Candidate. The orchestrator decides
        whether/when to submit it to the scheduler. The existing
        `compress_skills()` promotion path is untouched; this only exposes the
        same discoveries as candidates so adoption can be centralized later.
        """
        from cog.science.pipeline import Candidate  # local: avoid import cycle

        active = [
            r for r in memory.skills.search(limit=500) if not _INACTIVE & set(r.tags)
        ]
        out: list[Candidate] = []
        for specific in active:
            if specific.content.get("parameters"):
                continue
            for general in active:
                if general.id == specific.id or not general.content.get("parameters"):
                    continue
                if not self._covers(general, specific):
                    continue
                hypothesis = (
                    f"{general.content.get('name', general.id)} reproduces"
                    f" {specific.content.get('name', specific.id)} exactly on its goal"
                )
                out.append(
                    Candidate(
                        subject_id=specific.id,
                        hypothesis=hypothesis,
                        kind="formal",
                        # Ephemeral runtime state: the proof is trivially True
                        # because _covers already established behavioral equality.
                        producer_spec=lambda: True,
                        baseline_id=general.id,
                        treatment_id=specific.id,
                    )
                )
                break
        return out


def compress_and_store(memory: MemoryRouter) -> list[str]:
    return CompressionEngine(memory).compress_skills()


def discover_compression_candidates(memory: MemoryRouter) -> list["Candidate"]:
    """Module-level CandidateSource entry point (mirrors *_and_store helpers)."""
    return CompressionEngine(memory).discover_candidates(memory)
