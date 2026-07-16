"""Phase 10: the Abstraction Engine (weekly loop).

Asks "what higher-level concept explains several representations?" --
representations that share a subject family (e.g. "Fix Python import",
"Fix Rust dependency", "Fix Java package" all reduce to *resolve dependency
failure*) merge into one abstraction. This is the capability jump: instead of
storing individual victorious experiences, Cog discovers reusable *concepts*
(repeated algorithms, generalized procedures, common failure modes), which is
what makes skill replay transfer across superficially different tasks.

Compression is structural and measurable: len(abstractions) < len(representations)
merged, and each abstraction names a competence broader than any single
representation it spans.
"""

from __future__ import annotations

import hashlib
import re

from cog.learning.artifacts import Abstraction
from cog.memory.router import MemoryRouter


def _abstraction_id(subject: str) -> str:
    return "abs_" + hashlib.sha1(subject.encode()).hexdigest()[:12]


# Stopwords that carry no abstraction signal -- they are task-specific nouns,
# not the underlying competence. Stripping them yields the shared *family*.
_SUBJECT_NOISE = re.compile(
    r"\b(python|rust|java|go|javascript|typescript|node|cpp|c\+\+|c#|ruby|php|"
    r"docker|kubernetes|k8s|vm|terraform|aws|gcp|azure|linux|macos|windows|"
    r"package|dependency|import|module|file|config|service|api|database|db|"
    r"version|build|test|deploy|script|cli|app|server|client)\b",
    re.IGNORECASE,
)


def _family_key(subject: str) -> str:
    """Reduce a representation subject to its competence family.

    "competence: python import (failure)" and "competence: rust dependency (failure)"
    both collapse to "failure" -> same abstraction family.
    """
    cleaned = _SUBJECT_NOISE.sub(" ", subject.lower())
    tokens = [t for t in re.findall(r"[a-z0-9_]+", cleaned) if t]
    # The surviving tokens (e.g. "failure", "resolution", "error", "verify")
    # are the cross-language competence signal. Sort for a stable key.
    return " ".join(sorted(set(tokens))) or "abstract"


class AbstractionEngine:
    def __init__(self, min_merge: int = 2) -> None:
        self.min_merge = min_merge

    def merge(self, representations: list[Abstraction]) -> list[Abstraction]:
        # NOTE: callers pass Representation-shaped dicts via merge_records();
        # this typed method mirrors the other engines' reduce() contract.
        by_family: dict[str, list[Abstraction]] = {}
        for rep in representations:
            key = _family_key(_rep_subject(rep))
            by_family.setdefault(key, []).append(rep)

        abstractions: list[Abstraction] = []
        for family, group in by_family.items():
            if len(group) < self.min_merge:
                continue  # nothing to abstract yet
            abstractions.append(
                Abstraction(
                    id=_abstraction_id(family),
                    name=f"abstraction: {family or 'general'}",
                    merges=sorted(g.id for g in group),
                )
            )
        return abstractions


def _rep_subject(rep: Abstraction) -> str:
    """Representations are passed as lightweight objects with a .name/.merges.

    We recover the human-readable subject from the name so family-keying works
    even though Representation.name is the display string, not the raw subject.
    """
    return getattr(rep, "name", "") or " ".join(getattr(rep, "merges", []))


def merge_records(memory: MemoryRouter, min_merge: int = 2) -> list[Abstraction]:
    """Cluster stored *representations* (concepts tagged 'representation') into
    abstractions by competence family, store them, and link edges.

    Returns the abstractions actually created (empty => no compression yet).
    """
    rep_records = memory.concepts.search(tags=["representation"], limit=500)

    # Group representations by family; only families with >= min_merge reps
    # become abstractions (otherwise we are inventing concepts from thin air).
    families: dict[str, list[dict]] = {}
    for r in rep_records:
        name = r.content.get("name", "")
        family = _family_key(name)
        families.setdefault(family, []).append(
            {"id": r.id, "name": name, "merges": r.content.get("merges", [])}
        )

    abstractions: list[Abstraction] = []
    for family, group in families.items():
        if len(group) < min_merge:
            continue
        abs_id = _abstraction_id(family)
        abstraction = Abstraction(
            id=abs_id,
            name=f"abstraction: {family or 'general'}",
            merges=sorted(g["id"] for g in group),
        )
        memory.concepts.add(
            {
                "level": "abstraction",
                "name": abstraction.name,
                "family": family,
                "merges": abstraction.merges,
                "compression": f"{len(abstraction.merges)} representations -> 1 abstraction",
                "source_names": [g["name"] for g in group],
            },
            tags=["abstraction"],
            confidence=min(1.0, len(group) / 5),
            record_id=abs_id,
        )
        for rep_id in abstraction.merges:
            memory.add_edge(abs_id, rep_id, "merges")  # Phase 15 graph link
        abstractions.append(abstraction)
    return abstractions
