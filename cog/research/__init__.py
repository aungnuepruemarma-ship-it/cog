"""RESEARCH-GRADE MODULES — evidence-gated, not part of the core loop.

Everything in this package is experimental by policy: it may run, score,
and report, but nothing here is allowed to alter core behavior until its
outputs clear evidence thresholds recorded in the Scientific Ledger. Today,
by design, nothing does — the machinery to *earn* promotion exists; the
promotions have not been earned.
"""

from cog.research.organizational import OrganizationalMathematics
from cog.research.primitive_discovery import PrimitiveDiscovery, PrimitiveScore

__all__ = ["OrganizationalMathematics", "PrimitiveDiscovery", "PrimitiveScore"]
