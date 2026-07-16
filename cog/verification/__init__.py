"""Phase 3: verification — nothing enters long-term memory until verified."""

from cog.verification.checks import (
    BudgetCheck,
    Check,
    CheckResult,
    NoErrorsCheck,
    OutputCheck,
    default_checks,
)
from cog.verification.corroboration import Corroboration, canonical
from cog.verification.pipeline import VerificationPipeline, VerificationReport

__all__ = [
    "BudgetCheck",
    "Check",
    "CheckResult",
    "Corroboration",
    "NoErrorsCheck",
    "OutputCheck",
    "VerificationPipeline",
    "VerificationReport",
    "canonical",
    "default_checks",
]
