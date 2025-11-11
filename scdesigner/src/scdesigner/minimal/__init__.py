"""
Backward compatibility module for scDesigner.

DEPRECATED: This module is deprecated and will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'scdesigner.minimal' is deprecated and will be removed in a future version. ",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all user-facing classes for backward compatibility
from ..simulators import (
    BernoulliCopula,
    GaussianCopula,
    NegBinCopula,
    ZeroInflatedNegBinCopula,
    CompositeCopula,
    PositiveNMF,
)

__all__ = [
    "BernoulliCopula",
    "CompositeCopula",
    "GaussianCopula",
    "NegBinCopula",
    "PositiveNMF",
    "ZeroInflatedNegBinCopula"
]
