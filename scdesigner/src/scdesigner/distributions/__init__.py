"""Marginal distribution implementations."""

from .negbin import NegBin
from .zero_inflated_negbin import ZeroInflatedNegBin
from .gaussian import Gaussian
from .bernoulli import Bernoulli

__all__ = [
    "NegBin",
    "ZeroInflatedNegBin",
    "Gaussian",
    "Bernoulli",
]

