"""Regularized Bernoulli marginal model."""

from typing import Dict, Sequence, Union

from .bernoulli import Bernoulli
from .penalties import CoefficientPenalty


def PenalizedBernoulli(
    formula: Union[Dict[str, str], str],
    *,
    penalties: Sequence[CoefficientPenalty] = (),
    device=None,
) -> Bernoulli:
    """A Bernoulli marginal with coefficient penalties added to its training loss.

    Parameters
    ----------
    formula : dict of str to str, or str
        Formula for the logit of the success probability.
    penalties : sequence of CoefficientPenalty
        See :mod:`scdesigner.distributions.penalties`.
    device : torch.device or str, optional

    Examples
    --------
    >>> from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty
    >>> penalty = CoefficientPenalty("mean", ["x"], [1.0], QuadraticPenalty(lam=0.1))
    >>> model = PenalizedBernoulli("~ x", penalties=[penalty])
    """
    return Bernoulli(formula, penalties=penalties, device=device)
