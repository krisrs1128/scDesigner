"""Regularied negative-binomial marginal model."""

from typing import Dict, Sequence, Union

from .negbin import NegBin
from .penalties import CoefficientPenalty


def PenalizedNegBin(
    formula: Union[Dict[str, str], str],
    *,
    penalties: Sequence[CoefficientPenalty] = (),
    device=None,
) -> NegBin:
    """:class:`NegBin` with penalty ``P`` added to the training loss

    This is a regularized form of the NegBin marginal model.  By default, this
    does not use Poisson IRLS during initialization, since regularized settings
    typically have high-dimensional covariates, which are unstable in an
    ordinary linear model. These initial Poisson IRLS steps can be added back by
    passing ``max_iter`` to ``fit``.

    Parameters
    ----------
    formula : dict of str to str, or str
    penalties : sequence of CoefficientPenalty
        See :mod:`scdesigner.distributions.penalties`.
    device : torch.device or str, optional

    Examples
    --------
    >>> from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty
    >>> penalty = CoefficientPenalty("mean", ["x"], [1.0], QuadraticPenalty(lam=0.1))
    >>> model = PenalizedNegBin({"mean": "~ x", "dispersion": "~ 1"}, penalties=[penalty])
    >>> type(model).__name__
    'NegBin'
    """
    return NegBin(formula, penalties=penalties, device=device)
