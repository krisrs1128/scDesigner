"""Regularized zero-inflated truncated Gaussian marginal model."""

from typing import Dict, Sequence, Union

from .zero_inflated_truncated_gaussian import ZeroInflatedTruncatedGaussian
from .penalties import CoefficientPenalty


def PenalizedZeroInflatedTruncatedGaussian(
    formula: Union[Dict[str, str], str],
    *,
    prior_alpha: float = 1.1,
    penalties: Sequence[CoefficientPenalty] = (),
    device=None,
) -> ZeroInflatedTruncatedGaussian:
    """A regularized zero-inflated truncated Gaussian marginal

    The penalty ``P`` is added to the training loss, together with a
    Beta(``prior_alpha``, ``prior_alpha``) prior term already present in
    :class:`ZeroInflatedTruncatedGaussian`.

    Parameters
    ----------
    formula : dict of str to str, or str
    prior_alpha : float
        See :class:`ZeroInflatedTruncatedGaussian`.
    penalties : sequence of CoefficientPenalty
        See :mod:`scdesigner.distributions.penalties`.
    device : torch.device or str, optional

    Examples
    --------
    >>> from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty
    >>> penalty = CoefficientPenalty("mean", ["x"], [1.0], QuadraticPenalty(lam=0.1))
    >>> model = PenalizedZeroInflatedTruncatedGaussian({"mean": "~ x"}, penalties=[penalty])
    >>> type(model).__name__
    'ZeroInflatedTruncatedGaussian'
    """
    return ZeroInflatedTruncatedGaussian(
        formula, prior_alpha=prior_alpha, penalties=penalties, device=device
    )
