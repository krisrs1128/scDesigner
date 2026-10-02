"""Approximating E[sigma^2 | y] for between donor variance

This notebook includes helpers for estimating moments used in the updates for
between-donor variance in the negative binomial mixed effects models implemented
by NegBinEQTL. The outputs here are plugged into a larger EM loop. Mathematical
details in the docstrings.
"""
from typing import Callable, Dict, Optional, Tuple
import torch


Moments = Tuple[torch.Tensor, torch.Tensor]
MomentFunction = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
    Moments,
]


def laplace_moments(
    u: torch.Tensor,
    info: torch.Tensor,
    third: Optional[torch.Tensor] = None,
    fourth: Optional[torch.Tensor] = None,
) -> Moments:
    """Second moment implied by a Laplace approximation to the marginal
    likelihood

    For donor k, the contribution of the marginal likelihood is

            p(y_k; sigma^2) = ∫ exp{l_k(a; sigma^2)} da,

    where l_k is that donor's loglikelihood + their random effect drawn from the
    gaussian N(0, sigma^2_{donor} prior). The usual first-order Laplace
    approximation expands this log likelihood around the mode, which this code
    assumes has already been calculated

            H_k = -l_k''(u_k),      sigma_p,k = H_k^{-1/2},

    so

        log p(y_k; sigma^2) ≈ l_k(u_k; sigma^2) - 0.5 log H_k + const.

    Now, the donor variance can be found using an EM update,

        sigma^2 <- (1 / K) * sum_k E[u_k^2 | y],

    For the exact marginal likelihood, this can be found by differentiating
    log p(y; sigma^2) w.r.t. sigma^2 and setting it to zero. Here we do the
    analogous calculation after replacing the donor-level integrals by their
    Laplace approximation. That is, we choose sigma^2 by setting the derivative

        sum_k [l_k(u_k; sigma^2) - 0.5 log H_k]

    equal to zero. This derivative can be rearranged to look like,

        sigma^2 <- (1 / K) * sum_k m2_k,

    where m2_k is the donor contribution from donor k (a kind of approximation
    to E[u_k^2 | y]). Differentiating the Hessian log H_k involves the third
    derivatives of the loglikelihood (which is why we need it as an argument to
    this function). Working it out and using the notation,

        sigma_p,k = H_k^{-1/2},
        c3_k = l_k'''(u_k) * sigma_p,k^3,

    the it's possible to express,

        m2_k = u_k^2 + sigma_p,k * c3_k * u_k + sigma_p,k^2.

    Parameters
    ----------
    u
        Posterior mode u* for each donor.
    info
        Positive curvature H = -l''(u*) at the mode. The Laplace posterior
        standard deviation is sigma_p = H^{-1/2}.
    third
        Negative third derivative -l'''(u*).
    fourth
        Ignored. Only kept so that the API looks the same as the edgeworth
        approach.
    """
    sigma_p = info.rsqrt()
    c3 = -third * sigma_p.pow(3)
    mean = u + 0.5 * sigma_p * c3
    m2 = u.square() + sigma_p * c3 * u + sigma_p.square()
    return mean, m2


def edgeworth_moments(
    u: torch.Tensor,
    info: torch.Tensor,
    third: Optional[torch.Tensor] = None,
    fourth: Optional[torch.Tensor] = None,
) -> Moments:
    """Return Edgeworth-corrected donor contributions to the variance update

    This function has the same goal as `laplace_moments`, returning the
    donor-level quantity m2_k used in the variance update

        sigma^2 <- (1 / K) * sum_k m2_k.

    In `laplace_moments`, m2_k is a contribution coming from differentiating the
    first-order Laplace approximation to the marginal log-likelihood,

        sum_k [l_k(u_k; sigma^2) - 0.5 log H_k],

    This function uses a higher-order edgeworth expansion to approximate the
    donor posterior moment. I'll skip the derivation, but you can show that
    instead of the Laplace update,

            m2_k = u_k^2 +
                sigma_p,k * c3_k * u_k +
                sigma_p,k^2.

    an edgeworth correction updates that last term.

            m2_k = u_k^2
               + sigma_p,k * c3_k * u_k
               + sigma_p,k^2 * (1 + 5 * c3_k^2 / 4 + c4_k / 2).

    Parameters
    ----------
    u
        Posterior mode u* for each donor.
    info
        Positive curvature H = -l''(u*) at the mode. The Laplace posterior
        standard deviation is sigma_p = H^{-1/2}.
    third
        Negative third derivative -l'''(u*).
    fourth
        Negative fourth derivative -l'''(u*).
    """
    sigma_p = info.rsqrt()
    c3 = -third * sigma_p.pow(3)
    c4 = -fourth * sigma_p.pow(4)
    mean = u + 0.5 * sigma_p * c3
    m2 = (
        u.square()
        + sigma_p * c3 * u
        + sigma_p.square() * (1.0 + 1.25 * c3.square() + 0.5 * c4)
    )
    return mean, m2


POSTERIOR_MOMENTS: Dict[str, MomentFunction] = {
    "laplace": laplace_moments,
    "edgeworth": edgeworth_moments,
}
