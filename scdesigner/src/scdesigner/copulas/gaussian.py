r"""
Loglikelihood and sampling for a Gaussian copula given a fitted correlation

The copula loglikelihood of an entry is the joint normal log-density minus the
independent one,

.. math::
    \ell(z) = \log \phi_\Sigma(z) - \sum_j \log \phi(z_j),

Therefore, for each entry, we use :func:`batch_log_likelihood` to compute the
difference directly, then we form the (:func:`group_log_likelihood`) using a
trace formula documented in that function's docstring.
"""

from typing import Optional
import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm, multivariate_normal
from ..base.copula import CovarianceStructure


def group_log_likelihood(
    factor: np.ndarray, second_moment: np.ndarray, n_samples: int
) -> float:
    r"""
    Copula log-likelihood for a group

    Adding :math:`\ell(z_i)` over the group gives

    .. math::

        \sum_i \ell(z_i) = -\tfrac{1}{2}\left( n \log|\Sigma|
            + \operatorname{tr}(\Sigma^{-1} S) - \operatorname{tr}(S) \right),

    where :math:`S = \sum_i z_i z_i^\top`. Note that we assume the mean is zero,
    matching the copula assumptions. So, we only ever need S in our inputs.

    Parameters
    ----------
    factor : np.ndarray
        Lower Cholesky factor of the group's correlation matrix.
    second_moment : np.ndarray
        Uncentered second moment :math:`S`.
    n_samples : int
        Cells in the group.

    Returns
    -------
    float
        Total log-likelihood.
    """
    log_det = 2.0 * np.log(np.diag(factor)).sum()
    quadratic = np.trace(cho_solve((factor, True), second_moment))
    return -0.5 * (n_samples * log_det + quadratic - np.trace(second_moment))


def batch_log_likelihood(
    z: np.ndarray, cov_struct: CovarianceStructure
) -> np.ndarray:
    """
    Per-entry copula log-likelihood for a batch of gaussianized scores.

    Unlike :func:`group_log_likelihood`, this keeps the cells separate, so it
    needs the scores themselves.

    Parameters
    ----------
    z : np.ndarray
        Normal scores, shape ``(n_cells, n_genes)``.
    cov_struct : CovarianceStructure
        Fitted structure for the group these cells belong to.

    Returns
    -------
    np.ndarray
        Log-likelihood per cell, shape ``(n_cells,)``.
    """
    z_modeled = z[:, cov_struct.modeled_indices]
    joint = multivariate_normal.logpdf(
        z_modeled,
        np.zeros(cov_struct.num_modeled_genes),
        cov_struct.cov.values,
        allow_singular=False,
    )
    return joint - norm.logpdf(z_modeled).sum(axis=1)


def sample_pseudo_obs(
    n_samples: int,
    cov_struct: CovarianceStructure,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw dependent uniform pseudoobservations from a CovarianceStructure

    Genes outside the CovarianceStructure's top_k block are drawn independently.

    Parameters
    ----------
    n_samples : int
        Entries to generate.
    cov_struct : CovarianceStructure
        Jointly modeled block plus remaining independent genes.
    rng : np.random.Generator, optional
        Optional random number generator.

    Returns
    -------
    np.ndarray
        Uniform values, shape ``(n_samples, total_genes)``.
    """
    draw = np.random if rng is None else rng

    # jointly correlated block
    u = np.zeros((n_samples, cov_struct.total_genes))
    z_modeled = draw.multivariate_normal(
        mean=np.zeros(cov_struct.num_modeled_genes),
        cov=cov_struct.cov.values,
        size=n_samples,
    )
    u[:, cov_struct.modeled_indices] = norm.cdf(z_modeled)

    # remaining independent genes
    if cov_struct.num_remaining_genes > 0:
        z_remaining = draw.normal(
            loc=0, scale=1, size=(n_samples, cov_struct.num_remaining_genes)
        )
        u[:, cov_struct.remaining_indices] = norm.cdf(z_remaining)

    return u
