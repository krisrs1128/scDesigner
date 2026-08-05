r"""
Gaussian copula math: correlation factorization, log-likelihood, and sampling

The copula loglikelihood of an entry is the joint normal log-density minus the
independent one,

.. math::
    \ell(z) = \log \phi_\Sigma(z) - \sum_j \log \phi(z_j),

For each entry, we use :func:`batch_log_likelihood` to compute the difference
directly, then form the group total (:func:`group_log_likelihood`) using a
trace formula documented in that function's docstring. Both rely on a
Cholesky factor of the group's correlation matrix, produced once by
:func:`factorize_correlation` and reused rather than recomputed.
"""

from typing import Optional, Tuple, Union
import numpy as np
import warnings
from scipy.linalg import cho_solve
from scipy.stats import norm, multivariate_normal
from ..base.copula import CovarianceStructure

# Diagonal ridge added when a matrix is not positive definite
JITTER = 1e-8


# ------------------------------------------------------------------
# correlation and factorization
# ------------------------------------------------------------------
def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """
    Rescale a covariance matrix to unit diagonal.
    """
    std = np.sqrt(np.clip(np.diag(cov), a_min=np.finfo(float).eps, a_max=None))
    corr = cov / np.outer(std, std)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def cholesky_factor(corr: np.ndarray, group: Union[str, int]) -> np.ndarray:
    """
    Cholesky factor of a correlation matrix, or an error naming the remedy.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"The copula correlation matrix for group '{group}' is not positive "
            f"definite ({corr.shape[0]} x {corr.shape[1]}). Consider reducing top_k, "
            "or choosing a regularized covariance estimator."
        ) from exc


def factorize_correlation(
    corr: np.ndarray, group: Union[str, int], jitter: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factorize a correlation matrix

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix, shape ``(p, p)``.
    group : str or int
        Group label, used in messages.
    jitter : bool, optional
        Add jitter to the diagonal of the original input isn't PSD.

    Returns
    -------
    corr : np.ndarray
        The matrix used after applying potential ridge
    factor : np.ndarray
        Its lower Cholesky factor.
    """
    if not jitter:
        return corr, cholesky_factor(corr, group)

    try:
        return corr, np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        pass

    candidate = corr.copy()
    candidate.flat[:: corr.shape[0] + 1] += JITTER
    candidate = covariance_to_correlation(candidate)
    try:
        factor = np.linalg.cholesky(candidate)
    except np.linalg.LinAlgError:
        return corr, cholesky_factor(corr, group)

    warnings.warn(
        f"The correlation matrix for group '{group}' needed a ridge of "
        f"{JITTER:g} to factorize."
    )
    return candidate, factor


# ------------------------------------------------------------------
# log-likelihood and sampling
# ------------------------------------------------------------------
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
