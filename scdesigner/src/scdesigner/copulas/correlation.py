"""
Helpers for Correlation Matrices and Cholesky Factors

We can use a factorization both for computing the log-likelihood and for
sampling new pseudobservations. These helpers prevent us from recomputing those
factors multiple times.
"""

from typing import Tuple, Union
import numpy as np
import warnings

# Diagonal ridge added when a matrix is not positive definite
JITTER = 1e-8


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
