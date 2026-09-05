r"""
Shrinkage covariance estimators.

We can improve the stability of our estimators by regularizing. This especially
helps in rare cell tpes.  We follow scikit-learn's notation and base the
implementation on sufficient statistics, so the copula can be estimated
batchwise.
"""

from abc import abstractmethod
import numpy as np

from .base import register_estimator
from .moments import _MomentEstimator


def shrink_covariance(cov: np.ndarray, intensity: float) -> np.ndarray:
    """
    Shrink a covariance matrix toward the (scaled) identity.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of shape ``(p, p)``.
    intensity : float
        Shrinkage intensity in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        ``(1 - intensity) * cov + intensity * mu * I`` with
        ``mu = trace(cov) / p``. Positive definite for any positive intensity
        and any positive semi-definite ``cov``.
    """
    p = cov.shape[0]
    mu = np.trace(cov) / p
    shrunk = (1.0 - intensity) * cov
    shrunk.flat[:: p + 1] += intensity * mu
    return shrunk


def oas_intensity(cov: np.ndarray, n_samples: int) -> float:
    """
    Oracle Approximating Shrinkage intensity

    Based on ``sklearn.covariance.OAS`` (Chen et al. 2010, Eq. 23 and the
    sklearn implementation).

    Parameters
    ----------
    cov : np.ndarray
        Biased (divided by ``n``) centered covariance of shape ``(p, p)``.
    n_samples : int
        Number of observations the covariance was computed from.

    Returns
    -------
    float
        Shrinkage intensity in ``[0, 1]``.
    """
    p = cov.shape[0]
    if p == 1:
        return 0.0

    alpha = np.mean(cov**2)
    mu_squared = (np.trace(cov) / p) ** 2
    numerator = alpha + mu_squared
    denominator = (n_samples + 1) * (alpha - mu_squared / p)
    return 1.0 if denominator == 0 else float(min(numerator / denominator, 1.0))


def ledoit_wolf_intensity(
    cov: np.ndarray,
    n_samples: int,
    mean: np.ndarray,
    squared_sums: np.ndarray,
    second_moments: np.ndarray,
    third_moments: np.ndarray,
    fourth_moments: np.ndarray,
) -> float:
    r"""
    Ledoit-Wolf shrinkage from streaming statistics.

    ``sklearn.covariance.ledoit_wolf_shrinkage`` needs three quantities of the
    centered data ``X``: ``trace(C)``, ``delta_ = ||X^T X||_F^2 / n^2`` and
    ``beta_ = sum_ij sum_k x_ki^2 x_kj^2``. We center the data in a streaming
    way. We take ``m = sum_z / n`` and ``sum_k z_ki = n m_i``, then
    ``(z_ki - m_i)^2 (z_kj - m_j)^2`` collapses into``-3 n m_i^2 m_j^2``:

    .. math::
        M_{ij} = A_{ij} - 2 m_j B_{ij} - 2 m_i B_{ji}
                 + m_i^2 q_j + q_i m_j^2 + 4 m_i m_j S_{ij}
                 - 3 n m_i^2 m_j^2
    with ``beta_ = sum(M)``, and ``delta_ = ||C||_F^2`` because ``X^T X = n C``.

    Parameters
    ----------
    cov : np.ndarray
        Biased centered covariance ``S / n - outer(m, m)``, shape ``(p, p)``.
    n_samples : int
        Number of observations.
    mean : np.ndarray
        Per-feature mean ``m``, shape ``(p,)``.
    squared_sums : np.ndarray
        ``q_i = sum_k z_ki^2``, shape ``(p,)``.
    second_moments : np.ndarray
        ``S_ij = sum_k z_ki z_kj``, shape ``(p, p)``.
    third_moments : np.ndarray
        ``B_ij = sum_k z_ki^2 z_kj``, shape ``(p, p)``.
    fourth_moments : np.ndarray
        ``A_ij = sum_k z_ki^2 z_kj^2``, shape ``(p, p)``.

    Returns
    -------
    float
        Shrinkage intensity in ``[0, 1]``.
    """
    p = cov.shape[0]
    trace_cov = np.trace(cov)
    mu = trace_cov / p
    delta_ = np.sum(cov**2)

    mean_squared = mean**2
    centered_fourth = (
        fourth_moments
        - 2.0 * mean[None, :] * third_moments
        - 2.0 * mean[:, None] * third_moments.T
        + mean_squared[:, None] * squared_sums[None, :]
        + squared_sums[:, None] * mean_squared[None, :]
        + 4.0 * np.outer(mean, mean) * second_moments
        - 3.0 * n_samples * np.outer(mean_squared, mean_squared)
    )
    beta_ = centered_fourth.sum()

    beta = (beta_ / n_samples - delta_) / (p * n_samples)
    delta = (delta_ - 2.0 * mu * trace_cov + p * mu**2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else float(beta / delta)


class _ShrinkageEstimator(_MomentEstimator):
    """
    Parent class for shrinkage estimators

    Subclasses implement :meth:`_intensity`. The shrunk estimate is positive
    definite for any positive intensity, so :attr:`regularized` is ``True``.

    Attributes
    ----------
    intensity_ : float
        Shrinkage intensity chosen for this group, set by ``finalize``.
    """

    regularized = True

    @abstractmethod
    def _intensity(self, cov: np.ndarray) -> float:
        """Shrinkage intensity for the biased covariance ``cov``."""

    def _finalize(self) -> np.ndarray:
        cov = self._biased_covariance()
        self.intensity_ = self._intensity(cov)
        return shrink_covariance(cov, self.intensity_)

    def diagnostics(self):
        return {"intensity": self.intensity_}


class OAS(_ShrinkageEstimator):
    """
    Oracle Approximating Shrinkage (see also ``sklearn.covariance.OAS``)

    Examples
    --------
    >>> import numpy as np
    >>> from scdesigner.covariance import OAS
    >>>
    >>> rng = np.random.default_rng(0)
    >>> estimator = OAS()
    >>> estimator.start(30)
    >>> estimator.update(rng.standard_normal((12, 30)))
    >>> _ = estimator.finalize()
    >>> np.linalg.cholesky(estimator.covariance_).shape  # positive definite
    (30, 30)
    """
    def _intensity(self, cov: np.ndarray) -> float:
        return oas_intensity(cov, self.n_samples_)


class LedoitWolf(_ShrinkageEstimator):
    """
    Ledoit-Wolf shrinkage, matching ``sklearn.covariance.LedoitWolf``.

    Must track three extra moment matrices beyond the sample covariance,
    so costs roughly 3X the memory of :class:`OAS` per group. `OAS` is hence
    less resource hungryat least in this data setting.

    Examples
    --------
    >>> import numpy as np
    >>> from scdesigner.covariance import LedoitWolf
    >>>
    >>> rng = np.random.default_rng(0)
    >>> estimator = LedoitWolf()
    >>> estimator.start(30)
    >>> estimator.update(rng.standard_normal((12, 30)))
    >>> _ = estimator.finalize()
    >>> estimator.diagnostics()["intensity"] > 0
    True
    """
    def _start(self, n_features: int) -> None:
        super()._start(n_features)
        self._squared_sum = np.zeros(n_features)
        self._third = np.zeros((n_features, n_features))
        self._fourth = np.zeros((n_features, n_features))

    def _update(self, z: np.ndarray) -> None:
        super()._update(z)
        z_squared = z**2
        self._squared_sum += z_squared.sum(axis=0)
        self._third += z_squared.T @ z
        self._fourth += z_squared.T @ z_squared

    def _intensity(self, cov: np.ndarray) -> float:
        return ledoit_wolf_intensity(
            cov,
            self.n_samples_,
            self.mean_,
            self._squared_sum,
            self._cross,
            self._third,
            self._fourth,
        )

    def release(self) -> None:
        super().release()
        self._third = None
        self._fourth = None


register_estimator("oas", OAS)
register_estimator("ledoit_wolf", LedoitWolf)
