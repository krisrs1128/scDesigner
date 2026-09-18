"""
Streaming covariance estimators using batches of normal scores.

The estimators here depend only on the first two moments, and these can be
streamed batchwise.  :class:`_MomentEstimator` accumulates the three summaries,
the count, per-feature sum, and the uncentered cross-product ``Z.T @ Z``. It
then centers these at the end of the stream, once the mean is known.
"""

import numpy as np
from .base import CovarianceEstimator, register_estimator


class _MomentEstimator(CovarianceEstimator):
    """
    Shared base for estimators that need only ``n``, ``sum z`` and ``Z.T @ Z``.

    Subclasses implement :meth:`_finalize` and may add updates of their own by
    through methods :meth:`_start` and :meth:`_update`.
    """

    def _start(self, n_features: int) -> None:
        self._sum = np.zeros(n_features)
        self._cross = np.zeros((n_features, n_features))

    def _update(self, z: np.ndarray) -> None:
        self._sum += z.sum(axis=0)
        self._cross += z.T @ z

    @property
    def second_moment_(self):
        return self._cross

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature mean of the observations seen so far."""
        return self._sum / self.n_samples_

    def _biased_covariance(self) -> np.ndarray:
        """Centered covariance estimator"""
        mean = self.mean_
        return self._cross / self.n_samples_ - np.outer(mean, mean)

    def release(self) -> None:
        self._cross = None


class SampleCovariance(_MomentEstimator):
    """
    Unregularized sample covariance.

    Note that ``top_k`` must be smaller than the smallest within-group sample
    size.  :class:`~scdesigner.covariance.LedoitWolf` and
    :class:`~scdesigner.covariance.OAS` can be applied in the more general
    setting.

    Examples
    --------
    >>> import numpy as np
    >>> from scdesigner.covariance import SampleCovariance
    >>>
    >>> rng = np.random.default_rng(0)
    >>> estimator = SampleCovariance()
    >>> estimator.start(4)
    >>> estimator.update(rng.standard_normal((50, 4)))
    >>> estimator.finalize().covariance_.shape
    (4, 4)
    """

    def _finalize(self) -> np.ndarray:
        return self._biased_covariance()


register_estimator("sample", SampleCovariance)
register_estimator("none", SampleCovariance)
