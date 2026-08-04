"""
Factory for making (regularized) covariance estimators.

A Gaussian copula estimates one covariance per group of cells, these may have
fewer cells than genes if appropriately regularized. Different covariance
estimators can be substituted as estimator classes, leading to different copula
types .

Estimation streams. We make a single pass through the loader, using
:meth:`CovarianceEstimator.update` on each (batch, group) combination. This is
the basic recipe.

    estimator = prototype.clone()      # one per group
    estimator.start(n_features)
    estimator.update(z)                # once per batch
    estimator.finalize()               # sets covariance_
    estimator.release()                # drop accumulators before pickling

Examples
--------
>>> import numpy as np
>>> from scdesigner.covariance import LedoitWolf
>>>
>>> rng = np.random.default_rng(0)
>>> z = rng.standard_normal((40, 8))
>>> estimator = LedoitWolf()
>>> estimator.start(8)
>>> for start in range(0, 40, 16):
...     estimator.update(z[start:start + 16])
>>> _ = estimator.finalize()
>>> estimator.covariance_.shape
(8, 8)
>>> 0.0 <= estimator.intensity_ <= 1.0
True
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import inspect
import numpy as np


class CovarianceEstimator(ABC):
    """
    Streaming covariance estimator for a copula group.

    Attributes
    ----------
    regularized : bool
        Class attribute. ``True`` when :attr:`covariance_` is positive definite
        by construction; the copula then treats a failed factorization as a
        numerical error. An estimator that can return a singular estimate must
        leave this ``False``.
    n_features_ : int
        Number of features, set by :meth:`start`.
    n_samples_ : int
        Number of observations seen so far, modified by :meth:`update`.
    covariance_ : np.ndarray or None
        Estimated covariance of shape ``(n_features_, n_features_)``, set by
        :meth:`finalize`.
    """

    regularized = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def _param_names(cls):
        """
        Constructor parameter names, used by :meth:`get_params` and :meth:`clone`.
        """
        if cls.__init__ is object.__init__:
            return []

        parameters = inspect.signature(cls.__init__).parameters
        variadic = [
            name
            for name, p in parameters.items()
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
        if variadic:
            raise RuntimeError(
                f"{cls.__name__}.__init__ expects arguments {', '.join(variadic)}."
            )
        return [name for name in parameters if name != "self"]

    def get_params(self) -> Dict:
        """Return the constructor arguments of this estimator."""
        return {name: getattr(self, name) for name in self._param_names()}

    def clone(self) -> "CovarianceEstimator":
        """
        Create a new copy of the estimator with the same hyperparameters.
        """
        return type(self)(**self.get_params())

    def __repr__(self):
        arguments = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({arguments})"

    # ------------------------------------------------------------------
    # Stream across samples
    # ------------------------------------------------------------------
    def start(self, n_features: int) -> None:
        """
        Prepare to stream covariance over ``n_features`` features.

        Parameters
        ----------
        n_features : int
            Number of genes the copula models for this group.
        """
        self.n_features_ = int(n_features)
        self.n_samples_ = 0
        self.covariance_ = None
        self._start(self.n_features_)

    def update(self, z: np.ndarray) -> None:
        """
        Substitute a new block of z-scores into the streaming estimate

        Parameters
        ----------
        z : np.ndarray
            Array of shape ``(n_batch, n_features_)`` holding the latent normal
            scores for the cells of this group within one batch.
        """
        self.n_samples_ += z.shape[0]
        self._update(z)

    def finalize(self) -> "CovarianceEstimator":
        """
        Compute :attr:`covariance_` from the accumulators.

        Returns
        -------
        CovarianceEstimator
            ``self``, so calls can be chained.
        """
        if self.n_samples_ > 0:
            self.covariance_ = self._finalize()
        return self

    # ------------------------------------------------------------------
    # required steps for streaming
    # ------------------------------------------------------------------
    @abstractmethod
    def _start(self, n_features: int) -> None:
        """Allocate accumulators for ``n_features`` features."""

    @abstractmethod
    def _update(self, z: np.ndarray) -> None:
        """Fold one ``(n_batch, n_features)`` block into the accumulators."""

    @abstractmethod
    def _finalize(self) -> np.ndarray:
        """Return the estimated ``(n_features, n_features)`` covariance."""

    # ------------------------------------------------------------------
    # optional helpers
    # ------------------------------------------------------------------
    @property
    def second_moment_(self) -> Optional[np.ndarray]:
        """
        Uncentered second moment ``Z.T @ Z``, or ``None`` if not retained.
        """
        return None

    def num_params(self, n_features: int) -> int:
        """
        Free parameters in the estimate, used for AIC and BIC.

        The default counts every off-diagonal entry. This is actually an upper
        bound for regularized estimators, so those should implement their own
        parameter functions.
        """
        return n_features * (n_features - 1) // 2

    def diagnostics(self) -> Dict:
        """Any extra, method-specific diagnostics"""
        return {}

    def release(self) -> None:
        """
        Drop accumulators no longer needed after :meth:`finalize`.

        The copula calls this once it has consumed :attr:`second_moment_`.  This
        allows us to free up memory for large class definitions.
        """


_ESTIMATORS = {}


def register_estimator(name: str, estimator_class: type) -> None:
    """
    Make an estimator searchable by name.

    This allows us to search for estimators in a YAML config file.

    Parameters
    ----------
    name : str
        Lowercase name, as accepted by :func:`as_covariance_estimator`.
    estimator_class : type
        A :class:`CovarianceEstimator` subclass constructible with no arguments.
    """
    _ESTIMATORS[name] = estimator_class


def as_covariance_estimator(covariance) -> CovarianceEstimator:
    """
    Convert a covariance string to an estimator object

    Parameters
    ----------
    covariance : CovarianceEstimator, str, float, or None
        This can be an existing estimator, a registered name
        e.g.``"ledoit_wolf"`, or ``None`` for the unregularized sample
        covariance.

    Returns
    -------
    CovarianceEstimator
        The estimator to use as a prototype.

    Raises
    ------
    ValueError
        If the name is not registered.

    Examples
    --------
    >>> from scdesigner.covariance import as_covariance_estimator
    >>> as_covariance_estimator("oas")
    OAS()
    >>> as_covariance_estimator(None)
    SampleCovariance()
    """
    if isinstance(covariance, CovarianceEstimator):
        return covariance
    if covariance is None:
        return _ESTIMATORS["sample"]()
    if isinstance(covariance, str):
        name = covariance.lower()
        if name not in _ESTIMATORS:
            raise ValueError(
                f"Unknown covariance estimator {covariance!r}. Expected one of "
                f"{sorted(_ESTIMATORS)}, a float in [0, 1], a CovarianceEstimator "
                "instance, or None."
            )
        return _ESTIMATORS[name]()
    return _ESTIMATORS["fixed"](float(covariance))
