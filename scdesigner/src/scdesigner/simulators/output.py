"""Transforms applied to simulated counts after sampling."""

from typing import Optional
import numpy as np
from anndata import AnnData
from ..data.feature_summary import summarize_features


class ObservedMaxCap:
    """Caps features at the maximum observed during training

    Attributes
    ----------
    maximum : ndarray of shape (n_features,) or None
        Per-feature maximum from the last successful :meth:`fit`; ``None``
        before the first.
    """

    def __init__(self):
        self.maximum: Optional[np.ndarray] = None

    def fit(self, adata: AnnData) -> None:
        """Store the per-feature maximum of ``adata.X``.

        Parameters
        ----------
        adata : AnnData
            In-memory dataset.
        """
        self.maximum = summarize_features(adata, include_max=True).maximum

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """New array of ``samples`` capped at ``maximum``.

        Parameters
        ----------
        samples : ndarray of shape (n_obs, n_features)
        """
        if self.maximum is None:
            raise RuntimeError("ObservedMaxCap must be fit before transform.")
        return np.minimum(samples, self.maximum)
