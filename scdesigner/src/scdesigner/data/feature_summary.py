"""Per-feature summaries of an AnnData response matrix."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse
from anndata import AnnData


@dataclass(frozen=True)
class FeatureSummary:
    """Feature means and (optional) maxima of the response.

    Parameters
    ----------
    mean : ndarray of shape (n_features,)
    maximum : ndarray of shape (n_features,), optional
    """
    mean: np.ndarray
    maximum: Optional[np.ndarray]


def summarize_features(adata: AnnData, *, include_max: bool = False) -> FeatureSummary:
    """:class:`FeatureSummary` of ``adata.X`` across rows.

    Parameters
    ----------
    adata : AnnData
        In-memory dataset.
    include_max : bool
        Whether to compute ``maximum``. Otherwise returns None.
    """
    if adata.isbacked:
        raise NotImplementedError("Feature summaries require in-memory AnnData.")

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    maximum = X.max(axis=0).astype(np.float64) if include_max else None
    return FeatureSummary(mean=X.mean(axis=0, dtype=np.float64), maximum=maximum)
