"""Per-feature summaries of an AnnData response matrix."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from anndata import AnnData
from .response import iter_response_chunks


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


def summarize_features(
    adata: AnnData, *, include_max: bool = False, chunk_size: Optional[int] = None
) -> FeatureSummary:
    """:class:`FeatureSummary` of ``adata.X`

    This computes the mean over columns, one chunk at a time, so that it scales
    to large backed anndata objects.

    Parameters
    ----------
    adata : AnnData
        In memory or backed.
    include_max : bool
        Whether to compute ``maximum``. Otherwise it is None.
    chunk_size : int, optional
        Rows per read; see :func:`~scdesigner.data.response.iter_response_chunks`.
    """
    total = np.zeros(adata.n_vars)
    maximum = np.full(adata.n_vars, -np.inf) if include_max else None
    n_rows = 0
    for _, _, values in iter_response_chunks(adata, chunk_size):
        total += values.sum(axis=0, dtype=np.float64)
        n_rows += len(values)
        if include_max:
            np.maximum(maximum, values.max(axis=0), out=maximum)
    return FeatureSummary(mean=total / n_rows, maximum=maximum)
