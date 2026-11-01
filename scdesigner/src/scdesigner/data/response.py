"""Partial (chunked) reads of backed anndata matrices ``X``

This reads groups of cells at a time in a backend object. It only supports CSR
format, since cell-level reading is much more efficient in row-oriented tables.
"""

from typing import Iterator, Optional, Tuple

import h5py
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset

DEFAULT_CHUNK_SIZE = 20000

BACKED_CSC_MESSAGE = (
    """We only support CSR or dense matrices for chunked reads, not Backed CSC.
    To convert the data to this format, you may use `adata.X = adata.X.tocsr()`
    and then `adata.write_h5ad(path)`. Note subsequent training is done using
    batches loaded in row order (shuffling is not supported during training).
    Therefore, if cells were collected in a meaningful sample order, we
    recommend an initial permutation step.
    adata = adata[np.random.permutation(adata.n_obs)].copy()"""
)


def read_response_rows(adata: AnnData, start: int, stop: int) -> np.ndarray:
    """Read rows ``[start:stop]`` of ``adata.X``

    This only applies to CSR-oriented data. We read only views and never
    materialize the entire anndata object.
    """
    # check if the data is CSR format
    source, rows, cols = _source_selection(adata, start, stop)
    X = source.X
    if isinstance(X, CSCDataset):
        raise ValueError(BACKED_CSC_MESSAGE)

    # read data in memory
    if not isinstance(X, (h5py.Dataset, CSRDataset)):
        values = X[rows]
        values = values.toarray() if sp.issparse(values) else np.asarray(values)
        return values[:, cols]

    # Sort and deduplicate for backed reads; retain the requested row order.
    unique_rows, row_order = np.unique(rows, return_inverse=True)
    reader = _read_dense if isinstance(X, h5py.Dataset) else _read_csr
    values = reader(X, unique_rows)
    return values[np.ix_(row_order, cols)]


def iter_response_chunks(
    adata: AnnData, chunk_size: Optional[int] = None
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """``(start, stop, read_response_rows(adata, start, stop))`` across chunks.

    This generator provides indexing information and the actual materialized
    chunks associated with a backed anndata object. If chunk size is not
    provided, defaults to 20000.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        yield start, stop, read_response_rows(adata, start, stop)


def _source_selection(adata: AnnData, start: int, stop: int):
    """Return anndata subset and associated indices."""
    if not adata.is_view:
        return adata, np.arange(start, stop), np.arange(adata.n_vars)
    source = adata._adata_ref
    rows = np.arange(source.n_obs)[adata._oidx][start:stop]
    return source, rows, np.arange(source.n_vars)[adata._vidx]


def _runs(positions: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Find the endpoints of contiguous blocks of indices.

    We have this helper because it's more efficient to read backed anndata
    objects in contiguous blocks of rows.

    # input:
    # _runs(np.array([0, 1, 2, 5, 7, 8]))
    # yields:
    # (0, 3), (5, 6), (7, 9)
    """
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    for run in np.split(positions, breaks):
        if len(run):
            yield run[0], run[-1] + 1


def _read_dense(X: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    return np.concatenate([X[lo:hi] for lo, hi in _runs(rows)])


def _read_csr(X: CSRDataset, rows: np.ndarray) -> np.ndarray:
    """Read selected rows from a backed CSR matrix and return a dense array. """
    group, blocks = X.group, []
    for lo, hi in _runs(rows):
        indptr = group["indptr"][lo:hi + 1]
        payload = slice(indptr[0], indptr[-1])
        block = sp.csr_matrix(
            (group["data"][payload], group["indices"][payload], indptr - indptr[0]),
            shape=(hi - lo, X.shape[1]),
        )
        blocks.append(block.toarray())
    return np.concatenate(blocks)
