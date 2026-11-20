"""Partial (chunked) reads of backed anndata matrices ``X``

This reads groups of cells at a time in a backend object. It only supports CSR
format, since cell-level reading is much more efficient in row-oriented tables.

Note subsequent training is done using batches loaded in row order (shuffling is
not supported during training).  Therefore, if cells were collected in a
meaningful sample order, we recommend an initial permutation step.

    adata = anndata.read_h5ad(path)
    adata = adata[np.random.permutation(adata.n_obs)].copy()
"""

from typing import Iterator, Optional, Tuple

import h5py
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset

DEFAULT_CHUNK_SIZE = 20000

BACKED_CSC_MESSAGE = """We only support CSR or dense matrices for chunked reads, not Backed CSC.
    To convert the data to this format, you may use

    adata = anndata.read_h5ad(path)
    adata.X = adata.X.tocsr()
    adata.write_h5ad(csr_path)
    """
class ResponseReader:
    """Support efficient reading of blocks of rows from ``adata.X``

    This reader uses the structure of CSR matrices to support efficient
    iteration across backed anndata objects. For example, we make sure to reduce
    the number of reads from the backed sparse matrix by first sorting row
    indices, reading the continguous blocks, and reordering indices to match the
    request. By reducing the number of reads, we improve runtime.

    Parameters
    ----------
    adata : AnnData
        In memory or backed dataset/view. For the backed case, only CSR and
        dense formats are supported.
    """

    def __init__(self, adata: AnnData):
        source, rows, genes = adata, np.arange(adata.n_obs), None
        if adata.is_view:
            source = adata._adata_ref
            rows = np.arange(source.n_obs)[adata._oidx]
            genes = adata._vidx

        # Check if valid foramt
        self.X = source.X
        if isinstance(self.X, CSCDataset):
            raise ValueError(BACKED_CSC_MESSAGE)

        # Relevant attributes about this particular anndata.X object
        self.rows = rows
        self.increasing = bool(np.all(np.diff(rows) > 0))
        self.genes = genes
        self.sparse = sp.issparse(self.X) or isinstance(self.X, CSRDataset)
        self.read_rows = self.read_memory

        # Decide what type of reading we'll use.
        if isinstance(self.X, h5py.Dataset):
            self.read_rows = self.read_dense
        if isinstance(self.X, CSRDataset):
            self.read_rows = self.read_csr
            group = self.X.group

            # Avoid repeated reads of the same chunk by keeping the handle open.
            self.data = group["data"]
            self.indices = group["indices"]
            self.indptr = group["indptr"][:]

    def read(self, start: int, stop: int, dtype=None) -> np.ndarray:
        """Read rows ``[start:stop]`` and return a dense numpy array.

        If the indices are already sorted or require no subsetting, then we can
        bypass costly copies. This is the reason for the if/else checks below.
        """
        rows = self.rows[start:stop]
        if self.increasing:
            values = self.read_rows(rows)
        else:
            stored, order = np.unique(rows, return_inverse=True)
            values = self.read_rows(stored)[order]

        if self.genes is not None:
            values = values[:, self.genes]
        return self.densify(values, dtype)

    def read_memory(self, rows: np.ndarray):
        return self.X[rows]

    def read_dense(self, rows: np.ndarray) -> np.ndarray:
        """Read a dense block of rows

        Replaces two reads of adjacent blocks using one merged read. The check
        for whether blocks can be read once is done with ``continguous_runs``.
        """
        blocks = [self.X[lo:hi] for lo, hi in contiguous_runs(rows)]
        return join_blocks(blocks)

    def read_csr(self, rows: np.ndarray) -> sp.csr_matrix:
        """Read back to CSR rows.

        This uses a similar trick as read_dense to minimize the number of reads,
        but it keeps the returned data in sparse matrix format.
        """
        spans = [slice(self.indptr[lo], self.indptr[hi]) for lo, hi in contiguous_runs(rows)]
        data = join_blocks([self.data[span] for span in spans])
        indices = join_blocks([self.indices[span] for span in spans])
        counts = self.indptr[rows + 1] - self.indptr[rows]
        indptr = np.concatenate([[0], np.cumsum(counts)])
        return sp.csr_matrix((data, indices, indptr), shape=(len(rows), self.X.shape[1]))

    def densify(self, values, dtype=None) -> np.ndarray:
        """Return a dense version of a sparse matrix.

        It's important to first cast the sparse matrix before densifying.  This
        approach costs O(number of nonzeros) instead of O(rows x genes).
        """
        if not self.sparse:
            return np.ascontiguousarray(values, dtype=dtype)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.toarray(order="C")


def iter_response_chunks(
    adata: AnnData, chunk_size: Optional[int] = None
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate through ``(start, stop, rows)`` indices for chunks of ``adata.X``.

    This generator provides indexing information and the actual materialized
    chunks associated with a backed anndata object. If chunk size is not
    provided, defaults to 20000.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    reader = ResponseReader(adata)
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        yield start, stop, reader.read(start, stop)


def contiguous_runs(positions: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Find contiguous blocks of indices to simplify reads

    We have this helper because it's more efficient to read backed anndata
    objects in contiguous blocks of rows.

    # input:
    # contiguous_runs(np.array([0, 1, 2, 5, 7, 8]))
    # yields:
    # (0, 3), (5, 6), (7, 9)
    """
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    for run in np.split(positions, breaks):
        if len(run):
            yield run[0], run[-1] + 1


def join_blocks(blocks: list) -> np.ndarray:
    """``np.concatenate`` that avoids unnecessary copies."""
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks)
