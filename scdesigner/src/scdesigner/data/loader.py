"""Data loading utilities for scDesigner models.

The main function is `:func:`adata_loader`, which builds a
:class:`~torch.utils.data.DataLoader` that returns minibatches of,

- **X**: expression matrix rows (cells × genes) as a float tensor
- **obs**: a dict with formula keys and design matrices from a fitted
  :class:`~scdesigner.data.design.Design`

We support both in-memory and backed :class:`~anndata.AnnData` objects. For
backed anndata, we read ``chunk_size`` rows at a time. See :func:`adata_loader`.
"""

from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from .design import Design, fit_design
from .response import DEFAULT_CHUNK_SIZE, read_response_rows
from anndata import AnnData
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Set, Union
import numpy as np
import pandas as pd
import scipy.sparse
import torch

def get_device(device=None):
    """Detect and return the best available device (MPS, CUDA, or CPU).
    """
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class PreloadedDataset(Dataset):
    """Dataset wrapper where both response and predictors are preloaded.

    This dataset is used for in-memory AnnData objects where we can materialize:

    - ``y``: the full expression matrix on the target device
    - ``x``: a dict of design matrices (one per formula key) on the same device

    Parameters
    ----------
    y_tensor : torch.Tensor
        Expression tensor of shape (n_obs, n_features).
    x_tensors : dict[str, torch.Tensor]
        Mapping from formula key to design matrix tensor of shape (n_obs, p_k).
    predictor_names : dict[str, list[str]]
        Mapping from formula key to the names of the design matrix columns.
    """
    def __init__(
        self,
        y_tensor: torch.Tensor,
        x_tensors: Dict[str, torch.Tensor],
        predictor_names):
        self.y = y_tensor
        self.x = x_tensors
        self.predictor_names = predictor_names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], {k: v[idx] for k, v in self.x.items()}

    def __getitems__(self, indices):
        """Faster alternative to __getitem__

        On MPS, using ordinary __getitem__ is slow, because it requires us to
        read batch_size different slices and then stack. We can bypass that
        expensive read/concat by using __getitems__. See
	`examples/validation/loader_benchmark.ipynb` for a precise comparison.
        """
        idx = torch.as_tensor(indices, device=self.y.device)
        return self.y[idx], {k: v[idx] for k, v in self.x.items()}

    def to(self, device):
        """Move preloaded responses and design matrices to ``device``."""
        device = get_device(device)
        self.y = self.y.to(device)
        self.x = {key: value.to(device) for key, value in self.x.items()}
        return self


class AnnDataDataset(Dataset):
    """PyTorch dataset over AnnData rows, reading one chunk of rows at a time.

    Returns ``(X_row, obs_dict)`` as items, where ``obs_dict`` has one
    design-matrix row per key in the ``design``. Responses Y come from
    :func:`~scdesigner.data.response.read_response_rows` and designs run
    ``design`` to create X on the corresponding rows.

    Parameters
    ----------
    adata : AnnData
        In memory or backed.
    design : Design
        Fitted on the complete observation table.
    chunk_size : int
        Number of contiguous rows to cache at once.

    Attributes
    ----------
    predictor_names : dict[str, list[str]]
        Names of the design-matrix columns per key.
    device : torch.device
        Device used for caching tensors.
    """
    def __init__(self, adata: AnnData, design: Design, chunk_size: int,
                 device: Optional[torch.device] = None):
        self.adata = adata
        self.design = design
        self.chunk_size = chunk_size
        self.device = get_device(device)
        self.predictor_names = design.predictor_names
        self.obs_matrices = {}

        # Rows [_chunk_start, _chunk_stop) are cached.
        self._chunk_X = None
        self._chunk_start = self._chunk_stop = 0

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        """Returns ``(X_row, obs_dict)`` for the given index."""
        self._ensure_chunk_loaded(idx)
        local_idx = idx - self._chunk_start
        obs_dict = {key: matrix[local_idx] for key, matrix in self.obs_matrices.items()}
        return self._chunk_X[local_idx], obs_dict

    def __getitems__(self, indices):
        """Fetch a minibatch from an existing chunk."""
        indices = np.asarray(indices, dtype=np.int64)
        breaks = np.flatnonzero(np.diff(indices // self.chunk_size)) + 1
        responses = []

        # create designs for each parameter
        designs = {key: [] for key in self.predictor_names}
        for group in np.split(indices, breaks):
            self._ensure_chunk_loaded(int(group[0]))
            local = torch.as_tensor(group - self._chunk_start, device=self.device)
            responses.append(self._chunk_X.index_select(0, local))
            for key, matrix in self.obs_matrices.items():
                designs[key].append(matrix.index_select(0, local))

        if len(responses) == 1:
            return responses[0], {key: parts[0] for key, parts in designs.items()}

        return (
            torch.cat(responses, dim=0),
            {key: torch.cat(parts, dim=0) for key, parts in designs.items()},
        )

    def _ensure_chunk_loaded(self, idx: int) -> None:
        """Cache the chunk that contains ``idx``."""
        if self._chunk_start <= idx < self._chunk_stop:
            return
        start = (idx // self.chunk_size) * self.chunk_size
        stop = min(start + self.chunk_size, len(self))

        X = read_response_rows(self.adata, start, stop)
        self._chunk_X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.obs_matrices = {
            key: torch.tensor(mat.values, dtype=torch.float32).to(self.device)
            for key, mat in self.design.matrices(self.adata.obs.iloc[start:stop]).items()
        }
        self._chunk_start, self._chunk_stop = start, stop

    def to(self, device):
        """Move cached chunk data and use ``device`` for future chunks."""
        self.device = get_device(device)
        if self._chunk_X is not None:
            self._chunk_X = self._chunk_X.to(self.device)
        self.obs_matrices = {
            key: value.to(self.device) for key, value in self.obs_matrices.items()
        }
        return self


def resolve_chunk_size(adata: AnnData, chunk_size: Optional[int]) -> Optional[int]:
    """How many rows per chunk? Returns ``None`` for in-memory data."""
    if chunk_size is None and not adata.isbacked:
        return None
    return DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size


def adata_loader(
    adata: AnnData,
    formula: Union[Dict[str, str], Design],
    chunk_size: int = None,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
    device=None,
    partition_keys: Optional[Set[str]] = None,
    **kwargs
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` over an AnnData dataset.

    The resulting loader yields tuples ``(X_batch, obs_batch)`` where:

    - ``X_batch`` is a float tensor of shape (batch_size, n_genes)
    - ``obs_batch`` is a dict mapping each formula key to a float tensor
      of shape (batch_size, n_covariates) for that key's design matrix

    For backed AnnData objects, the underlying dataset uses chunk caching to
    limit memory usage. Chunks are read in the order stored on disk.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    formula : dict[str, str] or Design
        A fitted :class:`~scdesigner.data.design.Design`, or associated formulas,
        fitted on ``adata.obs`` with :func:`~scdesigner.data.design.fit_design`.
    chunk_size : int, optional
        Rows per chunked read. ``None`` preloads in-memory data and reads
        backed data in ``DEFAULT_CHUNK_SIZE``-row chunks.
    batch_size : int, optional
        Mini-batch size returned by the loader.
    shuffle : bool, optional
        Whether to shuffle observations each epoch. Not allowed for backed
        reads.
    num_workers : int, optional
        Number of DataLoader workers.
    partition_keys : set of str, optional
        Formula used to partition observations, usually used for copula
        covariance groups. Ignored when ``formula`` is a Design. See
        :func:`~scdesigner.data.formula.partition_formula`.
    **kwargs
        Additional keyword arguments filtered via ``DEFAULT_ALLOWED_KWARGS["data"]``
        and forwarded to :class:`~torch.utils.data.DataLoader`.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader over the dataset.
    """
    data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['data'])
    device = get_device(device)
    chunk_size = resolve_chunk_size(adata, chunk_size)
    design = formula if isinstance(formula, Design) else fit_design(
        adata.obs, formula, partition_keys=partition_keys or set(),
        chunked=chunk_size is not None,
    )

    if chunk_size is None:
        dataset = _preloaded_adata(adata, design, device)
    else:
        if shuffle:
            raise ValueError(
                """Chunked loading does not support shuffle=True. We read chunks
                consecutively and train in that order. If the data have sorted
                rows, we recommend an initial permutation step."""
            )
        dataset = AnnDataDataset(adata, design, chunk_size, device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        **data_kwargs
    )

def obs_loader(obs: pd.DataFrame, formula: Union[Dict[str, str], Design], **kwargs):
    """Create a loader that yields design matrices for an observation table.

    This is a convenience wrapper for prediction-time batching: it creates a
    dummy AnnData with a placeholder ``X`` and uses :func:`adata_loader` to
    construct batches of covariates derived from ``obs``.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation metadata used to build design matrices.
    formula : dict[str, str] or Design
        See :func:`adata_loader`. Using a Design from the training data ensures
        that the same state (e.g., spline knots) is used across batches.
    **kwargs
        Forwarded to :func:`adata_loader` (e.g. ``batch_size``, device options).

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding ``(X_batch, obs_batch)`` where ``X_batch`` is dummy.
    """
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    return adata_loader(adata, formula, **kwargs)

################################################################################
## Extraction of in-memory AnnData to PreloadedDataset
################################################################################

def _preloaded_adata(adata: AnnData, design: Design, device: torch.device) -> PreloadedDataset:
    """Materialize an in-memory AnnData into a :class:`PreloadedDataset`.

    Evaluate a design on an ``adata.obs`` object.
    """
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    y = torch.tensor(X, dtype=torch.float32).to(device)
    x = {
        key: torch.tensor(mat.values, dtype=torch.float32).to(device)
        for key, mat in design.matrices(adata.obs).items()
    }
    return PreloadedDataset(y, x, design.predictor_names)

################################################################################
## Helper functions
################################################################################

def dict_collate_fn(batch):
    """Custom collate function for handling dictionary obs tensors.
    """
    if isinstance(batch, tuple):
        return batch

    X_batch = torch.stack([item[0] for item in batch])
    obs_batch = [item[1] for item in batch]

    obs_dict = {}
    for key in obs_batch[0].keys():
        obs_dict[key] = torch.stack([obs[key] for obs in obs_batch])
    return X_batch, obs_dict

def to_tensor(X):
    """Convert ``X`` to a float tensor with conservative squeezing.
    """
    # If the tensor is 2D with second dim == 1, squeeze only the first
    # dim when appropriate (e.g. converting a single-row X to 1D samples)
    t = torch.tensor(X, dtype=torch.float32)
    if t.dim() == 2 and t.size(1) == 1:
        if t.size(0) == 1:
            return t.view(1)
        return t
    return t.squeeze()

###############################################################################
## Misc. Helper functions
###############################################################################

def _to_numpy(*tensors):
    """Convenience helper: detach, move to CPU, and convert tensors to numpy arrays."""
    return tuple(t.detach().cpu().numpy() for t in tensors)
