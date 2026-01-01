"""Data loading utilities for scDesigner models.

The core entry point is :func:`adata_loader`, which builds a PyTorch
:class:`~torch.utils.data.DataLoader` that yields mini-batches of:

- **X**: expression/count matrix rows (cells × genes), returned as a float tensor
- **obs**: a dict mapping formula keys to design-matrix tensors produced from
  ``adata.obs`` via :func:`formulaic.model_matrix`

This module supports both in-memory and backed :class:`~anndata.AnnData`
objects. For backed AnnData, a chunk cache is used to avoid loading all rows
into memory at once.
"""

from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from anndata import AnnData
from formulaic import model_matrix
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
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

class AnnDataDataset(Dataset):
    """PyTorch dataset over an AnnData object, with optional chunk caching.

    This dataset exposes AnnData rows as samples and returns a tuple
    ``(X_row, obs_dict)`` where ``obs_dict`` contains (one-row) design matrices
    computed from ``adata.obs`` according to the provided ``formula`` mapping.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    formula : dict[str, str]
        Mapping from key to a formula string used to build a design matrix from
        ``adata.obs`` via :func:`formulaic.model_matrix`.
    chunk_size : int
        Number of contiguous rows to cache at once (used when chunking is enabled).

    Attributes
    ----------
    predictor_names : dict[str, list[str]] or None
        Names of the design-matrix columns per formula key. Populated after the
        first chunk is loaded.
    device : torch.device
        Device used for caching tensors.
    """
    def __init__(self, adata: AnnData, formula: Dict[str, str], chunk_size: int,
                 device: Optional[torch.device] = None):
        self.adata = adata
        self.formula = formula
        self.chunk_size = chunk_size
        self.device = get_device(device)

        # keeping track of covariate-related information
        self.obs_levels = categories(self.adata.obs)
        self.obs_matrices = {}
        self.predictor_names = None

        # Internal cache for the currently loaded chunk
        self._chunk: AnnData | None = None
        self._chunk_X = None
        self._chunk_start = 0

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        """Returns (X, obs) for the given index.

        If `chunk_size` was specified the dataset will load a chunk
        containing `idx` into memory (if not already cached) and
        index into that chunk.
        """
        self._ensure_chunk_loaded(idx)
        local_idx = idx - self._chunk_start

        # Get obs data from GPU-cached matrices
        obs_dict = {}
        for key in self.formula.keys():
            obs_dict[key] = self.obs_matrices[key][local_idx]
        return self._chunk_X[local_idx], obs_dict

    def _ensure_chunk_loaded(self, idx: int) -> None:
        """Load the chunk that contains `idx` into the internal cache."""
        start = (idx // self.chunk_size) * self.chunk_size
        end = min(start + self.chunk_size, len(self.adata))

        if (self._chunk is None) or not (self._chunk_start <= idx < self._chunk_start + len(self._chunk)):
            # load the next chunk into memory
            chunk = self.adata[start:end]
            if getattr(chunk, 'isbacked', False):
                chunk = chunk.to_memory()
            self._chunk = chunk
            self._chunk_start = start

            # Move chunk to GPU
            X = chunk.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            self._chunk_X = torch.tensor(X, dtype=torch.float32).to(self.device)

            # Compute model matrices for this chunk's `obs` and move to GPU
            obs_coded_chunk = code_levels(self._chunk.obs.copy(), self.obs_levels)
            self.obs_matrices = {}
            predictor_names = {}
            for key, f in self.formula.items():
                mat = model_matrix(f, obs_coded_chunk)
                predictor_names [key] = list(mat.columns)
                self.obs_matrices[key] = torch.tensor(mat.values, dtype=torch.float32).to(self.device)

            # Capture predictor (column) names from the model matrices once.
            if self.predictor_names is None:
                self.predictor_names = predictor_names


def adata_loader(
    adata: AnnData,
    formula: Dict[str, str],
    chunk_size: int = None,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
    device=None,
    **kwargs
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` over an AnnData dataset.

    The resulting loader yields tuples ``(X_batch, obs_batch)`` where:

    - ``X_batch`` is a float tensor of shape (batch_size, n_genes)
    - ``obs_batch`` is a dict mapping each key in ``formula`` to a float tensor
      of shape (batch_size, n_covariates) for that key's design matrix

    For backed AnnData objects, the underlying dataset uses chunk caching to
    limit memory usage.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    formula : dict[str, str]
        Mapping from key to a formula string consumed by
        :func:`formulaic.model_matrix` (applied to ``adata.obs``).
    chunk_size : int, optional
        Chunk size used only for backed AnnData. If omitted, defaults to 5000.
    batch_size : int, optional
        Mini-batch size returned by the loader.
    shuffle : bool, optional
        Whether to shuffle observations each epoch.
    num_workers : int, optional
        Number of DataLoader workers.
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

    # separate chunked from non-chunked cases
    if not getattr(adata, 'isbacked', False):
        dataset = _preloaded_adata(adata, formula, device)
    else:
        dataset = AnnDataDataset(adata, formula, chunk_size or 5000, device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        **data_kwargs
    )

def obs_loader(obs: pd.DataFrame, marginal_formula, **kwargs):
    """Create a loader that yields design matrices for an observation table.

    This is a convenience wrapper for prediction-time batching: it creates a
    dummy AnnData with a placeholder ``X`` and uses :func:`adata_loader` to
    construct batches of covariates derived from ``obs``.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation metadata used to build design matrices.
    marginal_formula : dict[str, str]
        Formula mapping used for :func:`formulaic.model_matrix`.
    **kwargs
        Forwarded to :func:`adata_loader` (e.g. ``batch_size``, device options).

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding ``(X_batch, obs_batch)`` where ``X_batch`` is dummy.
    """
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    return adata_loader(
        adata,
        marginal_formula,
        **kwargs
    )

################################################################################
## Extraction of in-memory AnnData to PreloadedDataset
################################################################################

def _preloaded_adata(adata: AnnData, formula: Dict[str, str], device: torch.device) -> PreloadedDataset:
    """Materialize an in-memory AnnData into a :class:`PreloadedDataset`.

    This helper converts sparse matrices to dense, encodes categorical levels
    to ensure consistent model-matrix columns, builds the per-key design
    matrices, and moves everything to ``device``.
    """
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    y = torch.tensor(X, dtype=torch.float32).to(device)

    obs = code_levels(adata.obs.copy(), categories(adata.obs))
    x = {
        k: torch.tensor(model_matrix(f, obs).values, dtype=torch.float32).to(device)
        for k, f in formula.items()
    }
    predictor_names = {k: list(model_matrix(f, obs).columns) for k, f in formula.items()}
    return PreloadedDataset(y, x, predictor_names)

################################################################################
## Helper functions
################################################################################

def dict_collate_fn(batch):
    """
    Custom collate function for handling dictionary obs tensors.
    """
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

def categories(obs):
    """Collect levels for categorical/object columns in an observation table."""
    levels = {}
    for k in obs.columns:
        obs_type = str(obs[k].dtype)
        if obs_type in ["category", "object"]:
            levels[k] = obs[k].unique()
    return levels


def code_levels(obs, categories):
    """Cast categorical columns to a fixed set of categories.

    This ensures stable design-matrix columns across different chunks/batches.
    """
    for k in obs.columns:
        if str(obs[k].dtype) == "category":
            obs[k] = obs[k].astype(pd.CategoricalDtype(categories[k]))
    return obs

###############################################################################
## Misc. Helper functions
###############################################################################

def _to_numpy(*tensors):
    """Convenience helper: detach, move to CPU, and convert tensors to numpy arrays."""
    return tuple(t.detach().cpu().numpy() for t in tensors)
