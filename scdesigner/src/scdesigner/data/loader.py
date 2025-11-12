from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from anndata import AnnData
from formulaic import model_matrix
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import numpy as np
import pandas as pd
import scipy.sparse
import torch

def get_device():
    """Detect and return the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class PreloadedDataset(Dataset):
    """Dataset that assumes x and y are both fully in memory."""
    def __init__(self, y_tensor, x_tensors, predictor_names):
        self.y = y_tensor
        self.x = x_tensors
        self.predictor_names = predictor_names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], {k: v[idx] for k, v in self.x.items()}

class AnnDataDataset(Dataset):
    """Simple PyTorch Dataset for AnnData objects.

    Supports optional chunked loading for backed AnnData objects. When
    `chunk_size` is provided, the dataset will load contiguous slices
    of rows (of size `chunk_size`) into memory once and serve individual
    rows from that cached chunk. Chunks are moved to device for faster access.
    """
    def __init__(self, adata: AnnData, formula: Dict[str, str], chunk_size: int):
        self.adata = adata
        self.formula = formula
        self.chunk_size = chunk_size
        self.device = get_device()

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
            obs_dict[key] = self.obs_matrices[key][local_idx: local_idx + 1]
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
    **kwargs
) -> DataLoader:
    """Create a DataLoader from AnnData that returns batches of (X, obs)."""
    data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['data'])
    device = get_device()

    # separate chunked from non-chunked cases
    if not getattr(adata, 'isbacked', False):
        dataset = _preloaded_adata(adata, formula, device)
    else:
        dataset = AnnDataDataset(adata, formula, chunk_size or 5000)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        **data_kwargs
    )

def obs_loader(obs: pd.DataFrame, marginal_formula, **kwargs):
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
    # If the tensor is 2D with second dim == 1, squeeze only the first
    # dim when appropriate (e.g. converting a single-row X to 1D samples)
    t = torch.tensor(X, dtype=torch.float32)
    if t.dim() == 2 and t.size(1) == 1:
        if t.size(0) == 1:
            return t.view(1)
        return t
    return t.squeeze()

def categories(obs):
    levels = {}
    for k in obs.columns:
        obs_type = str(obs[k].dtype)
        if obs_type in ["category", "object"]:
            levels[k] = obs[k].unique()
    return levels


def code_levels(obs, categories):
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
