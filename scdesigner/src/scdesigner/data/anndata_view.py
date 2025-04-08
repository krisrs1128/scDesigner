from anndata import AnnData
import torch
import pandas as pd
import numpy as np
from formulaic import model_matrix
import torch.utils.data as td


class AnndataViewLoader:
    def __init__(
        self,
        adata: AnnData,
        formula=None,
        chunk_size=int(1e4),
        device=None,
        batch_size: int = None,
    ):
        ds = AnndataViewDataset(adata, formula, chunk_size, device)
        self.loader = td.DataLoader(ds, batch_size=batch_size)


class AnndataViewDataset(td.Dataset):
    def __init__(self, view, formula=None, chunk_size=int(1e4), device=None):
        super().__init__()
        self.view = view
        self.formula = formula
        self.len = len(view)
        self.cur_range = range(0, min(self.len, chunk_size))
        self.categories = column_levels(view.obs)
        self.x = None
        self.y = None
        self.device = device or check_device()

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        if self.x is None or ix not in self.cur_range:
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            view_inmem = self.view[self.cur_range].to_memory()
            self.x = safe_model_matrix(
                view_inmem.obs, self.formula, self.categories
            ).to(self.device)
            self.y = torch.from_numpy(view_inmem.X.toarray().astype(np.float32)).to(
                self.device
            )
        return self.x[ix - self.cur_range[0]], self.y[ix - self.cur_range[0]]


def safe_model_matrix(obs, formula, categories):
    if formula is None:
        return obs

    for k in obs.columns:
        if str(obs[k].dtype) == "category":
            obs[k] = obs[k].astype(pd.CategoricalDtype(categories[k]))

    x = model_matrix(formula, obs)
    return torch.from_numpy(np.array(x).astype(np.float32))


def column_levels(obs):
    categories = {}
    for k in obs.columns:
        obs_type = str(obs[k].dtype)
        if obs_type in ["category", "object"]:
            categories[k] = obs[k].unique()
    return categories


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
