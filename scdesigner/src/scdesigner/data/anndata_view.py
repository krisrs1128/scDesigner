from anndata import AnnData
from formulaic import model_matrix
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.utils.data as td


def formula_loader(adata: AnnData, formula=None, 
        chunk_size=int(1e4),
        batch_size: int=None
    ):
    device = check_device()
    if adata.isbacked:
        ds = FormulaViewDataset(adata, formula, chunk_size, device)
        dataloader = td.DataLoader(ds, batch_size=batch_size)
        ds.x_names = model_matrix_names(adata, formula, ds.categories)
    else:
        # convert sparse to dense matrix
        y = adata.X
        if isinstance(y, scipy.sparse._csc.csc_matrix):
            y = y.todense()

        # create tensor-based loader
        x = model_matrix(formula, adata.obs)
        ds = TensorDataset(
            torch.tensor(np.array(x), dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
        )
        ds.x_names = list(x.columns)
        dataloader = DataLoader(ds, batch_size=batch_size)

    return dataloader



class FormulaViewDataset(td.Dataset):
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

def replace_cols(obs, categories):
    for k in obs.columns:
        if str(obs[k].dtype) == "category":
            obs[k] = obs[k].astype(pd.CategoricalDtype(categories[k]))
    return obs
    


def model_matrix_names(adata, formula, categories):
    if adata.isbacked:
        obs = adata[:1].to_memory().obs
    else:
        obs = adata.obs

    if formula is None:
        return list(obs.columns)

    obs = replace_cols(obs, categories)
    return(list(model_matrix(formula, obs).columns))


def safe_model_matrix(obs, formula, categories):
    if formula is None:
        return obs

    obs = replace_cols(obs, categories)
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
