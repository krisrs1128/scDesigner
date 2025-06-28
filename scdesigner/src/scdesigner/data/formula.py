from anndata import AnnData
from formulaic import model_matrix
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.utils.data as td
import itertools

def formula_loader(
    adata: AnnData, formula=None, chunk_size=int(1e4), batch_size: int = None
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
        x = model_matrix(formula, pd.DataFrame(adata.obs))
        ds = TensorDataset(
            torch.tensor(np.array(x), dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
        )
        ds.x_names = list(x.columns)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return dataloader

# Attempt1: return a dictionary of data loaders for each formula
def multiple_formula_loader(
    adata: AnnData, formulas: dict, chunk_size=int(1e4), batch_size: int = None
):
    dataloaders = {}
    for key in formulas.keys():
        dataloaders[key] = formula_loader(adata, formulas[key], chunk_size, batch_size)
    return dataloaders

# class JoinedDataset(td.Dataset):
#     """
#     Wrap a list of datasets that share the same Y with different formulas for X.
#     For each index, concatenate the Xs and return the corresponding Y.
#     """
#     def __init__(self, datasets: list[td.Dataset]):
#         self.datasets = datasets
#         self.len = len(datasets[0])
#         for d in datasets[1:]:
#             assert len(d) == self.len, "All datasets must have the same length"
    
#         # merge covariate names
#         self.x_names = list(itertools.chain.from_iterable(
#             getattr(d, "x_names", [f'x_{i}' for i in range(d[0][0].shape[-1])]) 
#             for d in datasets
#         ))
        
#     def __getitem__(self, idx):
#         xs = []
#         y_ref = None
#         for d in self.datasets:
#             x_i, y_i = d[idx]
#             xs.append(x_i)
#             if y_ref is None:
#                 y_ref = y_i
#             else:
#                 # Check Y is identical
#                 if not torch.equal(y_ref, y_i):
#                     raise ValueError(f"Y mismatch at index {idx}")
#         x_joined = torch.cat(xs, dim=-1)
#         return x_joined, y_ref

#     def __len__(self):
#         return self.len
    
# def joined_dataloaders(dataloaders: dict, batch_size=None, **kwargs):
#     datasets = [dl.dataset for dl in dataloaders.values()]
#     joined_dataset = JoinedDataset(datasets)
#     return DataLoader(joined_dataset, 
#                       batch_size=batch_size or dataloaders[0].batch_size, 
#                       shuffle=False, 
#                       **kwargs)

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
    return list(model_matrix(formula, pd.DataFrame(obs)).columns)


def safe_model_matrix(obs, formula, categories):
    if formula is None:
        return obs

    obs = replace_cols(obs, categories)
    x = model_matrix(formula, pd.DataFrame(obs))
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
