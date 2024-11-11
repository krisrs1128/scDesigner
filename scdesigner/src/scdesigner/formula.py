from formulaic import model_matrix
from torch.utils.data import Dataset
import anndata
import gc
import numpy as np
import pandas as pd
import torch


def parse_formula(f, x_names):
    all_names = "+".join(x_names)
    return f.replace("~ .", all_names)


def initialize_formula(f, parameters=["alpha", "mu"], priority="mu"):
    if isinstance(f, str):
        f = {priority: f}

    for k in parameters:
        if k not in f.keys():
            f[k] = "~ 1"
    return f


def formula_to_groups(formula, obs):
    patterns = model_matrix(formula, obs)
    _, ix = np.unique(patterns, axis=0, return_inverse=True)
    return ix, int(ix.max())


class FormulaDataset(Dataset):
    def __init__(self, formula, adata, **kwargs):
        if adata.isbacked:
            ds = FormulaDatasetOnDisk(formula, adata, **kwargs)
        else:
            ds = FormulaDatasetInMemory(formula, adata, **kwargs)

        self.__class__ = ds.__class__
        self.__dict__ = ds.__dict__

class _FormulaDataset(Dataset):
    def __init__(self, formula, adata, **kwargs):
        self.len = len(adata)
        self.formula = initialize_formula(formula, **kwargs)
        self.obs = dict.fromkeys(self.formula.keys(), [])
        self.features = dict.fromkeys(self.formula.keys(), [])
        self.categories = column_levels(adata.obs)
        self.adata = adata

        for k, f in self.formula.items():
            self.formula[k] = parse_formula(f, adata.obs.columns)

    def __len__(self):
        return self.len

def read_obs(formula, obs, categories):
    features = {}
    model_df = {}
    for k in obs.columns:
        if str(obs[k].dtype) == "category":
            obs[k] = obs[k].astype(pd.CategoricalDtype(categories[k]))

    for k, f in formula.items():
        model_df_ = model_matrix(f, obs)
        features[k] = list(model_df_.columns)
        model_df[k] = torch.from_numpy(np.array(model_df_).astype(np.float32))

    return features, model_df

class FormulaDatasetInMemory(_FormulaDataset):
    def __init__(self, formula, adata, **kwargs):
        super().__init__(formula, adata, **kwargs)
        self.features, self.obs = read_obs(self.formula, self.adata.obs.copy(), self.categories)

        # setup the expression data tensor
        if "Sparse" in str(type(adata.X)):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X

        self.X = (
            torch.from_numpy(X.astype(np.float32))
            if adata.X is not None
            else torch.tensor([[float("nan")] * len(adata)]).reshape(-1, 1)
        )

    def __getitem__(self, ix):
        return self.X[ix], {k: self.obs[k][ix] for k in self.obs.keys()}

def column_levels(obs):
    categories = {}
    for k in obs.columns:
        obs_type = str(obs[k].dtype)
        if obs_type in ["category", "object"]:
            categories[k] = obs[k].unique()
    return categories

class FormulaDatasetOnDisk(_FormulaDataset):
    def __init__(self, formula, adata, chunk_size=int(2e4), **kwargs):
        super().__init__(formula, adata, **kwargs)
        self.cur_range = range(0, min(len(adata), chunk_size))
        self.vnames = list(self.adata.var_names)
        self.adata_inmem = read_range(self.adata.filename, self.cur_range, self.vnames)
        self.features, self.obs = read_obs(self.formula, self.adata_inmem.obs.copy(), self.categories)

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            del self.adata_inmem
            gc.collect()
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            self.adata_inmem = read_range(self.adata.filename, self.cur_range, self.vnames)
            self.features, self.obs = read_obs(self.formula, self.adata_inmem.obs.copy(), self.categories)

        if hasattr(self.adata_inmem, "X"):
            X = torch.from_numpy(
                self.adata_inmem.X[ix - self.cur_range[0]]
            )
        else:
            X = torch.tensor([[float("nan")]]).reshape(1, -1)

        return X, {k: self.obs[k][ix - self.cur_range[0]] for k in self.obs.keys()}

def read_range(filename, row_ix, var_names):
    view = anndata.read_h5ad(filename, backed="r")
    result = view[row_ix].to_memory()
    result = result[:, var_names].to_memory()
    if hasattr(result, "X"):
        if "Sparse" in str(type(result.X)):
            result.X = result.X.toarray().astype(np.float32)
    return result