import anndata
import numpy as np
import pandas as pd
from formulaic import model_matrix
import torch.utils.data as td

class DataParser:
    def __init__(self, data):
        self.loader = None
        self.names = None

class FormulaDataParser(DataParser):
    def __init__(self, data: anndata.AnnData, formula: str, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()
        obs_ = model_matrix(formula, data.obs)
        ds = BasicDataset(data.X, obs_)

        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), list(obs_.columns)

class BasicDataset(td.Dataset):
    def __init__(self, X, obs):
        self.X = X
        self.obs = obs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i, :], self.obs.iloc[i, :].values


class MultiformulaDataParser(DataParser):
    def __init__(self, data: anndata.AnnData, formula: dict, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()
        
        # covariates per formula element
        obs_ = {}
        for k, f in formula.items():
            obs_[k] = model_matrix(f, data.obs)

        # combine into a data loader
        ds = MultiformulaDataset(data.X, obs_)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {k: list(v.columns) for k, v in obs_.items()}


class MultiformulaDataset(td.Dataset):
    def __init__(self, X, obs):
        self.X = X
        self.obs = obs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i, :], {k: v.iloc[i, :].values for k, v in self.obs.items()}
