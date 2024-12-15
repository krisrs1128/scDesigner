from formulaic import model_matrix
import anndata
import gc
import numpy as np
import pandas as pd
import torch.utils.data as td

class DataParser:
    def __init__(self, data):
        self.loader = None
        self.names = None

class BasicDataset(td.Dataset):
    def __init__(self, X, obs):
        self.X = X
        self.obs = obs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i, :], self.obs.values[i, :]


class FormulaDataParser(DataParser):
    def __init__(self, data: anndata.AnnData, formula: str, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()
        obs_ = model_matrix(formula, data.obs)
        ds = BasicDataset(data.X, obs_)

        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), list(obs_.columns)

class MultiformulaDataParser(DataParser):
    def __init__(self, data: anndata.AnnData, formula: dict, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()
        
        obs = {}
        for k, f in formula.items():
            obs[k] = model_matrix(f, data.obs)

        ds = MultiformulaDataset(data.X, obs)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {k: list(v.columns) for k, v in obs.items()}

class MultiformulaDataset(BasicDataset):
    def __init__(self, X, obs):
        super().__init__(X, obs)

    def __getitem__(self, i):
        return self.X[i, :], {k: v.values[i, :] for k, v in self.obs.items()}

class BackedFormulaDataParser(DataParser):
    def __init__(self, data: anndata.AnnData, formula: str, chunk_size=int(2e4), **kwargs):
        data.obs = strings_as_categories(data.obs)
        ds = BackedFormulaDataset(data, formula, chunk_size)
        self.loader = td.DataLoader(ds, **kwargs)
        obs_names = model_matrix(formula, ds.data_inmem.obs).columns
        self.names = list(data.var_names), list(obs_names)

class BackedFormulaDataset(BasicDataset):
    def __init__(self, data: anndata.AnnData, formula: str, chunk_size: int):
        super().__init__(data.X, data.obs)
        self.cur_range = range(0, min(len(data), chunk_size))
        self.formula = formula
        self.filename = data.filename
        self.data_inmem = read_range(self.filename, self.cur_range)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            del self.data_inmem
            gc.collect()
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            self.data_inmem = read_range(self.filename, self.cur_range)

        X = self.data_inmem.X[ix - self.cur_range[0]]
        obs = model_matrix(self.formula, self.data_inmem.obs).values
        return X, obs[ix - self.cur_range[0]]

def strings_as_categories(df):
    for k in df.columns:
        if str(df[k].dtype) == "object":
            df[k] = pd.Categorical(df[k])
    return df

def read_range(filename, row_ix):
    view = anndata.read_h5ad(filename, backed="r")
    result = view[row_ix].to_memory()
    if "sparse" in str(type(result.X)):
        result.X = result.X.toarray().astype(np.float32)
    return result