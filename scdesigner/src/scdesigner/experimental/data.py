from formulaic import model_matrix
import anndata
import gc
import numpy as np
import pandas as pd
import torch.utils.data as td


class Loader:
    def __init__(self, data):
        self.loader = None
        self.names = None


class BasicDataset(td.Dataset):
    def __init__(self, X, obs):
        self.X = X
        self.obs = obs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i, :], self.obs.values[i, :]


class FormulaLoader(Loader):
    def __init__(self, data: anndata.AnnData, formula: str, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()
        obs_ = model_matrix(formula, data.obs)
        ds = BasicDataset(data.X, obs_)

        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), list(obs_.columns)


################################################################################
# Dataloader when there are different predictors across formula terms
################################################################################


class MultiformulaLoader(Loader):
    def __init__(self, data: anndata.AnnData, formula: dict, **kwargs):
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray()

        obs = model_matrix_dict(formula, data.obs)
        ds = MultiformulaDataset(data.X, obs)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {k: list(v.columns) for k, v in obs.items()}


class MultiformulaDataset(BasicDataset):
    def __init__(self, X, obs):
        super().__init__(X, obs)

    def __getitem__(self, i):
        return self.X[i, :], {k: v.values[i, :] for k, v in self.obs.items()}


################################################################################
# Read chunks in memory when there are multiple `obs` for simple string formulas
################################################################################


class BackedFormulaLoader(Loader):
    def __init__(
        self, data: anndata.AnnData, formula: str, chunk_size=int(2e4), **kwargs
    ):
        data.obs = strings_as_categories(data.obs)
        ds = BackedFormulaDataset(data, formula, chunk_size)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), list(ds.obs_inmem.columns)


class BackedFormulaDataset(BasicDataset):
    def __init__(self, data: anndata.AnnData, formula: str, chunk_size: int):
        super().__init__(data.X, data.obs)
        self.cur_range = range(0, min(len(data), chunk_size))
        self.formula = formula
        self.data_inmem = read_range(data.filename, self.cur_range)
        self.obs_inmem = model_matrix(self.formula, self.data_inmem.obs)

    def update_range(self, ix):
        del self.data_inmem
        gc.collect()
        self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
        self.data_inmem = read_range(self.data_inmem.filename, self.cur_range)
        self.obs_inmem = model_matrix(self.formula, self.data_inmem.obs)

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            self.update_range(ix)

        X = self.data_inmem.X[ix - self.cur_range[0]]
        return X, self.obs_inmem.values[ix - self.cur_range[0]]


################################################################################
# Read chunks in memory when there are multiple `obs` needed for different
# formula elements
################################################################################


class BackedMultiformulaLoader(Loader):
    def __init__(
        self, data: anndata.AnnData, formula: str, chunk_size=int(2e4), **kwargs
    ):
        data.obs = strings_as_categories(data.obs)
        ds = BackedMultiformulaDataset(data, formula, chunk_size)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {
            k: list(v.columns) for k, v in ds.obs_inmem.items()
        }


class BackedMultiformulaDataset(BasicDataset):
    def __init__(self, data: anndata.AnnData, formula: dict, chunk_size: int):
        super().__init__(data.X, data.obs)
        self.cur_range = range(0, min(len(data), chunk_size))
        self.formula = formula
        self.data_inmem = read_range(data.filename, self.cur_range)
        self.obs_inmem = model_matrix_dict(self.formula, self.data_inmem.obs)

    def update_range(self, ix):
        del self.data_inmem
        gc.collect()
        self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
        self.data_inmem = read_range(self.data_inmem.filename, self.cur_range)
        self.obs_inmem = model_matrix_dict(self.formula, self.data_inmem.obs)

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            self.update_range(ix)

        X = self.data_inmem.X[ix - self.cur_range[0]]
        return X, \
            {k: v.values[ix - self.cur_range[0], :] for k, v in self.obs_inmem.items()}


################################################################################
# Helper functions
################################################################################


def model_matrix_dict(formula, obs_df):
    obs = {}
    for k, f in formula.items():
        obs[k] = model_matrix(f, obs_df)
    return obs


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
