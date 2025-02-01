from formulaic import model_matrix
from typing import Union
import anndata
import numpy as np
import pandas as pd
import torch
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
        return self.obs.shape[0]

    def __getitem__(self, i):
        return self.X[i, :], self.obs.values[i, :]


################################################################################
# Dataloader when there are different predictors across formula terms
################################################################################


class FormulaLoader(Loader):
    def __init__(
        self, data: Union[anndata.AnnData, pd.DataFrame], formula: dict, **kwargs
    ):
        if type(data) is pd.DataFrame:
            data = anndata.AnnData(obs=data)

        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray().astype(np.float32)

        obs = model_matrix_dict(formula, data.obs)
        ds = FormulaDataset(data.X, obs)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {k: list(v.columns) for k, v in obs.items()}


class FormulaDataset(BasicDataset):
    def __init__(self, X, obs):
        super().__init__(X, obs)

    def __len__(self):
        v = list(self.obs.values())[0]
        return v.shape[0]

    def __getitem__(self, i):
        obs_i = {k: v.values[i, :].astype(np.float32) for k, v in self.obs.items()}
        if self.X is not None:
            return self.X[i, :], obs_i
        return obs_i


################################################################################
# Data loader when we want to track a "group" term outside of the main design
# matrix. This is mainly useful for mixed effects models, where group is the
# one-hot-encoded matrix for random effects.
################################################################################


class FormulaWithGroupsLoader(Loader):
    def __init__(
        self,
        data: Union[anndata.AnnData, pd.DataFrame],
        formula: str,
        group: str,
        **kwargs,
    ):
        if type(data) is pd.DataFrame:
            data = anndata.AnnData(obs=data)
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray().astype(np.float32)

        obs = model_matrix(formula, data.obs)
        groups = model_matrix(f"""-1 + {group}""", data.obs)
        ds = FormulaWithGroupsDataset(data.X, obs, groups)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), obs.columns


class FormulaWithGroupsDataset(BasicDataset):
    def __init__(self, X, obs, groups):
        super().__init__(X, obs)
        self.groups = groups

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        obs_i = self.obs.values[i, :].astype(np.float32)
        groups_i = self.groups.values[i, :].astype(np.float32)
        if self.X is not None:
            return self.X[i, :], obs_i, groups_i
        return obs_i, groups_i


################################################################################
# A version of the formula-groups that accomodates different formulas for each
# parameter. This is helpsful when we want to use different relationships for
# the mean and variance, for example.
################################################################################


class MultiFormulaWithGroupsLoader(Loader):
    def __init__(
        self,
        data: Union[anndata.AnnData, pd.DataFrame],
        formula: dict,
        group: str,
        **kwargs,
    ):
        if type(data) is pd.DataFrame:
            data = anndata.AnnData(obs=data)
        if "sparse" in str(type(data.X)):
            data.X = data.X.toarray().astype(np.float32)

        obs = model_matrix_dict(formula, data.obs)
        groups = model_matrix(f"""-1 + {group}""", data.obs)
        ds = MultiFormulaWithGroupsDataset(data.X, obs, groups)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {k: list(v.columns) for k, v in obs.items()}


class MultiFormulaWithGroupsDataset(BasicDataset):
    def __init__(self, X, obs, groups):
        super().__init__(X, obs)
        self.groups = groups

    def __len__(self):
        v = list(self.obs.values())[0]
        return v.shape[0]

    def __getitem__(self, i):
        obs_i = {k: v.values[i, :].astype(np.float32) for k, v in self.obs.items()}
        groups_i = self.groups.values[i, :].astype(np.int32)
        if self.X is not None:
            return self.X[i, :], obs_i, groups_i
        return obs_i, groups_i


################################################################################
# Read chunks in memory when there are multiple `obs` needed for different
# formula elements
################################################################################


class BackedFormulaLoader(Loader):
    def __init__(
        self,
        data: Union[anndata.AnnData, pd.DataFrame],
        formula: dict,
        chunk_size=int(2e4),
        **kwargs,
    ):
        if type(data) is pd.DataFrame:
            data = anndata.AnnData(obs=data)

        data.obs = strings_as_categories(data.obs)
        ds = BackedFormulaDataset(data, formula, chunk_size)
        self.loader = td.DataLoader(ds, **kwargs)
        self.names = list(data.var_names), {
            k: list(v.columns) for k, v in ds.obs_inmem.items()
        }


class BackedFormulaDataset(BasicDataset):
    def __init__(self, data: anndata.AnnData, formula: dict, chunk_size: int):
        super().__init__(data.X, {"obs": data.obs})
        self.cur_range = range(0, min(len(data), chunk_size))
        self.formula = formula
        self.filename = data.filename
        self.data_inmem = read_range(self.filename, self.cur_range)
        self.obs_inmem = model_matrix_dict(self.formula, self.data_inmem.obs)

    def update_range(self, ix):
        self.cur_range = range(ix, min(ix + len(self.cur_range), self.__len__()))
        self.data_inmem = read_range(self.filename, self.cur_range)
        self.obs_inmem = model_matrix_dict(self.formula, self.data_inmem.obs)

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            self.update_range(ix)

        obs_i = {
            k: v.values[ix - self.cur_range[0], :].astype(np.float32)
            for k, v in self.obs_inmem.items()
        }
        if self.X is not None:
            return self.data_inmem.X[ix - self.cur_range[0]], obs_i
        return obs_i


################################################################################
# Dataloader when different sets of features need different formulas
################################################################################


class CompositeFormulaLoader(Loader):
    def __init__(
        self,
        data: Union[list[anndata.AnnData], pd.DataFrame],
        formula: list[dict],
        **kwargs,
    ):
        if type(data) is pd.DataFrame:
            data = [anndata.AnnData(obs=data)] * len(formula)

        loader = []
        names = []
        for i, f in enumerate(formula):
            fl = FormulaLoader(data[i], f, **kwargs)
            loader.append(fl.loader)
            names.append(fl.names)

        self.loader = loader
        self.names = names


class BackedCompositeFormulaLoader(Loader):
    def __init__(
        self,
        data: Union[list[anndata.AnnData], pd.DataFrame],
        formula: list[dict],
        **kwargs,
    ):
        if type(data) is pd.DataFrame:
            data = [anndata.AnnData(obs=data)] * len(formula)

        loader = []
        names = []
        for i, f in enumerate(formula):
            fl = BackedFormulaLoader(data[i], f, **kwargs)
            loader.append(fl.loader)
            names.append(fl.names)

        self.loader = loader
        self.names = names


################################################################################
# Data for when the .X matrix is a sparse matrix
################################################################################


class SparseMatrixLoader(Loader):
    def __init__(self, adata: anndata.AnnData, batch_size: int = None):
        ds = SparseMatrixDataset(adata, batch_size)
        self.loader = td.DataLoader(ds, batch_size=None)


class SparseMatrixDataset(td.IterableDataset):
    def __init__(self, anndata: anndata.AnnData, batch_size: int = None):
        self.n_rows = anndata.X.shape[0]
        if batch_size is None:
            batch_size = self.n_rows

        self.sparse_matrix = anndata.X
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, self.n_rows, self.batch_size):
            batch_indices = range(i, min(i + self.batch_size, self.n_rows))
            batch_rows = self.sparse_matrix[batch_indices, :]

            # Convert to sparse CSR tensor
            batch_indices_rows, batch_indices_cols = batch_rows.nonzero()
            batch_values = batch_rows.data

            batch_sparse_tensor = torch.sparse_coo_tensor(
                torch.tensor([batch_indices_rows, batch_indices_cols]),
                torch.tensor(batch_values, dtype=torch.float32),
                (len(batch_indices), self.sparse_matrix.shape[1]),
            ).to_sparse_csr()

            yield batch_sparse_tensor

    def __len__(self):
        return (self.n_rows + self.batch_size - 1) // self.batch_size


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
