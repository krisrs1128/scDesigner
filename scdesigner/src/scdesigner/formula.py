from torch.utils.data import Dataset
import torch
import numpy as np
import anndata
from formulaic import model_matrix


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
        self.adata = adata
        
        for k, f in self.formula.items():
            self.formula[k] = parse_formula(f, adata.obs.columns)
            model_df = model_matrix(self.formula[k], adata.obs.copy())
            self.features[k] = list(model_df.columns)
            self.obs[k] = torch.from_numpy(np.array(model_df).astype(np.float32))

    def __len__(self):
        return self.len

class FormulaDatasetInMemory(_FormulaDataset):
    def __init__(self, formula, adata, **kwargs):
        super().__init__(formula, adata, **kwargs)
        if "csc" in str(type(adata.X)):
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

class FormulaDatasetOnDisk(_FormulaDataset):
    def __init__(self, formula, adata, chunk_size=int(1000), **kwargs):
        super().__init__(formula, adata, **kwargs)
        self.cur_range = range(0, min(len(adata), chunk_size))
        self.vnames = list(self.adata.var_names)
        self.adata_inmem = read_range(self.adata.filename, self.cur_range, self.vnames)
        self.is_sparse = "csc" in str(type(self.adata_inmem.X[0, 0]))
        if self.is_sparse:
            self.adata_inmem.X = self.adata_inmem.X.toarray()

    def __getitem__(self, ix):
        if ix not in self.cur_range:
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            self.adata_inmem = read_range(self.adata.filename, self.cur_range, self.vnames)
            if self.is_sparse:
                self.adata_inmem.X = self.adata_inmem.X.toarray()

        if hasattr(self.adata_inmem, "X"):
            X = torch.from_numpy(
                self.adata_inmem.X[
                    ix - self.cur_range[0]
                ].astype(np.float32)
            )
        else:
            X = torch.tensor([[float("nan")]]).reshape(1, -1)

        return X, {k: self.obs[k][ix] for k in self.obs.keys()}

def read_range(filename, row_ix, var_names):
    view = anndata.read_h5ad(filename, backed=True)
    result = view[row_ix].to_memory()
    return result[:, var_names].to_memory()