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
    def __init__(self, formula, adata, **kwargs):
        super().__init__(formula, adata, **kwargs)
        self.is_sparse = "csc" in str(type(adata.X[0, 0]))

    def __getitem__(self, ix):
        if self.is_sparse:
            X = self.adata.X[ix].toarray()
        else:
            vnames = list(self.adata.var_names)
            view = anndata.read_h5ad(self.adata.filename, backed=True)
            X = view[:, vnames].X[ix]

        X = (
            torch.from_numpy(X.astype(np.float32))
            if self.adata.X is not None
            else torch.tensor([[float("nan")]]).reshape(1, -1)
        )

        return X, {k: self.obs[k][ix] for k in self.obs.keys()}