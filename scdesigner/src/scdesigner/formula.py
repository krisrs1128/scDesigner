from torch.utils.data import Dataset
import torch
import numpy as np
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
        self.len = len(adata)
        self.X = (
            torch.from_numpy(adata.X)
            if adata.X is not None
            else torch.tensor([[float("nan")] * len(adata)]).reshape(-1, 1)
        )

        self.formula = initialize_formula(formula, **kwargs)
        self.obs = dict.fromkeys(self.formula.keys(), [])
        for k, f in self.formula.items():
            self.formula[k] = parse_formula(f, adata.obs.columns)
            self.obs[k] = torch.from_numpy(
                np.array(model_matrix(self.formula[k], adata.obs.copy())).astype(
                    np.float32
                )
            )

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        return self.X[ix], {k: self.obs[k][ix] for k in self.obs.keys()}
