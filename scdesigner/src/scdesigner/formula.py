from torch.utils.data import Dataset
from formulaic import model_matrix
import numpy as np


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
        self.adata = adata

        self.formula = initialize_formula(formula, **kwargs)
        for k, f in self.formula.items():
            self.formula[k] = parse_formula(f, self.adata.obs.columns)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, ix):
        obs_ = {}
        for k, _ in self.formula.items():
            obs_[k] = self.adata.obs.iloc[[ix]]

        X_ = self.adata.X[ix, :] if self.adata.X is not None else []
        return X_, obs_