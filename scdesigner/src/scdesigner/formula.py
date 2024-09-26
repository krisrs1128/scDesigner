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
        self.X = adata.X
        self.obs = adata.obs

        self.formula = initialize_formula(formula, **kwargs)
        for k, f in self.formula.items():
            self.formula[k] = parse_formula(f, self.obs.columns)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        obs_ = {}
        for k, f in self.formula.items():
            obs_[k] = np.array(model_matrix(f, self.obs.iloc[[ix]])).astype(np.float32)

        X_ = self.X[ix, :] if self.X is not None else None
        return X_, obs_
