from collections import defaultdict
from .margins.marginal import args
from torch.optim import LBFGS
from torch.utils.data import DataLoader
import pandas as pd

def merge_predictions(param_hat):
    merged = defaultdict(list)
    for genes, d in param_hat:
        for k, v in d.items():
            df = pd.DataFrame(v.numpy(), columns=genes)
            merged[k].append(df)

    return {k: pd.concat(v, axis=1) for k, v in merged.items()}


class Simulator:
    def __init__(self, margins, copula=None):
        super().__init__()
        self.margins = margins
        self.copula = copula

    def fit(self, anndata, max_epochs=10):
        for margin in self.margins:
            y_names, submodel = margin
            submodel.fit(anndata[:, y_names], max_epochs)

    def predict(self, obs):
        param_hat = [] 
        for genes, submodel in self.margins:
            param_hat.append([genes, submodel.predict(obs)])
        return merge_predictions(param_hat)

    def parameters(self):
        theta = []
        for genes, submodel in self.margins:
            theta += (genes, submodel.parameters())
        return theta

def safe_update(d1, d2):
    d1.update({k: v for k, v in d2.items() if k not in d1})
    return d1

def simulator(anndata, margins, delay=False, copula=None, max_epochs=10, **kwargs):
    if not isinstance(margins, list):
        margins = [(list(anndata.var_names), margins)]

    for _, margin in margins:
        margin.loader_opts = safe_update(margin.loader_opts, args(DataLoader, **kwargs))
        margin.optimizer_opts = safe_update(margin.optimizer_opts, args(LBFGS, **kwargs))

    simulator = Simulator(margins, copula)
    if not delay:
        simulator.fit(anndata, max_epochs)

    return simulator
