from scdesigner.print import print_simulator
from collections import defaultdict
import pandas as pd
import torch


def merge_predictions(param_hat):
    merged = defaultdict(list)
    for d in param_hat:
        for k, v in d.items():
            merged[k].append(v)

    return {k: pd.concat(v, axis=1) for k, v in merged.items()}


class Simulator:
    def __init__(self, margins, copula=None):
        super().__init__()
        self.margins = margins
        self.copula = copula

    def fit(self, anndata, index=None, **kwargs):
        if index is None:
            index = range(len(self.margins))
        for ix in index:
            y_names, submodel = self.margins[ix]
            submodel.fit(
                torch.from_numpy(anndata[:, y_names].X.toarray()),
                anndata.obs,
                y_names,
                **kwargs
            )

    def predict(self, X, index=None, **kwargs):
        if index is None:
            index = range(len(self.margins))
        param_hat = []
        for ix in index:
            _, submodel = self.margins[ix]
            param_hat.append(submodel.predict(X))
        return merge_predictions(param_hat)

    def parameters(self, index=None):
        if index is None:
            index = range(len(self.margins))

        theta = []
        for ix in index:
            genes, submodel = self.margins[ix]
            theta += (genes, submodel.parameters)
        return theta

    def __repr__(self):
        print_simulator(self.margins, self.copula)
        return ""

    def __str__(self):
        return ""


def simulator(anndata, margins, delay=False, copula=None, **kwargs):
    if not isinstance(margins, list):
        margins = [(list(anndata.var_names), margins)]

    simulator = Simulator(margins, copula)
    if not delay:
        simulator.fit(anndata, **kwargs)

    return simulator
