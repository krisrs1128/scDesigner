from collections import defaultdict
import pandas as pd
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

    def fit(self, anndata, **kwargs):
        for margin in self.margins:
            y_names, submodel = margin
            submodel.fit(anndata[:, y_names], **kwargs)

    def predict(self, obs):
        param_hat = []
        for ix in range(self.margins):
            _, submodel = self.margins[ix]
            param_hat.append(submodel.predict(obs))
        return merge_predictions(param_hat)

    def parameters(self):
        theta = []
        for genes, submodel in self.margins:
            theta += (genes, submodel.parameters())
        return theta


def simulator(anndata, margins, delay=False, copula=None, **kwargs):
    if not isinstance(margins, list):
        margins = [(list(anndata.var_names), margins)]

    simulator = Simulator(margins, copula)
    if not delay:
        simulator.fit(anndata, **kwargs)

    return simulator
