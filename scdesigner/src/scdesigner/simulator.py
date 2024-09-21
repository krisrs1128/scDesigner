from scdesigner.print import print_simulator
import torch

class Simulator():
    def __init__(self, margins, copula=None):
        super().__init__()
        self.margins = margins
        self.copula = copula

    def fit(self, anndata, index=None, **kwargs):
        if index is None:
            index = range(len(self.margins))
        for ix in index:
            genes, submodel = self.margins[ix]
            submodel.fit(
                torch.from_numpy(anndata[:, genes].X.toarray()),
                anndata.obs,
                **kwargs
            )

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