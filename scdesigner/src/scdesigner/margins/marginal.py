import anndata as ad
import lightning as pl
import numpy as np
import torch
from collections import defaultdict
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from inspect import getmembers
from .regressors import NBRegression, NormalRegression
from ..formula import FormulaDataset


def formula_collate(data):
    obs = defaultdict(list)
    X = []

    for d in data:
        X.append(d[0])
        for k, v in d[1].items():
            obs[k].append(v)

    return torch.stack(X), {k: torch.stack(v) for k, v in obs.items()}


class MarginalModel:
    def __init__(self, formula, module, **kwargs):
        super().__init__()
        self.formula = formula
        self.module = module
        self.parameter_names = None
        self.loader_opts = args(DataLoader, **kwargs)
        self.optimizer_opts = args(LBFGS, **kwargs)

    def configure_loader(self, anndata):
        if self.loader_opts.get("batch_size") is None:
            self.loader_opts["batch_size"] = len(anndata)

        dataset = FormulaDataset(self.formula, anndata, parameters=self.parameter_names)
        self.loader_opts["collate_fn"] = formula_collate
        return DataLoader(dataset, **self.loader_opts)

    def configure_module(self, anndata):
        ds = self.configure_loader(anndata)
        _, obs = next(iter(ds))
        n_input = {k: v.shape[-1] for k, v in obs.items()}
        self.module = self.module(n_input, anndata.var_names)
        self.module.optimizer_opts = self.optimizer_opts

    def fit(self, anndata, max_epochs=10):
        if isinstance(self.module, type):
            self.configure_module(anndata)
        ds = self.configure_loader(anndata)
        pl.Trainer(max_epochs=max_epochs, barebones=False).fit(
            self.module, train_dataloaders=ds
        )

    def predict(self, obs):
        ds = self.configure_loader(ad.AnnData(obs=obs))
        preds = []
        for _, obs_ in ds:
            with torch.no_grad():
                preds.append(self.module(obs_))
        return {k: torch.stack([d[k] for d in preds]).squeeze() for k in preds[0]}

    def sample(self, obs):
        pass


    def parameters(self):
        pass


def args(m, **kwargs):
    members = getmembers(m.__init__)[0][1].keys()
    return {kw: kwargs[kw] for kw in kwargs if kw in members}


class NB(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, NBRegression, **kwargs)
        self.parameter_names = ["mu", "alpha"]
    
    def sample(self, obs):
        params = self.predict(obs)
        total_count = 1 / params["alpha"]
        p = 1 / (1 + params["alpha"] * params["mu"])
        return np.random.negative_binomial(total_count, p)

class Normal(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, NormalRegression, **kwargs)
        self.parameter_names = ["mu", "sigma"]

    def sample(self, obs):
        params = self.predict(obs)
        return np.random.normal(params["mu"], params["sigma"])