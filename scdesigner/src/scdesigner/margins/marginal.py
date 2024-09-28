import anndata as ad
import lightning as pl
import torch
import numpy as np
import pandas as pd
from formulaic import model_matrix
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from inspect import getmembers
from .regressors import NBRegression
from ..formula import FormulaDataset

def formula_collate(formula):
    def f(data):
        obs = {}
        for k in data[0][1]:
            obs[k] = []
            for d in data:
                obs[k].append(d[1][k])
            obs[k] = torch.from_numpy(np.array(model_matrix(formula[k], pd.concat(obs[k]))).astype(np.float32))

        X = np.concatenate([d[0] for d in data]).reshape(len(data), -1)
        return torch.from_numpy(X), obs
    return f


class MarginalModel:
    def __init__(self, formula, module, **kwargs):
        super().__init__()
        self.formula = formula
        self.module = module
        self.loader_opts = args(DataLoader, **kwargs)
        self.optimizer_opts = args(LBFGS, **kwargs)

    def configure_loader(self, anndata):
        if self.loader_opts.get("batch_size") is None:
            self.loader_opts["batch_size"] = len(anndata)

        dataset = FormulaDataset(self.formula, anndata)
        self.loader_opts["collate_fn"] = formula_collate(dataset.formula)
        return DataLoader(dataset, **self.loader_opts)

    def configure_module(self, anndata):
        ds = self.configure_loader(anndata)
        _, obs = next(iter(ds))
        n_input = {k: v.shape[-1] for k, v in obs.items()}
        self.module = self.module(n_input, anndata.var_names)
        self.module.configure_optimizers(**self.optimizer_opts)

    def fit(self, anndata, max_epochs=10):
        self.configure_module(anndata)
        ds = self.configure_loader(anndata)
        pl.Trainer(max_epochs=max_epochs, barebones=True).fit(self.module, train_dataloaders=ds)

    def predict(self, obs):
        ds = self.configure_loader(ad.AnnData(obs=obs))
        preds = []
        for _, obs_ in ds:
            with torch.no_grad():
                preds.append(self.module(obs_))

        return {k: torch.cat([d[k] for d in preds]).squeeze() for k in preds[0]}

    def parameters(self):
        pass

def args(m, **kwargs):
    members = getmembers(m.__init__)[0][1].keys()
    return {kw: kwargs[kw] for kw in kwargs if kw in members}

class NB(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, NBRegression, **kwargs)