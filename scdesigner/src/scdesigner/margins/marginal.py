from ..formula import FormulaDataset
from .distributions import NegativeBinomial
from .regressors import NBRegression, NormalRegression
from collections import defaultdict
from inspect import getmembers
from lightning.pytorch.callbacks import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader
import anndata as ad
import lightning as pl
import torch
import torch.distributions


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
        self.optimizer_opts = args(Adam, **kwargs)

    def configure_loader(self, anndata):
        if self.loader_opts.get("batch_size") is None:
            self.loader_opts["batch_size"] = len(anndata)
        if self.loader_opts.get("pin_memory") is None:
            self.loader_opts["pin_memory"] = False

        dataset = FormulaDataset(self.formula, anndata, parameters=self.parameter_names)
        self.loader_opts["collate_fn"] = formula_collate
        return DataLoader(dataset, **self.loader_opts)

    def configure_module(self, anndata):
        if self.optimizer_opts.get("lr") is None:
            self.optimizer_opts["lr"] = 0.1

        ds = self.configure_loader(anndata)
        _, obs = ds.dataset[0]
        n_input = {k: v.shape[-1] for k, v in obs.items()}
        self.module = self.module(n_input, anndata.var_names)
        self.module.optimizer_opts = self.optimizer_opts

    def fit(self, anndata, max_epochs=500):
        if isinstance(self.module, type):
            self.configure_module(anndata)
        ds = self.configure_loader(anndata)
        early_stopping = EarlyStopping(monitor="NLL", min_delta=5e-4, patience=20)

        pl.Trainer(max_epochs=max_epochs, callbacks=[early_stopping]).fit(
            self.module, train_dataloaders=ds
        )

    def predict(self, obs):
        ds = self.configure_loader(ad.AnnData(obs=obs))
        preds = []
        for _, obs_ in ds:
            with torch.no_grad():
                preds.append(self.module(obs_))
        return {k: torch.concatenate([d[k] for d in preds], axis=0) for k in preds[0]}
    
    def sample(self, obs):
        return self.distn(obs).sample()

    def cdf(self, X, obs):
        return self.distn(obs).cdf(X)

    def icdf(self, U, obs):
        return self.distn(obs).icdf(U)

    def distn(self, obs):
        pass


def args(m, **kwargs):
    members = getmembers(m.__init__)[0][1].keys()
    return {kw: kwargs[kw] for kw in kwargs if kw in members}


class NB(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, NBRegression, **kwargs)
        self.parameter_names = ["mu", "alpha"]
    
    def distn(self, obs):
        params = self.predict(obs)
        total_count = 1 / params["alpha"]
        p = 1 - 1 / (1 + params["alpha"] * params["mu"])
        return NegativeBinomial(total_count, p)

class Normal(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, NormalRegression, **kwargs)
        self.parameter_names = ["mu", "sigma"]

    def distn(self, obs):
        params = self.predict(obs)
        return torch.distributions.Normal(params["mu"], params["sigma"])