import anndata as ad
import lightning as pl
import torch
import torch.distributions
from collections import defaultdict
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from inspect import getmembers
from . import regressors as reg
from .distributions import NegativeBinomial
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
        _, obs = ds.dataset[0]
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
        super().__init__(formula, reg.NBRegression, **kwargs)
        self.parameter_names = ["mu", "alpha"]
    
    def distn(self, obs):
        params = self.predict(obs)
        total_count = 1 / params["alpha"]
        p = 1 - 1 / (1 + params["alpha"] * params["mu"])
        return NegativeBinomial(total_count, p)

class Normal(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, reg.NormalRegression, **kwargs)
        self.parameter_names = ["mu", "sigma"]

    def distn(self, obs):
        params = self.predict(obs)
        return torch.distributions.Normal(params["mu"], params["sigma"])
    
class Poisson(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, reg.PoissonRegression, **kwargs)
        self.parameter_names = ["mu"]

    def distn(self, obs):
        params = self.predict(obs)
        return torch.distributions.Poisson(params["mu"]) 
    
    def cdf(self, X, obs):
        mu = self.predict(obs)["mu"]
        return torch.special.gammaincc(torch.floor(X+1), mu) # https://github.com/pytorch/pytorch/issues/97156
    
    def icdf(self, U, obs):
        pass
    
class Bernoulli(MarginalModel):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, reg.BernoulliRegression, **kwargs)
        self.parameter_names = ["mu"]

    def distn(self, obs):
        params = self.predict(obs)
        return torch.distributions.Bernoulli(logits=params["mu"]) 
    
    def cdf(self, X, obs):
        mu = self.predict(obs)["mu"]
        p_0 = 1 / (1+torch.exp(mu))
        
        cdf = torch.zeros_like(X, dtype=torch.float32)
        cdf = torch.where((X >= 0) & (X < 1), p_0, cdf)
        return torch.where(X >= 1, torch.tensor(1.0), cdf)
    
    def icdf(self, U, obs):
        pass
    