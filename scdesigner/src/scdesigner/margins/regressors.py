from .. import design as ds
from . import parameter as pm
import lightning as pl
import torch
import itertools
import torch.optim
import torch.nn as nn
import numpy as np

class RegressionModule(pl.LightningModule):
    def __init__(self, n_input, gene_names=None):
        super().__init__()
        self.linear = {"mu": nn.Linear(n_input["mu"], len(gene_names))}
        self.gene_names = gene_names

    def configure_optimizers(self, lr=0.01, **kwargs):
        parameters = [p.parameters() for p in self.linear.values()]
        return torch.optim.LBFGS(itertools.chain(*parameters), lr=lr, **kwargs)

    def training_step(self, batch):
        return -self.loglikelihood(*batch)

    def validation_step(self):
        pass

    def loglikelihood(self):
        pass

class NBRegression(RegressionModule):
    def __init__(self, n_input, gene_names):
        super().__init__(n_input, gene_names)
        self.linear = {
            "alpha": nn.Linear(n_input["alpha"], len(gene_names)),
            "mu": nn.Linear(n_input["mu"], len(gene_names))
        }

    def forward(self, X):
        result = {}
        for k, v in X.items():
            f = self.linear[k].to(self.device)
            result[k] = torch.exp(f(v))
        return result

    def loglikelihood(self, X, obs, eps=1e-6):
        theta = self.forward(obs)
        mu, alpha = theta["mu"], theta["alpha"]

        return (
            X * torch.log(mu * alpha + eps)
            - (1 / alpha + X) * torch.log(1 + mu * alpha)
            + torch.lgamma(1 / alpha + X)
            - torch.lgamma(1 + X)
            - torch.lgamma(1 / alpha)
        ).mean()
