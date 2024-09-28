import lightning as pl
import torch
import itertools
import torch.optim
from math import log, pi
import torch.nn as nn

class RegressionModule(pl.LightningModule):
    def __init__(self, n_input, gene_names):
        super().__init__()
        self.linear = {k: nn.Linear(n_input[k], len(gene_names)) for k in n_input.keys()}
        self.gene_names = list(gene_names)
        self.automatic_optimization = False

    def configure_optimizers(self, lr=0.05, **kwargs):
        parameters = [p.parameters() for p in self.linear.values()]
        return torch.optim.LBFGS(itertools.chain(*parameters), lr=lr, **kwargs)

    def training_step(self, batch):
        opt = self.optimizers()
        def closure():
            loss = -self.loglikelihood(*batch)
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        opt.step(closure=closure)

    def loglikelihood(self):
        pass

class NBRegression(RegressionModule):
    def __init__(self, n_input, gene_names):
        super().__init__(n_input, gene_names)

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

class NormalRegression(RegressionModule):
    def __init__(self, n_input, gene_names):
        super().__init__(n_input, gene_names)

    def forward(self, X):
        result = {}
        for k, v in X.items():
            f = self.linear[k].to(self.device)
            result[k] = f(v)
        result["sigma"] = torch.exp(result["sigma"])
        return result

    def loglikelihood(self, X, obs):
        theta = self.forward(obs)
        mu, sigma = theta["mu"], theta["sigma"]
        D = X.shape[1]

        return - D / 2  * log(2 * pi) + \
            D * (- torch.log(sigma) - 1 / 2 * ((X - mu) / sigma) ** 2).mean()