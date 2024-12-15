from inspect import getmembers
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from typing import Callable
import lightning as pl
import torch
import torch.optim
import torch.utils.data as td


class GeneralizedLinearModelML:
    def __init__(self, hyper: dict):
        self.hyper = hyper
        self.module = None

    def link(self, x, parameter):
        return x

    def loglikelihood(self):
        pass

    def estimate(self, loader: td.DataLoader):
        n_output, n_input = data_dims(loader)
        self.module = GeneralizedLinearModelLightning(
            n_input, n_output, self.loglikelihood, self.link, self.hyper
        )
        early_stopping = EarlyStopping(monitor="NLL", min_delta=5e-4, patience=20)
        pl.Trainer(max_epochs=self.hyper["max_epochs"], callbacks=[early_stopping]).fit(
            self.module, train_dataloaders=loader
        )
        return {
            k.split(".")[1]: v.detach() for (k, v) in self.module.named_parameters()
        }


class GeneralizedLinearModelLightning(pl.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_output: dict,
        loglikelihood: Callable,
        link: dict,
        hyper: dict,
    ):
        super().__init__()
        self.loglikelihood = loglikelihood
        self.link = link
        self.optimizer_opts = args(torch.optim.Adam, **hyper)
        self.linear = nn.ModuleDict(
            {k: nn.Linear(n_input[k], n_output, bias=False) for k in n_input.keys()}
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_opts)

    def training_step(self, batch):
        loss = -self.loglikelihood(*batch)
        self.log("NLL", loss.item())
        return loss

    def forward(self, X):
        result = {}
        for k, v in X.items():
            f = self.linear[k].to(self.device)
            result[k] = self.link(f(v), k)
        return result


class NegativeBinomialML(GeneralizedLinearModelML):
    def __init__(self, hyper: dict):
        self.hyper = hyper

    def link(self, x, parameter):
        return torch.exp(x)

    def loglikelihood(self, Y, X, eps=1e-6):
        theta = self.module(X)
        mu, alpha = theta["mu"], theta["alpha"]

        return (
            Y * torch.log(mu * alpha + eps)
            - (1 / alpha + Y) * torch.log(1 + mu * alpha)
            + torch.lgamma(1 / alpha + Y)
            - torch.lgamma(1 + Y)
            - torch.lgamma(1 / alpha)
        ).mean()


def data_dims(loader):
    Y, Xs = next(iter(loader))
    return Y.shape[1], {k: v.shape[1] for k, v in Xs.items()}


def args(m, **kwargs):
    members = getmembers(m.__init__)[0][1].keys()
    return {kw: kwargs[kw] for kw in kwargs if kw in members}
