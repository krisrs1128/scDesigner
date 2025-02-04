from ..samplers.samplers import nb_distn, linear_module
from inspect import getmembers
from lightning.pytorch.callbacks import EarlyStopping
from scipy.stats import norm
from torch import nn
from typing import Callable, Union
import lightning as pl
import numpy as np
import torch
import torch.optim
import torch.utils.data as td


class Estimator:
    def __init__(self, hyper: dict):
        self.hyper = hyper

    def estimate(self, loader: td.DataLoader):
        pass


class GeneralizedLinearModelML(Estimator):
    def __init__(self, hyper: dict):
        super().__init__(hyper)
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
        super().__init__(hyper)

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


class CompositeEstimator(Estimator):
    def __init__(self, estimators: Union[Estimator, list[Estimator]], hyper: dict):
        super().__init__(hyper)
        self.estimators = estimators

    def estimate(self, loader: list[td.DataLoader]):
        if type(self.estimators) is not "list":
            self.estimators = [self.estimators] * len(loader)

        parameters = []
        for i, estimator in enumerate(self.estimators):
            parameters.append(estimator(self.hyper).estimate(loader[i]))
        return parameters


class GCopulaEstimator:
    def __init__(self, hyper: dict):
        self.hyper = hyper

    def estimate(self, loader: td.DataLoader):
        margins = self.marginal_estimator(self.hyper).estimate(loader)
        z = self.gaussianizer.__func__(margins, loader)
        covariance = self.covariance_fun.__func__(z.T)
        return {"covariance": torch.from_numpy(covariance), "margins": margins}

def gcopula_estimator_factory(
    marginal_estimator: Estimator,
    gaussianizer: Callable,
    covariance_fun: Callable = None,
) -> GCopulaEstimator:
    if covariance_fun is None:
        covariance_fun = np.cov

    copula = GCopulaEstimator
    copula.marginal_estimator = marginal_estimator
    copula.gaussianizer = gaussianizer
    copula.covariance_fun = covariance_fun
    return copula


def negbin_gaussianizer(margins: dict, loader: td.DataLoader) -> np.array:
    z = []
    f = linear_module(margins)
    for y, x in loader:
        distribution = nb_distn(margins, x)
        alpha = np.random.uniform(size=y.shape)
        u = clip(alpha * distribution.cdf(y) + (1 - alpha) * distribution.cdf(1 + y))
        z.append(norm().ppf(u))

    return np.concatenate(z)


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def data_dims(loader: td.DataLoader) -> tuple:
    Y, Xs = next(iter(loader))
    return Y.shape[1], {k: v.shape[1] for k, v in Xs.items()}


def args(m: Callable, **kwargs) -> dict:
    members = getmembers(m.__init__)[0][1].keys()
    return {kw: kwargs[kw] for kw in kwargs if kw in members}


NegativeBinomialCopulaEstimator = gcopula_estimator_factory(
    NegativeBinomialML, negbin_gaussianizer
)
