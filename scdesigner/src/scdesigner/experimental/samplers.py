from .data import Loader, FormulaLoader
from anndata import AnnData
from copy import deepcopy
from scipy.stats import nbinom, norm
from torch import nn
from typing import Union, Callable
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td


class Sampler:
    def __init__(self, parameters):
        self.parameters = parameters

    def sample(loader: td.DataLoader):
        pass


class NegativeBinomialSampler(Sampler):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader):
        samples = []
        for l in loader:
            nb = nb_distn(self.parameters, process_batch(l))
            samples.append(nb.rvs())
        return np.concatenate(samples, axis=0)


class CompositeSampler(Sampler):
    def __init__(self, parameters: list[dict], sampler: Union[Sampler, list[Sampler]]):
        super().__init__(parameters)
        self.samplers = sampler

    def sample(self, loader: list[td.DataLoader]):
        if type(self.samplers) is not "list":
            self.samplers = [self.samplers] * len(loader)

        samples = []
        for i, sampler in enumerate(self.samplers):
            samples.append(sampler(self.parameters[i]).sample(loader[i]))
        return samples


class GCopulaSampler:
    def __init__(self, parameters: dict):
        self.parameters = parameters

    def sample(self, loader: td.DataLoader):
        samples = []
        for l in loader:
            x = process_batch(l)
            N = len(list(x.values())[0])
            G = self.parameters["covariance"].shape[0]

            z = np.random.multivariate_normal(
                np.zeros(G), self.parameters["covariance"], N
            )
            normal_distn = norm(0, np.diag(self.parameters["covariance"] ** 0.5))
            u = normal_distn.cdf(z)

            icdf = self.inverter.__func__(self.parameters["margins"], x)
            samples.append(icdf(u))
        return np.concatenate(samples)


def anndata_sample_n(sampler: Sampler, var_names: list, obs_names: dict):
    result = deepcopy(sampler)

    def new_sample(loader: td.DataLoader):
        y = pd.DataFrame(sampler.sample(loader), columns=var_names)
        obs = []
        for l in loader:
            l = process_batch(l)
            batch = pd.DataFrame()
            for k in list(l.keys()):
                cur = pd.DataFrame(l[k], columns=obs_names[k])
                ix = [c for c in cur.columns if c not in batch.columns]
                batch = pd.concat([batch, cur[ix]], axis=1)
            obs.append(batch)

        obs = pd.concat(obs)
        obs = obs[np.unique(obs.columns)]
        return AnnData(obs=obs, X=y)

    result.sample = new_sample
    return result

def anndata_sample_l(sampler: Sampler, formula: dict, loader: Loader = None):
    if loader is None:
        loader = FormulaLoader

    result = deepcopy(sampler)
    def new_sample(obs: pd.DataFrame):
        dl = loader(obs, formula, batch_size=len(obs))
        return AnnData(obs=obs, X=sampler.sample(dl.loader))

    result.sample = new_sample
    return result


def gcopula_sampler_factory(inverter: Callable):
    sampler = GCopulaSampler
    sampler.inverter = inverter
    return sampler


def parameter_dims(parameters: dict):
    n_output = parameters["mu"].shape[0]
    n_input = {k: v.shape[1] for k, v in parameters.items()}
    return n_output, n_input


def nb_distn(parameters: dict, X: dict):
    f = linear_module(parameters)
    total_count, p = nb_parameters(f, X)
    return nbinom(n=total_count, p=p)


def nb_parameters(f: nn.ModuleDict, X: dict):
    with torch.no_grad():
        alpha = torch.exp(f["alpha"](X["alpha"])).numpy()
        mu = torch.exp(f["mu"](X["mu"])).numpy()
    return 1 / alpha, 1 - 1 / (1 + alpha * mu)


def nb_inverter(parameters: dict, X: dict):
    distn = nb_distn(parameters, X)
    return lambda u: distn.ppf(u)


def process_batch(l: Union[list, dict]):
    # ignore training response Y if provided
    if type(l) is list:
        _, l = l
    return l


def linear_module(parameters):
    n_output, n_input = parameter_dims(parameters)
    return nn.ModuleDict(
        {k: nn.Linear(n_input[k], n_output, bias=False) for k in n_input.keys()}
    )


NegativeBinomialCopulaSampler = gcopula_sampler_factory(nb_inverter)
