import numpy as np
from typing import Union, Callable
from scipy.stats import nbinom, norm
import torch.utils.data as td
from torch import nn
import torch

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

            z = np.random.multivariate_normal(np.zeros(G), self.parameters["covariance"], N)
            normal_distn = norm(0, np.diag(self.parameters["covariance"] ** 0.5))
            u = normal_distn.cdf(z)

            icdf = self.inverter.__func__(self.parameters["margins"], x)
            samples.append(icdf(u))
        return np.concatenate(samples)


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