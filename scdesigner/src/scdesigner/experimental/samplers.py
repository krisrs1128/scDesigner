import numpy as np
from typing import Union
from scipy.stats import nbinom
import torch.utils.data as td
from torch import nn
import torch

class Sampler:
    def __init__(self, parameters):
        self.parameters = parameters

    def process_batch(self, l: Union[list, dict]):
        # ignore training response Y if provided
        if type(l) is list:
            _, l = l
        return l

    def sample(loader: td.DataLoader):
        pass

class NegativeBinomialSampler(Sampler):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader):
        n_output, n_input = parameter_dims(self.parameters)
        f = nn.ModuleDict({k: nn.Linear(n_input[k], n_output, bias=False) for k in n_input.keys()})

        samples = []
        for l in loader:
            total_count, p = nb_parameters(f, self.process_batch(l))
            samples.append(nbinom(n=total_count, p=p).rvs())
        return np.concatenate(samples, axis=0)

def parameter_dims(parameters: dict):
    n_output = parameters["mu"].shape[0]
    n_input = {k: v.shape[1] for k, v in parameters.items()}
    return n_output, n_input
    
def nb_parameters(f: nn.ModuleDict, X: dict):
    with torch.no_grad():
        alpha = torch.exp(f["alpha"](X["alpha"])).numpy()
        mu = torch.exp(f["mu"](X["mu"])).numpy()
    return 1 / alpha, 1 - 1 / (1 + alpha * mu)