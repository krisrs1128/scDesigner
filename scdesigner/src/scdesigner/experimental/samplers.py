import numpy as np
from scipy.stats import nbinom
import torch.utils.data as td
from torch import nn
import torch

class Sampler:
    def __init__(self, parameters):
        self.parameters = parameters

    def sample(loader: td.DataLoader):
        pass

class NegativeBinomialSampler(Sampler):
    def __init__(self, parameters):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader):
        n_output, n_input = parameter_dims(self.parameters)
        f = nn.ModuleDict({k: nn.Linear(n_input[k], n_output, bias=False) for k in n_input.keys()})

        with torch.no_grad():
            samples = [
                nbinom(
                    n=(1 / torch.exp(f["alpha"](X["alpha"]))).numpy(),
                    p=(1 - 1 / (1 + torch.exp(f["alpha"](X["alpha"])) * torch.exp(f["mu"](X["mu"])))).numpy()
                ).rvs()
                for _, X in loader
            ]

        return np.concatenate(samples, axis=0)

def parameter_dims(parameters):
    n_output = parameters["mu"].shape[0]
    n_input = {k: v.shape[1] for k, v in parameters.items()}
    return n_output, n_input
