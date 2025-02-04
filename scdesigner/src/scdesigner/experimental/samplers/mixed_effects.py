from .samplers import Sampler
import torch.utils.data as td
import numpy as np
import torch


class LinearMixedEffectsSampler(Sampler):
    """
    Examples
    --------
    loader = FormulaWithGroupsLoader(example_sce, "~ pseudotime", "cell_type", batch_size=200)
    lme = LinearMixedEffectsEstimator(max_iter=100)
    parameters = lme.estimate(loader.loader)
    sampler = LinearMixedEffectsSampler(parameters)
    y = sampler.sample(loader.loader)
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader) -> np.array:
        beta = self.parameters["beta"]
        sigma_e = self.parameters["sigma_e"]
        sigma_b = self.parameters["sigma_b"]

        samples = []
        for Y, X, Z in loader:
            n_samples, n_responses = Y.shape
            n_groups = Z.shape[1]

            b = torch.randn((n_groups, n_responses)) * sigma_b
            E = torch.randn((n_samples, n_responses)) * sigma_e
            samples.append(X @ beta + Z @ b + E)

        return np.concatenate(samples, axis=0)


class PoissonMixedEffectsSampler(Sampler):
    """
    Examples
    --------
    loader = FormulaWithGroupsLoader(example_sce, "~ pseudotime", "cell_type", batch_size=200)
    pme = PoissonMixedEffectsEstimator(lr=1e-3, max_iter=1000)
    parameters = pme.estimate(loader.loader)

    sampler = PoissonMixedEffectsSampler(parameters)
    sampler.sample(loader.loader)
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader) -> np.array:
        beta = self.parameters["beta"]
        b = self.parameters["b"]

        samples = []
        for Y, X, Z in loader:
            mu = torch.exp(X @ beta + Z @ b)
            samples.append(torch.poisson(mu))
        return np.concatenate(samples, axis=0)
