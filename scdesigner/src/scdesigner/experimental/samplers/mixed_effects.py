from .samplers import Sampler
import torch.utils.data as td

class LinearMixedEffectsSampler(Sampler):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader):
        samples = []
        for l in loader:
            pass
        return samples

class PoissonMixedEffectsSampler(Sampler):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def sample(self, loader: td.DataLoader):
        samples = []
        for l in loader:
            pass
        return samples