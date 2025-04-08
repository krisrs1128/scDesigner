from ..data import SparseMatrixLoader
from ..samplers.samplers import Sampler
from ..estimators.memento import memento_error
import numpy as np

class MementoSampler(Sampler):
    def __init__(self, parameters: dict, q: float=0.07):
        super().__init__(parameters)
        try:
            import memento.simulate
        except ImportError:
            memento_error()
        self.q = q

    def sample(self, loader: SparseMatrixLoader) -> np.array:
        import memento.simulate as ms
        mean = self.parameters["mean"]
        variance = self.parameters["variance"]
        Nc = self.parameters["Nc"]
        norm_cov = self.parameters["norm_cov"]

        samples = []
        for x in loader:
            n_cells = x.shape[0]
            x_ = ms.simulate_transcriptomes(n_cells, mean, variance, Nc, norm_cov)
            samples.append(ms.capture_sampling(x_, q=self.q)[1])

        return np.concatenate(samples, axis=0)
