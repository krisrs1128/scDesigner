from .formula import standardize_formula
from .marginal import GLMPredictor, Marginal
from .loader import _to_numpy
from typing import Union, Dict, Optional
import torch
import numpy as np
from scipy.stats import nbinom, bernoulli

class ZeroInflatedNegBin(Marginal):
    """Zero-inflated negative-binomial marginal estimator"""
    def __init__(self, formula: Union[Dict, str]):
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion', 'zero_inflation'])
        super().__init__(formula)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        link_fns = {"mean": torch.sigmoid}
        nll = lambda batch: -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_fns,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch):
        """Compute the negative log-likelihood"""
        y, x = batch
        params = self.predict(x)
        theta = params.get("mean")
        return y * torch.log(theta) + (1 - y) * torch.log(1 - theta)

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations."""
        theta, u = self._local_params(x, u)
        y = bernoulli(theta).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6):
        """Return uniformized pseudo-observations for counts y given covariates x."""
        theta, y = self._local_params(x, y)
        u1 =  bernoulli(theta).cdf(y)
        u2 = np.where(y > 0,  bernoulli(theta).cdf(y - 1), 0)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None):
        params = self.predict(x)
        theta = params.get('mean')
        if y is None:
            return _to_numpy(theta)
        return _to_numpy(theta, y)
