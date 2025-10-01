from .formula import standardize_formula
from .marginal import GLMPredictor, Marginal
from .loader import _to_numpy
from typing import Union, Dict, Optional
import torch
import numpy as np
from scipy.stats import nbinom

class NegBin(Marginal):
    """Negative-binomial marginal estimator"""
    def __init__(self, formula: Union[Dict, str]):
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion'])
        super().__init__(formula)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        nll = lambda batch: -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch):
        """Compute the negative log-likelihood"""
        y, x = batch
        params = self.predict(x)
        mu = params.get('mean')
        r = params.get('dispersion')
        return (
            torch.lgamma(y + r)
            - torch.lgamma(r)
            - torch.lgamma(y + 1.0)
            + r * torch.log(r)
            + y * torch.log(mu)
            - (r + y) * torch.log(r + mu)
        )

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations."""
        mu, r, u = self._local_params(x, u)
        p = r / (r + mu)
        y = nbinom(n=r, p=p).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6):
        """Return uniformized pseudo-observations for counts y given covariates x."""
        # cdf values using scipy's parameterization
        mu, r, y = self._local_params(x, y)
        p = r / (r + mu)
        u1 = nbinom(n=r, p=p).cdf(y)
        u2 = np.where(y > 0, nbinom(n=r, p=p).cdf(y - 1), 0.0)

        # randomize within discrete mass to get uniform(0,1)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1.0 - v) * u2, epsilon, 1.0 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None):
        params = self.predict(x)
        mu = params.get('mean')
        r = params.get('dispersion')
        if y is None:
            return _to_numpy(mu, r)
        return _to_numpy(mu, r, y)
