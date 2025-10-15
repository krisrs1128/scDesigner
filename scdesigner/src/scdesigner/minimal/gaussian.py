from .formula import standardize_formula
from .marginal import GLMPredictor, Marginal
from .loader import _to_numpy
from typing import Union, Dict, Optional
import torch
import numpy as np
from scipy.stats import norm

class Gaussian(Marginal):
    """Gaussian marginal estimator"""
    def __init__(self, formula: Union[Dict, str]):
        formula = standardize_formula(formula, allowed_keys=['mean', 'sdev'])
        super().__init__(formula)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        nll = lambda batch: -self.likelihood(batch).sum()
        link_fns = {"mean": lambda x: x}
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
        mu = params.get("mean")
        sigma = params.get("sdev")

        # log likelihood for Gaussian
        log_likelihood = -0.5 * (torch.log(2 * torch.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))
        return log_likelihood

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations."""
        mu, sdev, u = self._local_params(x, u)
        y = norm(loc=mu, scale=sdev).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6):
        """Return uniformized pseudo-observations for counts y given covariates x."""
        # cdf values using scipy's parameterization
        mu, sdev, y = self._local_params(x, y)
        u = norm.cdf(y, loc=mu, scale=sdev)
        u = np.clip(u, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None):
        params = self.predict(x)
        mu = params.get('mean')
        sdev = params.get('sdev')
        if y is None:
            return _to_numpy(mu, sdev)
        return _to_numpy(mu, sdev, y)
