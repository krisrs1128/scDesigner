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

        link_funs = {
            "mean": torch.exp,
            "dispersion": torch.exp,
            "zero_inflation": torch.sigmoid,
        }
        nll = lambda batch: -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_funs,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch):
        """Compute the negative log-likelihood"""
        y, x = batch
        params = self.predict(x)
        mu = params.get("mean")
        r = params.get("dispersion")
        pi = params.get("zero_inflation")

        # negative binomial component
        negbin_loglikelihood = (
            torch.lgamma(y + r)
            - torch.lgamma(r)
            - torch.lgamma(y + 1)
            + r * torch.log(r)
            + y * torch.log(mu)
            - (r + y) * torch.log(r + mu)
        )

        # return the mixture, with an offset to prevent log(0)
        return torch.log(pi * (y == 0) + (1 - pi) * torch.exp(negbin_loglikelihood) + 1e-10)

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations."""
        mu, r, pi, u = self._local_params(x, u)
        y = nbinom(n=r, p=r / (r + mu)).ppf(u)
        delta = bernoulli(1 - pi).ppf(u)
        return torch.from_numpy(y * delta).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6):
        """Return uniformized pseudo-observations for counts y given covariates x."""
        # cdf values using scipy's parameterization
        mu, r, pi, y = self._local_params(x, y)
        nb_distn = nbinom(n=r, p=r / (r + mu))
        u1 = pi + (1 - pi) * nb_distn.cdf(y)
        u2 = np.where(y > 0, pi + (1 - pi) * nb_distn.cdf(y-1), 0)

        # randomize within discrete mass to get uniform(0,1)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None):
        params = self.predict(x)
        mu = params.get('mean')
        r = params.get('dispersion')
        pi = params.get('zero_inflation')
        if y is None:
            return _to_numpy(mu, pi, r)
        return _to_numpy(mu, r, pi, y)
