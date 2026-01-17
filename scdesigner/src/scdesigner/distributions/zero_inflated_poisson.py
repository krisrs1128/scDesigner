from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import poisson, bernoulli

class ZeroInflatedPoisson(Marginal):
    """Zero-Inflated Poisson marginal estimator

    This subclass models counts with an explicit zero-inflation component.
    For each feature j the observation follows a mixture: with probability
    `pi_j(x)` the value is an extra zero, otherwise the count is drawn from
    a Poisson distribution with mean `mu_j(x)`. Both `mu_j(x)` and the
    inflation probability `pi_j(x)` may depend on covariates `x` through the
    `formula` argument.

    The allowed formula keys are 'mean' and 'zero_inflation'. If a string
    formula is supplied it is taken to specify the `mean` by default.

    Examples
    --------
    >>> from scdesigner.distributions import ZeroInflatedPoisson
    >>> from scdesigner.datasets import pancreas
    >>>
    >>> sim = ZeroInflatedPoisson(formula={"mean": "~ pseudotime", "zero_inflation": "~ pseudotime"})
    >>> sim.setup_data(pancreas)
    >>> sim.fit(max_epochs=1, verbose=False)
    >>>
    >>> # evaluate p(y | x) and model parameters
    >>> y, x = next(iter(sim.loader))
    >>> l = sim.likelihood((y, x))
    >>> y_hat = sim.predict(x)
    >>>
    >>> # convert to quantiles and back
    >>> u = sim.uniformize(y, x)
    >>> x_star = sim.invert(u, x)
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        formula = standardize_formula(formula, allowed_keys=['mean', 'zero_inflation'])
        super().__init__(formula, **kwargs)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        link_funs = {
            "mean": torch.exp,
            "zero_inflation": torch.sigmoid,
        }
        def nll(batch):
            return -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_funs,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood"""
        y, x = batch
        params = self.predict(x)
        mu = params.get("mean")
        pi = params.get("zero_inflation")

        poisson_loglikelihood = y * torch.log(mu + 1e-10) - mu - torch.lgamma(y + 1)
        return torch.log(
            pi * (y == 0) + (1 - pi) * torch.exp(poisson_loglikelihood) + 1e-10
        )

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        mu, pi, u = self._local_params(x, u)
        y = poisson(mu).ppf(u)
        delta = bernoulli(1 - pi).ppf(u)
        return torch.from_numpy(y * delta).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
        """Return uniformized pseudo-observations for counts y given covariates x."""
        # cdf values using scipy's parameterization
        mu, pi, y = self._local_params(x, y)
        nb_distn = poisson(mu)
        u1 = pi + (1 - pi) * nb_distn.cdf(y)
        u2 = np.where(y > 0, pi + (1 - pi) * nb_distn.cdf(y-1), 0)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None) -> Tuple:
        params = self.predict(x)
        mu = params.get('mean')
        pi = params.get('zero_inflation')
        if y is None:
            return _to_numpy(mu, pi)
        return _to_numpy(mu, pi, y)