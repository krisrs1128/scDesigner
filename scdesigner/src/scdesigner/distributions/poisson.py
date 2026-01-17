from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import poisson

class Poisson(Marginal):
    """Poisson marginal estimator

    This subclass behaves like `Marginal` but assumes each gene follows a
    Poisson distribution with mean `mu_j(x)` that depends on covariates `x`
    via the provided `formula` object.

    The allowed formula keys are 'mean', defaulting to a single `mean` term
    if a string formula is supplied.

    Examples
    --------
    >>> from scdesigner.distributions import Poisson
    >>> from scdesigner.datasets import pancreas
    >>>
    >>> sim = Poisson(formula="~ bs(pseudotime, df=5)")
    >>> sim.setup_data(pancreas)
    >>> sim.fit(max_epochs=1, verbose=False)
    >>>
    >>> # evaluate p(y | x) and mu(x)
    >>> y, x = next(iter(sim.loader))
    >>> l = sim.likelihood((y, x))
    >>> y_hat = sim.predict(x)
    >>>
    >>> # convert to quantiles and back
    >>> u = sim.uniformize(y, x)
    >>> x_star = sim.invert(u, x)
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        formula = standardize_formula(formula, allowed_keys=['mean'])
        super().__init__(formula, **kwargs)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        def nll(batch):
            return -self.likelihood(batch).sum()

        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood"""
        y, x = batch
        params = self.predict(x)
        mu = params.get("mean")
        return y * torch.log(mu) - mu - torch.lgamma(y + 1)

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        mu, u = self._local_params(x, u)
        y = poisson(mu).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
        """Return uniformized pseudo-observations for counts y given covariates x."""
        # cdf values using scipy's parameterization
        mu, y = self._local_params(x, y)
        u1 = poisson(mu).cdf(y)
        u2 = np.where(y > 0, poisson(mu).cdf(y - 1), 0)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        params = self.predict(x)
        mu = params.get('mean')
        if y is None:
            return _to_numpy(mu)
        return _to_numpy(mu, y)