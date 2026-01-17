from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import bernoulli

class Bernoulli(Marginal):
    """Bernoulli marginal estimator

    This subclass behaves like `Marginal` but assumes each feature follows a
    Bernoulli distribution with success probability `theta_j(x)` that depends
    on covariates `x` through the `formula` argument.

    The allowed formula keys are 'mean' (interpreted as the logit of the
    success probability when used with a GLM link). If a string formula is
    provided, it is taken to specify the `mean` model.

    Examples
    --------
    >>> from scdesigner.distributions import Bernoulli
    >>> from scdesigner.datasets import pancreas
    >>>
    >>> sim = Bernoulli(formula="~ pseudotime")
    >>> sim.setup_data(pancreas)
    >>> sim.fit(max_epochs=1, verbose=False)
    >>>
    >>> # evaluate p(y | x) and theta(x)
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
        super().__init__(formula)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        link_fns = {"mean": torch.sigmoid}
        def nll(batch):
            return -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_fns,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood"""
        y, x = batch
        params = self.predict(x)
        theta = params.get("mean")
        return y * torch.log(theta) + (1 - y) * torch.log(1 - theta)

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        theta, u = self._local_params(x, u)
        y = bernoulli(theta).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
        """Return uniformized pseudo-observations for counts y given covariates x."""
        theta, y = self._local_params(x, y)
        u1 =  bernoulli(theta).cdf(y)
        u2 = np.where(y > 0,  bernoulli(theta).cdf(y - 1), 0)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None) -> Tuple:
        params = self.predict(x)
        theta = params.get('mean')
        if y is None:
            return _to_numpy(theta)
        return _to_numpy(theta, y)
