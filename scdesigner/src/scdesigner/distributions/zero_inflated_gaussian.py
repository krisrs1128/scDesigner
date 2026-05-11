from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from ..distributions.zero_inflated_gaussian_funs import _initialize_zig_intercepts
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import norm


class ZeroInflatedGaussian(Marginal):
    """Zero-Inflated Gaussian marginal estimator.

    Feature j has zero-inflation probability `pi_j(x)`. With probability
    `pi_j`, Y_j = 0 exactly; otherwise Y_j ~ N(mu_j(x), sigma_j(x)^2).
    Unlike the ZIP/ZINB classes, the `mean` formula models the Gaussian
    component mean directly (not the marginal mean), since the marginal-mean
    reparameterization mu = lambda / (1 - pi) interacts poorly with an
    identity link.

    The likelihood for y = 0 mixes a point mass with the Gaussian density
    (Tobit-style); this is improper in the strict measure-theoretic sense
    but is the standard convention.

    Allowed formula keys: `mean`, `sdev`, `zero_inflation`. A string formula
    is treated as the `mean` formula.

    Examples
    --------
    >>> from scdesigner.distributions import ZeroInflatedGaussian
    >>> import scdesigner.datasets
    >>>
    >>> pancreas = scdesigner.datasets.pancreas()
    >>>
    >>> sim = ZeroInflatedGaussian(formula={
    ...     "mean": "~ pseudotime",
    ...     "sdev": "~ pseudotime",
    ...     "zero_inflation": "~ pseudotime",
    ... })
    >>> sim.setup_data(pancreas)
    >>> sim.fit(max_epochs=1, verbose=False)
    >>>
    >>> y, x = next(iter(sim.loader))
    >>> l = sim.likelihood((y, x))
    >>> y_hat = sim.predict(x)
    >>>
    >>> u = sim.uniformize(y, x)
    >>> x_star = sim.invert(u, x)
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        formula = standardize_formula(
            formula, allowed_keys=['mean', 'sdev', 'zero_inflation']
        )
        super().__init__(formula, **kwargs)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.Adam,
            **optimizer_kwargs,
    ):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        link_fns = {
            "mean": lambda x: x,
            "sdev": torch.exp,
            "zero_inflation": torch.sigmoid,
        }
        def nll(batch):
            return -self.likelihood(batch).sum()
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_fns,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        mean_init, log_sdev_init, logit_pi_init = _initialize_zig_intercepts(
            self.loader, self.n_outcomes
        )
        self.predict.coefs["mean"].data[0].copy_(mean_init)
        self.predict.coefs["sdev"].data[0].copy_(log_sdev_init)
        self.predict.coefs["zero_inflation"].data[0].copy_(logit_pi_init)

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood."""
        y, x = batch
        params = self.predict(x)
        mu = params.get("mean")
        sigma = params.get("sdev")
        pi = params.get("zero_inflation")

        gaussian_logpdf = -0.5 * (
            torch.log(2 * torch.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2)
        )
        return torch.log(
            pi * (y == 0) + (1 - pi) * torch.exp(gaussian_logpdf) + 1e-10
        )

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        mu, sdev, pi, u = self._local_params(x, u)
        conditional_u = np.where(u > pi, (u - pi) / (1 - pi + 1e-10), 0.0)
        conditional_u = np.clip(conditional_u, 1e-6, 1 - 1e-6)
        y = norm(loc=mu, scale=sdev).ppf(conditional_u)
        delta = (u > pi).astype(float)
        return torch.from_numpy(y * delta).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
        """Return uniformized pseudo-observations for y given covariates x."""
        mu, sdev, pi, y = self._local_params(x, y)
        cdf = norm.cdf(y, loc=mu, scale=sdev)

        # right and left CDF limits, accounting for the atom at y = 0
        u1 = np.where(y >= 0, pi, 0.0) + (1 - pi) * cdf
        u2 = np.where(y > 0, pi, 0.0) + (1 - pi) * cdf
        v = np.random.uniform(size=y.shape)

        # combine u1 and u2 according to randomizer v
        u = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None) -> Tuple:
        params = self.predict(x)
        mu = params.get("mean")
        sdev = params.get("sdev")
        pi = params.get("zero_inflation")
        if y is None:
            return _to_numpy(mu, sdev, pi)
        return _to_numpy(mu, sdev, pi, y)
