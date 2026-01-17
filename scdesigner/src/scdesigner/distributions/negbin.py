from ..base.marginal import GLMPredictor, Marginal
from ..data.formula import standardize_formula
from ..data.loader import _to_numpy
from ..utils.kwargs import _filter_kwargs, DEFAULT_ALLOWED_KWARGS
from .negbin_irls_funs import initialize_parameters
from scipy.stats import nbinom
from typing import Union, Dict, Optional, Tuple
import numpy as np
import torch


class NegBin(Marginal):
    """Negative-binomial marginal estimator with poisson initialization

    This subclass behaves like `Marginal` but assumes each gene follows a
    negative binomial distribution NB(mu_j(x), r_j(x)) parameterized via a mean
    `mu_j(x)` and dispersion `r_j(x)` that depend on covariates `x` through the
    provided `formula` object.

    The allowed formula keys are 'mean' and 'dispersion', defaulting to
    'mean' with a fixed dispersion if only a string formula is passed in.

    Examples
    --------
    >>> from scdesigner.distributions import NegBin
    >>> from scdesigner.datasets import pancreas
    >>>
    >>> sim = NegBin(formula={"mean": "~ bs(pseudotime, df=5)", "dispersion": "~ pseudotime"})
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
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion'])
        super().__init__(formula, **kwargs)

    def setup_optimizer(
            self,
            optimizer_class: Optional[callable] = torch.optim.AdamW,
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
            optimizer_kwargs=optimizer_kwargs,
            device=self.device
        )

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood"""
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

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor],
               r_min: float = 1e-3, r_max: float = 1e3, mu_min: float = 1e-5,
               mu_max: float = 1e4) -> torch.Tensor:
        mu, r, u = self._local_params(x, u)
        r = np.clip(r, r_min, r_max)
        mu = np.clip(mu, mu_min, mu_max)

        p = r / (r + mu)
        y = nbinom(n=r, p=p).ppf(u)
        return torch.from_numpy(y).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor],
                   epsilon: float =1e-6, r_min: float = 1e-3, r_max: float = 1e3,
                   mu_min: float = 1e-5, mu_max: float = 1e5) -> torch.Tensor:
        # extractl ocal parameters
        mu, r, y = self._local_params(x, y)
        r = np.clip(r, r_min, r_max)
        mu = np.clip(mu, mu_min, mu_max)
        p = r / (r + mu)

        # generate associated quantiles
        u1 = nbinom(n=r, p=p).cdf(y)
        u2 = np.where(y > 0, nbinom(n=r, p=p).cdf(y - 1), 0.0)
        v = np.random.uniform(size=y.shape)
        u = np.clip(v * u1 + (1.0 - v) * u2, epsilon, 1.0 - epsilon)
        return torch.from_numpy(u).float()

    def _local_params(self, x, y=None) -> Tuple:
        params = self.predict(x)
        mu = params.get('mean')
        r = params.get('dispersion')
        if y is None:
            return _to_numpy(mu, r)
        return _to_numpy(mu, r, y)

    def fit(self, max_epochs: int = 100, verbose: bool = True, **kwargs):
        if self.predict is None:
                self.setup_optimizer(**kwargs)

        # initialize using a poisson fit
        initialize_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['initialize'])
        beta_init, gamma_init = initialize_parameters(
            self.loader, self.n_outcomes, self.feature_dims['mean'],
            self.feature_dims['dispersion'],
            **initialize_kwargs
        )
        with torch.no_grad():
            self.predict.coefs['mean'].copy_(beta_init)
            self.predict.coefs['dispersion'].copy_(gamma_init)

        return Marginal.fit(self, max_epochs, verbose, **kwargs)