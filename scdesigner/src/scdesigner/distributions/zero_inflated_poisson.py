from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from ..distributions.negbin_irls_funs import initialize_parameters
from ..distributions.zero_inflated_poisson_funs import _initialize_zi_intercept, _lam_to_mu
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import poisson

class ZeroInflatedPoisson(Marginal):
    """Zero-Inflated Poisson marginal estimator

    Feature j's zero inflation probability is `pi_j(x)`.  If not zero-ed, the
    draw is Poi(mu_j(x)). The 'mean' formula models the marginal mean
    λ_j(x) = (1−π_j)μ_j. During likelihood calculation, μ_j is set to
    λ_j/(1−π_j).

    The allowed formula keys are 'mean' and 'zero_inflation'. If a string
    formula is supplied it is taken to specify the `mean` by default.

    Examples
    --------
    >>> from scdesigner.distributions import ZeroInflatedPoisson
    >>> import scdesigner.datasets
    >>>
    >>> pancreas = scdesigner.datasets.pancreas()
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
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )

        beta, _ = initialize_parameters(
            self._active_train_loader(),
            self.n_outcomes,
            self.feature_dims["mean"],
            p_disp=1,
        )
        self.predict.coefs["mean"].data.copy_(beta)

        logit_pi = _initialize_zi_intercept(
            self._active_train_loader(), beta, self.n_outcomes
        )
        self.predict.coefs["zero_inflation"].data[0].copy_(logit_pi)

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood"""
        y, x = batch
        params = self.predict(x)
        lam = params.get("mean")
        pi = params.get("zero_inflation")
        mu = _lam_to_mu(lam, pi)

        poisson_loglikelihood = y * torch.log(mu + 1e-10) - mu - torch.lgamma(y + 1)
        return torch.log(
            pi * (y == 0) + (1 - pi) * torch.exp(poisson_loglikelihood) + 1e-10
        )

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        mu, pi, u = self._local_params(x, u)
        conditional_u = np.where(u > pi, (u - pi) / (1 - pi + 1e-10), 0.0)
        y = poisson(mu).ppf(conditional_u)
        delta = (u > pi).astype(float)
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
        lam = params.get('mean')
        pi = params.get('zero_inflation')
        mu = _lam_to_mu(lam, pi)
        if y is None:
            return _to_numpy(mu, pi)
        return _to_numpy(mu, pi, y)
