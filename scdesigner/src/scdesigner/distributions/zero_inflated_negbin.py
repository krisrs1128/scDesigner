from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from ..distributions.negbin_irls_funs import initialize_parameters
from ..distributions.zero_inflated_negbin_funs import _initialize_zi_intercept_nb, _lam_to_mu
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import nbinom

class ZeroInflatedNegBin(Marginal):
    """Zero-inflated negative-binomial marginal estimator

    Feature j's zero inflation probability is `pi_j(x)`.  If not zero-ed, the
    draw is NB(mu_j(x), r_j(x)). The 'mean' formula models the marginal mean
    λ_j(x) = (1−π_j)μ_j. During likelihood calculation, μ_j is set to
    λ_j/(1−π_j).

    The allowed formula keys are 'mean', 'dispersion', and 'zero_inflation'.

    Examples
    --------
    >>> from scdesigner.distributions import ZeroInflatedNegBin
    >>> import scdesigner.datasets
    >>>
    >>> pancreas = scdesigner.datasets.pancreas()
    >>>
    >>> sim = ZeroInflatedNegBin(formula={"mean": "~ pseudotime", "dispersion": "~ 1", "zero_inflation": "~ pseudotime"})
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
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion', 'zero_inflation'])
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
            "dispersion": torch.exp,
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

    def _initialize_parameters(self, **kwargs):
        beta, gamma = initialize_parameters(
            self._active_train_loader(),
            self.n_outcomes,
            self.feature_dims["mean"],
            self.feature_dims["dispersion"],
        )
        self.predict.coefs["mean"].data.copy_(beta)
        self.predict.coefs["dispersion"].data.copy_(gamma)

        logit_pi = _initialize_zi_intercept_nb(
            self._active_train_loader(), beta, gamma, self.n_outcomes
        )
        self.predict.coefs["zero_inflation"].data[0].copy_(logit_pi)

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the negative log-likelihood"""
        y, x = batch
        params = self.predict(x)
        lam = params.get("mean")
        r = params.get("dispersion")
        pi = params.get("zero_inflation")
        mu = _lam_to_mu(lam, pi)

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

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations."""
        mu, r, pi, u = self._local_params(x, u)
        conditional_u = np.where(u > pi, (u - pi) / (1 - pi + 1e-10), 0.0)
        y = nbinom(n=r, p=r / (r + mu)).ppf(conditional_u)
        delta = (u > pi).astype(float)
        return torch.from_numpy(y * delta).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
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

    def _local_params(self, x, y=None) -> Tuple:
        params = self.predict(x)
        lam = params.get('mean')
        r = params.get('dispersion')
        pi = params.get('zero_inflation')
        mu = _lam_to_mu(lam, pi)
        if y is None:
            return _to_numpy(mu, pi, r)
        return _to_numpy(mu, r, pi, y)
