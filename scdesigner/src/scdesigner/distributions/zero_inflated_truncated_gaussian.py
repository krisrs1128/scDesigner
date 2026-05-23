from ..data.formula import standardize_formula
from ..base.marginal import GLMPredictor, Marginal
from ..data.loader import _to_numpy
from ..distributions.zero_inflated_truncated_gaussian_funs import _initialize_zitg_intercepts
from typing import Union, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import norm, truncnorm


class ZeroInflatedTruncatedGaussian(Marginal):
    """Zero-Inflated Truncated Gaussian marginal estimator.

    For each feature j, the distribution is a mixture:
      - with probability π_j(x), Y_j = 0 (point mass),
      - with probability 1-π_j(x), Y_j ~ N(μ_j(x), σ_j(x)^2) truncated to (0, ∞).

    This yields a density on {0} ∪ (0, ∞). The `mean` formula models the latent
    Gaussian mean μ_j(x), not the marginal mean.

    Allowed formula keys: `mean`, `sdev`, `zero_inflation`. A string formula is
    treated as the `mean` formula.

    Examples
    --------
    >>> from scdesigner.distributions import ZeroInflatedTruncatedGaussian
    >>> import scdesigner.datasets
    >>>
    >>> pancreas = scdesigner.datasets.pancreas()
    >>>
    >>> sim = ZeroInflatedTruncatedGaussian(formula={
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
    def __init__(self, formula: Union[Dict, str], prior_alpha: float = 1.1, **kwargs):
        """
        Parameters
        ----------
        formula : Union[Dict, str]
            Model formulas for 'mean', 'sdev', and 'zero_inflation'.
        prior_alpha : float, default=1.1
            Alpha for Beta(alpha, alpha) prior on zero-inflation probabilities.
            Values > 1 discourage extreme π, improving generalization for sparse genes.
        **kwargs
            Additional arguments passed to the Marginal base class.
        """
        formula = standardize_formula(
            formula, allowed_keys=['mean', 'sdev', 'zero_inflation']
        )
        self.prior_alpha = prior_alpha
        # Remove prior_alpha from kwargs before passing to parent
        kwargs.pop('prior_alpha', None)
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
            ll = -self.likelihood(batch).sum()

            # negative Beta(alpha, alpha) prior on zero-inflation probabilities
            pi = self.predict(batch[1]).get("zero_inflation")
            beta_penalty = (1 - self.prior_alpha) * (
                torch.log(pi.clamp(min=1e-10)) +
                torch.log((1 - pi).clamp(min=1e-10))
            ).sum()

            # -(likelihood + prior)
            return ll + beta_penalty

        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            link_fns=link_fns,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )

        mean_init, log_sdev_init, logit_pi_init = _initialize_zitg_intercepts(
            self._active_train_loader(), self.n_outcomes
        )
        self.predict.coefs["mean"].data[0].copy_(mean_init)
        self.predict.coefs["sdev"].data[0].copy_(log_sdev_init)
        self.predict.coefs["zero_inflation"].data[0].copy_(logit_pi_init)

    def likelihood(self, batch) -> torch.Tensor:
        """Compute the log-likelihood.

        For y == 0: log(pi).
        For y > 0: log(1 - pi) + log N(y; mu, sigma) - log Phi(mu / sigma),
        where the last term is the log-normalizer of the truncated normal on
        (0, inf). Computed via the standard-normal CDF; for nonnegative
        data mu / sigma should stay in a numerically benign range.
        """
        y, x = batch
        params = self.predict(x)
        mu = params.get("mean")
        sigma = params.get("sdev")
        pi = params.get("zero_inflation")

        # zero inflation component
        log_pi = torch.log(pi.clamp(min=1e-10))
        log_1mpi = torch.log((1 - pi).clamp(min=1e-10))

        # truncated gaussian component
        gaussian_logpdf = -0.5 * (torch.log(2 * torch.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))
        std_normal = torch.distributions.Normal(0.0, 1.0)
        log_trunc_norm = torch.log(std_normal.cdf(mu / sigma).clamp(min=1e-30))  # log Phi(mu / sigma)
        tn_logpdf = gaussian_logpdf - log_trunc_norm

        # combine
        return torch.where((y == 0), log_pi, log_1mpi + tn_logpdf)

    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Invert pseudoobservations.

        u <= pi maps to the atom at zero; u > pi maps through the truncated
        normal quantile function via `scipy.stats.truncnorm`.
        """
        mu, sdev, pi, u = self._local_params(x, u)
        q = np.clip((u - pi) / (1 - pi + 1e-10), 1e-6, 1 - 1e-6)
        a = -mu / sdev

        # invert using these truncated normal parameters
        y_pos = truncnorm.ppf(q, a=a, b=np.inf, loc=mu, scale=sdev)
        delta = (u > pi).astype(float)
        return torch.from_numpy(y_pos * delta).float()

    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor], epsilon=1e-6) -> torch.Tensor:
        """Return uniformized pseudo-observations for y given covariates x.

        The CDF is 0 for y < 0, equals pi at y = 0 (atom), and
        pi + (1 - pi) * F_TN(y) for y > 0, where F_TN is the CDF of a normal
        N(mu, sigma^2) truncated to (0, inf). Values y == 0 are randomized
        uniformly within the atom [0, pi]; values y > 0 are deterministic.
        """
        mu, sdev, pi, y = self._local_params(x, y)
        Phi_neg = norm.cdf(0.0, loc=mu, scale=sdev)  # = Phi(-mu / sigma)
        denom = 1.0 - Phi_neg                         # = Phi(mu / sigma)

        gsn = (norm.cdf(y, loc=mu, scale=sdev) - Phi_neg) / (denom + 1e-12)
        F_tn = np.where(y > 0, gsn, 0.0)

        # right and left CDF limits at y; equal for y > 0, separated by the
        # atom of width pi at y == 0.
        u1 = pi + (1 - pi) * F_tn
        u2 = np.where(y > 0, u1, 0.0)
        v = np.random.uniform(size=y.shape)

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
