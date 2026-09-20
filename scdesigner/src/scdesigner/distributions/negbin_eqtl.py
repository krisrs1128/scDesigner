import warnings
from contextlib import contextmanager
from typing import Dict, List, Union
from anndata import AnnData
import numpy as np
import pandas as pd
import torch

from .donor_moments import POSTERIOR_MOMENTS
from .negbin import NegBin, nb_loglik
from ..base.marginal import EQTLPredictor, Marginal
from ..data.formula import standardize_formula


class NegBinEQTL(NegBin):
    """Negative-binomial marginal for single-cell eQTL data.

    This provides an alternative to the ordinary negative binomial marginal
    model that's suitable for single-cell eQTL studies. It supports donor-level
    random intercepts that are integrated out in an EM step. It also ensures
    that different SNPs can be used as genotype predictors for every response
    gene without having to store all SNPs in the same design matrix.

    We still use a Poisson model for the initialization, But we further use the
    method of moments to initialize random effects of specific hyperparameters.

    Parameters
    ----------
    formula : dict or str
        Dictionary giving the formula for "mean", "dispersion", and optionally
        "interaction". Interaction can be used to define cell-type specific
        genotype effects (though it's not limited to that, it can also be used
        for dynamic effects).
    donor_col : str
        Column name and `adata.obs` given the individual/donor ID
    dosage : pandas.DataFrame
        An n_donor x SNP table giving variant counts per donor. Donor order in
        all later estimates is determined by the row indices from this matrix.
    snp_map : dict[str, list[str]]
        Association between the gene names and `adata.var_names` to the column
        names of snps in dosage. This is used to understand which SNPs are
        potentially related to which genes.
    estimate_sigma2 : bool
        Whether to re-estimate the random intercept variances after each epoch.
        This is also used as the regularization strength lam_j.
    em_warmup_epochs : int
        How many epochs across which we fix `lam`. This is necessary because the
        first few epochs don't really give reliable estimates for U.
    sigma2_method : {"laplace", "edgeworth"}
        Strategy for estimating the posterior moments for the donor variance in
        the EM update See donor_moments.py for more details. Laplace is similar
        to what packages like glmmTMB implement, Edgeworth is a high-order
        correction that does well for sparse genes.
    mode_polish : bool
        After each epoch, should we apply a Newton step to improve estimation
        for U? This can accelerate convergence and lead to better donor variance
        estimates.
    sigma2_bounds : tuple of float
        Minimum and maximum values for `sigma_j^2` to use to stabilize estimates
        during the EM updates.
    donor_blocked : bool or None
        Should we first order the cells by donor rather than randomly sampling
        them? If sorted by donor, the data structure representing each batch is
        much more memory efficient.
    """

    def __init__(
        self,
        formula: Union[Dict, str],
        *,
        donor_col: str,
        dosage: pd.DataFrame,
        snp_map: Dict[str, List[str]],
        estimate_sigma2: bool = True,
        em_warmup_epochs: int = 3,
        sigma2_method: str = "laplace",
        sigma2_bounds: tuple = (1e-4, 10.0),
        mode_polish: bool = True,
        donor_blocked=None,
        **kwargs,
    ):
        has_interaction = isinstance(formula, dict) and "interaction" in formula
        formula = standardize_formula(
            formula, allowed_keys=["mean", "dispersion", "genotype", "interaction"]
        )
        if not has_interaction:
            del formula["interaction"]
        Marginal.__init__(self, formula, **kwargs)

        self.donor_col = donor_col
        self.dosage = dosage
        self.snp_map = snp_map
        self._estimate_sigma2 = estimate_sigma2
        self._em_warmup_epochs = em_warmup_epochs
        self._sigma2_method = sigma2_method
        self._sigma2_bounds = sigma2_bounds
        self._mode_polish = mode_polish
        self._donor_blocked = donor_blocked
        self.donor_order = list(dosage.index.astype(str))
        self.donor_to_code = {name: i for i, name in enumerate(self.donor_order)}
        self.D = None
        self.mask = None

    def _augment_obs(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `obs` with a donor ID column."""
        donor_names = obs[self.donor_col].astype(str)
        obs = obs.copy()
        obs["_donor_code"] = donor_names.map(self.donor_to_code).astype(float)
        return obs

    def _build_dosage_tensors(self, var_names: List[str]):
        """Gather per-gene SNP dosages into the (K, J, K_max) tensor + mask."""
        # Number of SNPs considered per gene.
        k_max = max((len(v) for v in self.snp_map.values()), default=1)
        k_max = max(k_max, 1)
        n_individuals, n_genes = len(self.donor_order), len(var_names)

        # Build the dosage matrix, gene by gene and snp by snp.
        D = torch.zeros(n_individuals, n_genes, k_max)
        mask = torch.zeros(n_genes, k_max)
        dosage = self.dosage.loc[self.donor_order]
        for j, gene in enumerate(var_names):
            snp_cols = self.snp_map.get(gene, [])
            if snp_cols:
                D[:, j, : len(snp_cols)] = torch.tensor(
                    dosage[snp_cols].values, dtype=torch.float32
                )
                mask[j, : len(snp_cols)] = 1.0
        return D, mask

    def setup_data(self, adata: AnnData, batch_size: int = 1024, **kwargs):
        new_obs = self._augment_obs(adata.obs)
        adata = AnnData(X=adata.X, obs=new_obs, var=adata.var)
        self.formula["genotype"] = "~ 0 + _donor_code"

        self.D, self.mask = self._build_dosage_tensors(list(adata.var_names))
        if self._donor_blocked is None:
            self._donor_blocked = self.D.shape[2] > 1

        # By sorting cells by donor, the dosage lookup gets simplified
        # during data loading.
        if self._donor_blocked:
            order = np.argsort(new_obs["_donor_code"].values, kind="stable")
            adata = adata[order].copy()

        # Make sure to match the device with the overall marginal model
        kwargs.setdefault("device", self.device)
        super().setup_data(adata, batch_size=batch_size, **kwargs)

    def _subset_loader(self, indices, training: bool):
        # Split the data loader into train and test loaders
        if self._donor_blocked:
            indices = np.sort(indices)
        return super()._subset_loader(indices, training)

    def setup_optimizer(self, optimizer_class=torch.optim.AdamW, **optimizer_kwargs):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        n_train = len(self.train_loader.dataset)

        def penalized_nll(batch):
            # [Explain: why one forward pass serves all three consumers.]
            y, x = batch
            params = self.predict(x)
            mu, r = params["mean"], params["dispersion"]
            nll = -nb_loglik(y, mu, r).sum()

            donor_idx = x["genotype"][:, 0].long()
            accumulators = self.predict.accumulators
            if accumulators:
                # The score is the derivative of the ordinary log likelihood,
                # while the higher order moments are derivatives of the negative
                # log likelihood.
                with torch.no_grad():
                    p = mu / (mu + r)
                    q = (y + r) * p * (1.0 - p)
                    score = r * (y - mu) / (r + mu)
                    accumulators["score"].index_add_(0, donor_idx, score)
                    accumulators["observed"].index_add_(0, donor_idx, q)
                    third = q * (1.0 - 2.0 * p)
                    accumulators["third"].index_add_(0, donor_idx, third)
                    if "fourth" in accumulators:
                        fourth = q * (1.0 - 6.0 * p + 6.0 * p.square())
                        accumulators["fourth"].index_add_(0, donor_idx, fourth)

            # rescale the penalty to the current batch
            penalty = 0.5 * (self.predict.lam * self.predict.U ** 2).sum()
            return nll + penalty * (y.shape[0] / n_train)

        # Summary statistics to accumulate for the sigma2 estimation strategy
        sigma2_accumulators = {
            "laplace": ("score", "observed", "third"),
            "edgeworth": ("score", "observed", "third", "fourth"),
        }
        accumulator_names = (
            sigma2_accumulators[self._sigma2_method]
            if self._estimate_sigma2
            else ()
        )
        self.predict = EQTLPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            dosage=self.D,
            mask=self.mask,
            interaction_dim=self.feature_dims.get("interaction", 0),
            accumulator_names=accumulator_names,
            loss_fn=penalized_nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )

    def _center_random_effects(self):
        """Center each gene's mean donor effect

        The intercept term and the donor effects are linearly dependent, though
        the donor effects are regularized (via the prior) to be closer to 0.
        This centering makes sure that shifts in U due to the intercept are not
        going to inflate sigma_j^2.
        """
        with torch.no_grad():
            if self.predictor_names["mean"][0] == "Intercept":
                offset = self.predict.U.mean(dim=0)
                self.predict.U.sub_(offset.unsqueeze(0))
                self.predict.coefs["mean"][0].add_(offset)

    def _em_update_sigma2(self) -> torch.Tensor:
        """Update the random-intercept variance and return posterior means.

        These are approximations to ``E[u_kj | y]`` and ``E[u_kj**2 | y]``,
        which are necessary for the EM update,

            sigma_j^2 = (1/K) sum_k E[u_kj**2 | y].

        See donor_moments.py for more details.
        """
        with torch.no_grad():
            accumulators = self.predict.accumulators
            info = (
                self.predict.lam.unsqueeze(0)
                + accumulators["observed"]
            )
            mean, m2 = POSTERIOR_MOMENTS[self._sigma2_method](
                self.predict.U,
                info,
                accumulators.get("third"),
                accumulators.get("fourth"),
            )

            # Only the edgeworth variance can ever be negative.
            invalid = (
                (m2 < mean.square()).sum()
                if self._sigma2_method == "edgeworth"
                else torch.zeros((), dtype=torch.long, device=m2.device)
            )
            self.predict.invalid_variances.copy_(invalid)

            sigma2 = m2.mean(dim=0).clamp(*self._sigma2_bounds)
            self.predict.lam.copy_(1.0 / sigma2)
            return mean

    def _newton_step(self) -> torch.Tensor:
        """Newton update of `U`

        We can prove the estimate of U after each epoch by taking a Newton step.
        The necessary information is already accumulated in the previous
        training data pass. The specific form of the update is,


            delta_kj = (score_kj - lam_j U_kj) / (observed_kj + lam_j)
        """
        info = self.predict.lam.unsqueeze(0) + self.predict.accumulators["observed"]
        return (
            self.predict.accumulators["score"]
            - self.predict.lam * self.predict.U
        ) / info

    def _polish_mode(self, delta: torch.Tensor):
        """Move `U` toward the conditional mode

        Writing, `A(u) = sum_i (y_i + r) p(1-p)`, it's possible to show that the
        curvature `-l'' = A + lam` satisfies,

            |A'| = |sum_i (y_i + r) p(1-p)(1-2p)| <= A
            =>  |d/du log(-l'')| = |A'| / (A + lam) <= 1

        because `|1 - 2p| < 1`. This bounds the change in curvature in a step of
        size t by `e^t`, so

            l'(u+t) >= (-l''(u)) [ delta - (e^t - 1) ]

        B y setting `t - log(1  + |delta|)`, we can ensure that `l` strictly
        increases. Therefore we choose step sizes,

            step = sign(delta) * log1p(|delta|)

        which are the largest steps guaranteed not to overshoot.
        """
        self.predict.U.add_(delta.sign() * delta.abs().log1p())

    def _on_epoch_end(self, epoch: int):
        if self._estimate_sigma2 and epoch > self._em_warmup_epochs:
            self._center_random_effects()
            with torch.no_grad():
                delta = self._newton_step()
                self.predict.mode_gap.copy_(delta.abs().amax(dim=0))
                self._em_update_sigma2()
                if self._mode_polish:
                    self._polish_mode(delta)
        self.predict.zero_accumulators()

    def _mom_sigma2(self) -> torch.Tensor:
        """Method-of-moments initialization for sigma_j^2.

        Using the delta method we can show,

            sigma_j^2 ~= [ Var_k(m_kj) - E_k(m_kj / n_k) ] / (E_k m_kj)^2
        """
        n_donors = len(self.donor_order)
        sums = torch.zeros(n_donors, self.n_outcomes, device=self.device)
        counts = torch.zeros(n_donors, device=self.device)

        with torch.no_grad():
            for batch in self.train_loader:
                y, x = self._move_batch_to_device(batch)
                donor_idx = x["genotype"][:, 0].long()
                sums.index_add_(0, donor_idx, y)
                counts.index_add_(0, donor_idx, torch.ones_like(donor_idx, dtype=sums.dtype))

        present = counts > 0
        m = sums[present] / counts[present].unsqueeze(1)
        n = counts[present].unsqueeze(1)

        between = m.var(dim=0, unbiased=True)
        within = (m / n).mean(dim=0)
        denom = m.mean(dim=0).clamp(min=1e-8) ** 2
        return ((between - within) / denom).clamp(*self._sigma2_bounds)

    def _mom_beta(self):
        """Initialize the genotype effect from the three dosage groups.

        The SNP dosages can only take on values {0, 1, 2}. We can estimate the
        associated genotype effects in a single pass through the data using the
        method of moments,

            log m_d = alpha + beta * d,   d in {0, 1, 2}

        This only estimate the dosage for d = {0}, but that's enough of a warm
        start.

        Returns
        -------
        beta : torch.Tensor
            (J,) effects across gene.
        mean_dose : torch.Tensor
            (J,) mean dosages per cell. This is just used during centering.
        """
        n = torch.zeros(3, self.n_outcomes, device=self.device)
        sum_y = torch.zeros(3, self.n_outcomes, device=self.device)
        dose_sum = torch.zeros(self.n_outcomes, device=self.device)
        n_cells = 0

        lead = self.predict.D[:, :, 0] * self.predict.mask[:, 0]

        with torch.no_grad():
            for batch in self.train_loader:
                y, x = self._move_batch_to_device(batch)
                dose = lead[x["genotype"][:, 0].long()]
                dose_sum += dose.sum(dim=0)
                n_cells += y.shape[0]
                for d in range(3):
                    in_group = (dose == d).to(y.dtype)
                    n[d] += in_group.sum(dim=0)
                    sum_y[d] += (y * in_group).sum(dim=0)

        # weighted least squares of group mean on dosage. The weights are the
        # group sizes.
        group_mean = sum_y / n.clamp(min=1.0)
        log_mean = torch.log(group_mean.clamp(min=1e-6))
        levels = torch.arange(3, device=self.device, dtype=n.dtype).unsqueeze(1)

        w = n
        sw = w.sum(dim=0).clamp(min=1e-8)
        mean_d = (w * levels).sum(dim=0) / sw
        mean_l = (w * log_mean).sum(dim=0) / sw
        cov = (w * (levels - mean_d) * (log_mean - mean_l)).sum(dim=0)
        var_d = (w * (levels - mean_d) ** 2).sum(dim=0)

        beta = torch.where(var_d > 1e-8, cov / var_d.clamp(min=1e-8),
                           torch.zeros_like(cov))
        return beta, dose_sum / max(n_cells, 1)

    def _initialize_parameters(self, **kwargs):
        super()._initialize_parameters(**kwargs)

        beta, mean_dose = self._mom_beta()
        with torch.no_grad():
            self.predict.beta[:, 0].copy_(beta)
            if self.predictor_names["mean"][0] == "Intercept":
                self.predict.coefs["mean"][0, :] -= beta * mean_dose

        sigma2 = self._mom_sigma2()
        with torch.no_grad():
            self.predict.lam.copy_(1.0 / sigma2.clamp(*self._sigma2_bounds))

    @property
    def sigma2(self) -> torch.Tensor:
        """Current per-gene random-intercept variance, `1 / lam`."""
        return (1.0 / self.predict.lam).detach()

    @contextmanager
    def simulated_donor_effects(self, seed=None):
        """Generate new donor effects `u ~ N(0, sigma_j^2)`

        If we don't use this method, then all simulation is based on the
        estimated `U` from the observed donors. If you want to imagine a brand
        new set cohort, then we should use this method. Especially important for
        power analysis.

        Args:
            seed: optional integer for a reproducible draw. With `None` the
                ambient torch RNG is used, so repeated blocks differ.
        """
        predict = self.predict
        previous_flag = predict.use_simulated_donor_effects
        previous_draw = predict.U_sim.clone()
        try:
            with torch.no_grad():
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=predict.U_sim.device)
                    generator.manual_seed(int(seed))
                noise = torch.randn(
                    predict.U_sim.shape,
                    generator=generator,
                    device=predict.U_sim.device,
                    dtype=predict.U_sim.dtype,
                )
                predict.U_sim.copy_(noise * self.sigma2.sqrt().unsqueeze(0))
            predict.use_simulated_donor_effects = True
            yield self
        finally:
            predict.use_simulated_donor_effects = previous_flag
            with torch.no_grad():
                predict.U_sim.copy_(previous_draw)

    @property
    def invalid_posterior_variance_count(self) -> int:
        """Donor-gene pairs with negative Edgeworth variances"""
        return int(self.predict.invalid_variances.item())

    @property
    def mode_gap(self) -> torch.Tensor:
        """Per-gene max `|Newton step|` for `U` at the last EM update, `(J,)`.

        This shows how far U was from the true posterior mode. This gets at the
        skewness in the posterior for sigma^2_j.
        """
        return self.predict.mode_gap.detach()
