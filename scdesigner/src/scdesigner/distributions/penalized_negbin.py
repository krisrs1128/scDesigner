import numpy as np
import torch
from scdesigner.distributions import NegBin
from scdesigner.base.marginal import GLMPredictor


class PenalizedNegBin(NegBin):
    """NegBin with smoothness penalties on both mean and dispersion spatial coefficients."""

    def __init__(self, formula, mean_penalty_diag=None, disp_penalty_diag=None,
                 n_parametric=0, n_parametric_disp=0, lam=1.0, lam_disp=1.0,
                 gene_means=None, **kwargs):
        super().__init__(formula, **kwargs)
        self._mean_penalty_diag = mean_penalty_diag
        self._disp_penalty_diag = disp_penalty_diag
        self._n_parametric = n_parametric
        self._n_parametric_disp = n_parametric_disp
        self._lam = lam
        self._lam_disp = lam_disp
        self._gene_means = gene_means

    def setup_optimizer(self, optimizer_class=torch.optim.AdamW, **optimizer_kwargs):
        if self.loader is None:
            raise RuntimeError("self.loader is not set (call setup_data first)")

        # Structured penalty diagonals
        pen_mean = (torch.tensor(self._mean_penalty_diag, dtype=torch.float32, device=self.device)
                    if self._mean_penalty_diag is not None else None)
        pen_disp = (torch.tensor(self._disp_penalty_diag, dtype=torch.float32, device=self.device)
                    if self._disp_penalty_diag is not None else None)

        # Per-gene adaptive weights: 1 / mean_j
        if self._gene_means is not None:
            gm = np.array(self._gene_means, dtype=np.float64)
            gm = np.maximum(gm, 0.01)
            gene_weights = 1.0 / gm
            gene_w = torch.tensor(gene_weights, dtype=torch.float32, device=self.device)
        else:
            gene_w = None

        n_par = self._n_parametric
        n_par_d = self._n_parametric_disp
        lam = self._lam
        lam_d = self._lam_disp

        def nll(batch):
            ll = -self.likelihood(batch).sum()

            # Mean penalty: lam * (1/mean_j) * Σ_k d_k β_kj²
            if pen_mean is not None:
                beta = self.predict.coefs["mean"]
                beta_spatial = beta[n_par:, :]
                weighted = pen_mean[:, None] * beta_spatial ** 2
                if gene_w is not None:
                    weighted = weighted * gene_w[None, :]
                ll = ll + lam * weighted.sum()

            # Dispersion penalty: lam_disp * (1/mean_j) * Σ_k d_k γ_kj²
            if pen_disp is not None:
                gamma = self.predict.coefs["dispersion"]
                gamma_spatial = gamma[n_par_d:, :]
                weighted_d = pen_disp[:, None] * gamma_spatial ** 2
                if gene_w is not None:
                    weighted_d = weighted_d * gene_w[None, :]
                ll = ll + lam_d * weighted_d.sum()

            return ll

        #optimizer_kwargs["weight_decay"] = 0.01
        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            loss_fn=nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )