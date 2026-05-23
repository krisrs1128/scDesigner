import numpy as np
import torch
from scdesigner.distributions import NegBin
from scdesigner.base.marginal import GLMPredictor


def count_parametric(predictor_names, basis_cols):
    """Count non-basis columns in a design matrix."""
    basis_set = set(basis_cols)
    return sum(1 for name in predictor_names if name not in basis_set)


class PenalizedNegBin(NegBin):
    """NegBin with quadratic penalties on a subset of mean and dispersion coefficients.

    Augments the negative-binomial log-likelihood with two quadratic penalties:

        lam * Σ_j (1/μ_j) Σ_k  d_k  β_kj²   (mean)
        lam_disp * Σ_j (1/μ_j) Σ_k  d_k  γ_kj²   (dispersion)

    where d_k = ``mean_penalty_diag[k]`` (resp. ``disp_penalty_diag[k]``) are
    per-basis-column weights (e.g. eigenvalues of a graph Laplacian), β_kj and
    γ_kj are the k-th basis coefficients for gene j, and 1/μ_j is a per-gene
    weight that normalises across genes with very different expression levels.
    Columns whose names appear in ``mean_basis_cols`` / ``disp_basis_cols`` are
    penalised; all others (intercept, cell type, etc.) are left unpenalised.

    Parameters
    ----------
    formula : str
        Formulaic formula passed to the parent :class:`NegBin`.
    mean_penalty_diag : array-like of shape (n_pen_mean,) or None
        Per-basis-column penalty weights for the mean linear predictor.
        ``None`` disables the mean penalty.
    disp_penalty_diag : array-like of shape (n_pen_disp,) or None
        Per-basis-column penalty weights for the dispersion linear predictor.
        ``None`` disables the dispersion penalty.
    mean_basis_cols : list of str
        Names of the penalised basis columns in the mean design matrix
        (e.g. ``["sp_basis_0", "sp_basis_1", ...]``).
    disp_basis_cols : list of str
        Names of the penalised basis columns in the dispersion design matrix.
    lam : float
        Global penalty strength for mean basis coefficients.
    lam_disp : float
        Global penalty strength for dispersion basis coefficients.
    gene_means : array-like of shape (n_genes,) or None
        Per-gene mean expression used to compute adaptive weights 1/μ_j.
        Values are floored at 0.01 before inversion. ``None`` disables
        adaptive weighting (all genes treated equally).
    **kwargs
        Additional keyword arguments forwarded to :class:`NegBin`.

    Examples
    --------
    >>> import numpy as np
    >>> penalty = np.array([1.0, 2.0, 4.0])
    >>> model = PenalizedNegBin(
    ...     "~ 0 + sp_basis_0 + sp_basis_1 + sp_basis_2",
    ...     mean_penalty_diag=penalty,
    ...     mean_basis_cols=["sp_basis_0", "sp_basis_1", "sp_basis_2"],
    ...     lam=0.1,
    ... )
    """

    def __init__(self, formula, mean_penalty_diag=None, disp_penalty_diag=None,
                 mean_basis_cols=None, disp_basis_cols=None, lam=1.0, lam_disp=1.0,
                 gene_means=None, **kwargs):
        super().__init__(formula, **kwargs)
        self._mean_penalty_diag = mean_penalty_diag
        self._disp_penalty_diag = disp_penalty_diag
        self._mean_basis_cols = mean_basis_cols or []
        self._disp_basis_cols = disp_basis_cols or []
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

        n_par = count_parametric(self.predictor_names.get("mean", []), self._mean_basis_cols)
        n_par_d = count_parametric(self.predictor_names.get("dispersion", []), self._disp_basis_cols)
        lam = self._lam
        lam_d = self._lam_disp

        def penalized_nll(batch):
            ll = -self.likelihood(batch).sum()

            # Mean penalty: lam * (1/mean_j) * Σ_k d_k β_kj²
            if pen_mean is not None:
                beta = self.predict.coefs["mean"]
                beta_pen = beta[n_par:, :]
                weighted = pen_mean[:, None] * beta_pen ** 2
                if gene_w is not None:
                    weighted = weighted * gene_w[None, :]
                ll = ll + lam * weighted.sum()

            # Dispersion penalty: lam_disp * (1/mean_j) * Σ_k d_k γ_kj²
            if pen_disp is not None:
                gamma = self.predict.coefs["dispersion"]
                gamma_pen = gamma[n_par_d:, :]
                weighted_d = pen_disp[:, None] * gamma_pen ** 2
                if gene_w is not None:
                    weighted_d = weighted_d * gene_w[None, :]
                ll = ll + lam_d * weighted_d.sum()

            return ll

        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            loss_fn=penalized_nll,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )
