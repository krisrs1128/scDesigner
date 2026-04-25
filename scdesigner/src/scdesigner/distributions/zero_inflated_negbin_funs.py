from ..distributions.negbin_irls_funs import _is_intercept_column
import torch

# Shared with zero_inflated_poisson_funs — see comments there.
_LOGIT_PI_CLAMP = (-6, 4)
_MAX_INIT_BATCHES = 10


def _lam_to_mu(lam, pi, eps=1e-10):
    """mu = lam / (1 - pi); eps prevents division by zero"""
    return lam / (1 - pi + eps)


def _initialize_zi_intercept_nb(loader, beta_mean, gamma_disp, n_genes):
    """Return per-gene logit(pi) intercepts estimated from a short data scan.

    NB has expected zero fraction P(Y=0 | mu, r) = (r/(r+mu))^r. Besides this,
    uses the same logic as zero_inflated_poisson_funs.py
    """
    zero_counts = torch.zeros(n_genes)
    expected_zero_sum = torch.zeros(n_genes)
    n_cells = 0

    for i, (y_batch, x_batch) in enumerate(loader):
        if i >= _MAX_INIT_BATCHES:
            break
        if i == 0 and not _is_intercept_column(x_batch["zero_inflation"].to("cpu")[:, 0]):
            return torch.zeros(n_genes)

        mu_hat = torch.exp(x_batch["mean"].to("cpu") @ beta_mean)
        r_hat = torch.exp(x_batch["dispersion"].to("cpu") @ gamma_disp)
        zero_counts += (y_batch.to("cpu") == 0).float().sum(dim=0)
        expected_zero_sum += (r_hat / (r_hat + mu_hat)).pow(r_hat).sum(dim=0)
        n_cells += y_batch.shape[0]

    obs_zero_frac = zero_counts / n_cells
    expected_zero_frac = expected_zero_sum / n_cells
    pi_excess = (obs_zero_frac - expected_zero_frac).clamp(1e-6, 1 - 1e-6)
    return torch.log(pi_excess / (1 - pi_excess)).clamp(*_LOGIT_PI_CLAMP)
