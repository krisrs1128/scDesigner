from ..distributions.negbin_irls_funs import _is_intercept_column
import torch

# Clamp logit(pi) to avoid degenerate initialization: [-6, 4] → [0.25%, 98%]
# inflation. A sparse batch could otherwise push logit(pi) to the extremes.
_LOGIT_PI_CLAMP = (-6, 4)

# Scan at most this many batches to estimate zero fractions.
_MAX_INIT_BATCHES = 10


def _lam_to_mu(lam, pi, eps=1e-10):
    """mu = lam / (1 - pi); eps prevents division by zero if pi → 1 during optimization."""
    return lam / (1 - pi + eps)


def _initialize_zi_intercept(loader, beta_mean, n_genes):
    """Return per-gene logit(pi) intercepts estimated from a short data scan.

    Computes excess zeros (observed minus Poisson-expected, using beta_mean)
    over up to _MAX_INIT_BATCHES batches. Returns zeros if the zero_inflation
    design matrix has no intercept column.
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
        zero_counts += (y_batch.to("cpu") == 0).float().sum(dim=0)
        expected_zero_sum += torch.exp(-mu_hat).sum(dim=0)
        n_cells += y_batch.shape[0]

    obs_zero_frac = zero_counts / n_cells
    expected_zero_frac = expected_zero_sum / n_cells
    pi_excess = (obs_zero_frac - expected_zero_frac).clamp(1e-6, 1 - 1e-6)
    return torch.log(pi_excess / (1 - pi_excess)).clamp(*_LOGIT_PI_CLAMP)
