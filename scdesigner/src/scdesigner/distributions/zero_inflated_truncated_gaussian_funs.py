from ..distributions.negbin_irls_funs import _is_intercept_column
import torch

# Clamp logit(pi) to avoid degenerate initialization: [-6, 4] -> [0.25%, 98%]
# inflation. A sparse batch could otherwise push logit(pi) to the extremes.
_LOGIT_PI_CLAMP = (-6, 4)

# Scan at most this many batches to estimate per-gene statistics.
_MAX_INIT_BATCHES = 10


def _initialize_zitg_intercepts(loader, n_genes):
    """Estimate per-gene intercepts for (mean, log-sdev, logit-pi) from a short scan.

    Uses sample mean/variance of nonzero observations as initial estimates for the
    truncated-normal parameters and the empirical zero fraction for π. This
    approximation is biased when μ is small relative to σ. Returns three
    (n_genes,) tensors. If the first batch has no intercept column for a head,
    the corresponding intercept is left at zero.
    """
    sum_y = torch.zeros(n_genes)
    sum_y2 = torch.zeros(n_genes)
    n_nonzero = torch.zeros(n_genes)
    n_zero = torch.zeros(n_genes)

    has_intercept = {"mean": True, "sdev": True, "zero_inflation": True}
    for i, (y_batch, x_batch) in enumerate(loader):
        if i >= _MAX_INIT_BATCHES:
            break
        y = y_batch.to("cpu")
        if i == 0:
            for k in has_intercept:
                if k in x_batch:
                    has_intercept[k] = _is_intercept_column(x_batch[k].to("cpu")[:, 0])

        nz_mask = (y != 0).float()
        n_nonzero += nz_mask.sum(dim=0)
        n_zero += (1 - nz_mask).sum(dim=0)
        sum_y += (y * nz_mask).sum(dim=0)
        sum_y2 += ((y ** 2) * nz_mask).sum(dim=0)

    n_total = n_nonzero + n_zero
    nz_safe = n_nonzero.clamp(min=1)
    gene_mean = sum_y / nz_safe
    gene_var = (sum_y2 / nz_safe - gene_mean ** 2).clamp(min=1e-6)
    log_sdev = 0.5 * torch.log(gene_var)

    zero_frac = (n_zero / n_total.clamp(min=1)).clamp(1e-6, 1 - 1e-6)
    logit_pi = torch.log(zero_frac / (1 - zero_frac)).clamp(*_LOGIT_PI_CLAMP)

    if not has_intercept["mean"]:
        gene_mean = torch.zeros(n_genes)
    if not has_intercept["sdev"]:
        log_sdev = torch.zeros(n_genes)
    if not has_intercept["zero_inflation"]:
        logit_pi = torch.zeros(n_genes)

    return gene_mean, log_sdev, logit_pi
