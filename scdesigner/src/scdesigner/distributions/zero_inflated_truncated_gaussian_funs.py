from ..data.formula import offset_key
from itertools import islice
import torch

# Clamp logit(pi) to avoid degenerate initialization: [-6, 4] -> [0.25%, 98%]
# inflation. A sparse batch could otherwise push logit(pi) to the extremes.
_LOGIT_PI_CLAMP = (-6, 4)

# Scan at most this many batches to estimate per-gene statistics.
_MAX_INIT_BATCHES = 10


def _offset(x, head, y):
    """Offset on a head's linear predictor, zeros of shape (n_cells, 1) if absent."""
    return x.get(offset_key(head), torch.zeros_like(y[:, :1]))


def _nonzero_moments(y, x):
    """Per-gene sums of w, w * r and w * r^2 over the nonzero cells of a batch.

    Subtracting the mean offset leaves r = y - a ~ Normal(mean, sd0^2 * exp(2b)).
    The weights w = exp(-2b) equalize the variances, and zeros get weight 0.
    """
    nonzero = (y != 0).to(y.dtype)
    residual = y - _offset(x, "mean", y)
    weights = nonzero * torch.exp(-2 * _offset(x, "sdev", y))
    return (weights.sum(dim=0),
            (weights * residual).sum(dim=0),
            (weights * residual.square()).sum(dim=0))


def _mean_and_log_sdev(s0, s1, s2, n_nonzero):
    """Weighted mean of r, and the log of the sd0 estimated from the weighted
    squared deviations w * (r - mean)^2, each of which has expectation sd0^2.
    """
    mean = s1 / torch.where(s0 > 0, s0, torch.ones_like(s0))
    squared_deviations = s2 - 2 * mean * s1 + mean.square() * s0
    variance = (squared_deviations / n_nonzero.clamp(min=1)).clamp(min=1e-6)
    return mean, 0.5 * variance.log()


def _logit_pi(n_zero, logit_offsets, n_steps=30):
    """Intercept t where the expected zero count, sum_i sigmoid(t + c_i), equals
    the observed one. Solved per gene by bisection over the clamp range.
    """
    low = torch.full_like(n_zero, _LOGIT_PI_CLAMP[0])
    high = torch.full_like(n_zero, _LOGIT_PI_CLAMP[1])
    for _ in range(n_steps):
        mid = (low + high) / 2
        expected = sum(torch.sigmoid(mid + c).sum(dim=0) for c in logit_offsets)
        below = expected < n_zero
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    return (low + high) / 2


def _initialize_zitg_intercepts(loader, n_genes):
    """Estimate per-gene intercepts for (mean, log-sdev, logit-pi) from a short scan.

    Uses moments of the nonzero observations (after removing offsets) as initial
    estimates for the truncated-normal parameters and the zero count for pi.
    Returns three (n_genes,) tensors, which only mean something for heads whose
    design starts with an intercept column.
    """
    device = next(iter(loader))[0].device
    n_nonzero = torch.zeros(n_genes, device=device)
    n_zero = torch.zeros_like(n_nonzero)
    s0, s1, s2 = (torch.zeros_like(n_nonzero) for _ in range(3))
    logit_offsets = []

    # quick scan to accumulate moments and offsets
    for y, x in islice(loader, _MAX_INIT_BATCHES):
        n_nonzero += (y != 0).sum(dim=0)
        n_zero += (y == 0).sum(dim=0)
        for total, batch_total in zip((s0, s1, s2), _nonzero_moments(y, x)):
            total += batch_total
        logit_offsets.append(_offset(x, "zero_inflation", y))

    # initialization for main parameters
    mean, log_sdev = _mean_and_log_sdev(s0, s1, s2, n_nonzero)
    return mean, log_sdev, _logit_pi(n_zero, logit_offsets)
