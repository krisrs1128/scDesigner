"""Zero-inflation initializers must work on whatever device the data lives on.

`5b4f925` made the Poisson IRLS initializer allocate `beta` on the device of the
incoming batch. The zero-inflation intercept scans that consume `beta` still
forced every batch onto the CPU, so on CUDA/MPS they multiply a CPU design
matrix by a GPU `beta`:

    RuntimeError: Expected all tensors to be on the same device, but found at
    least two devices, cpu and cuda:0!
"""
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.data.loader import adata_loader
from scdesigner.distributions import ZeroInflatedNegBin, ZeroInflatedPoisson
from scdesigner.distributions.negbin_irls_funs import initialize_parameters
from scdesigner.distributions.zero_inflated_negbin_funs import (
    _initialize_zi_intercept_nb as zinb_intercept,
)
from scdesigner.distributions.zero_inflated_poisson_funs import (
    _initialize_zi_intercept as zip_intercept,
)

MODELS = [
    (ZeroInflatedPoisson, ["mean", "zero_inflation"]),
    (ZeroInflatedNegBin, ["mean", "dispersion", "zero_inflation"]),
]


def gpu():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def make_adata(n_cells=200, n_genes=5):
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, size=(n_cells, n_genes)).astype("float32")
    counts[rng.random(counts.shape) < 0.3] = 0
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    return AnnData(X=counts, obs=obs)


@pytest.mark.skipif(gpu() is None, reason="requires CUDA or MPS")
@pytest.mark.parametrize("model, keys", MODELS)
def test_marginal_fits_on_gpu(model, keys):
    """The reported failure: fitting a zero-inflated marginal with data on a GPU."""
    device = gpu()
    marginal = model({key: "~ 1" for key in keys}, device=device)
    marginal.setup_data(make_adata(), batch_size=64, device=device)

    marginal.fit(max_epochs=2, verbose=False, val_frac=0)

    for coef in marginal.predict.coefs.values():
        assert torch.isfinite(coef).all()
        assert coef.device.type == device


def test_intercept_scans_do_not_move_batches_to_cpu(monkeypatch):
    """Runs without a GPU, where a forced `.to("cpu")` is harmless and so can't
    raise. Instead, record every explicit request to move a tensor to the CPU.
    """
    adata = make_adata()
    formula = {key: "~ 1" for key in ["mean", "dispersion", "zero_inflation"]}
    loader = adata_loader(adata, formula, batch_size=64, device="cpu")
    beta, gamma = initialize_parameters(loader, adata.n_vars, 1, 1, max_iter=2)

    to = torch.Tensor.to
    cpu_moves = []

    def record(tensor, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if str(target).startswith("cpu"):
            cpu_moves.append(target)
        return to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record)
    zip_intercept(loader, beta, adata.n_vars)
    zinb_intercept(loader, beta, gamma, adata.n_vars)

    assert not cpu_moves, f"scans moved tensors to the CPU {len(cpu_moves)} times"
