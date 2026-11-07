import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.distributions import Bernoulli, Poisson
from scdesigner.simulators import CompositeCopula, NegBinCopula, ZeroInflatedPoissonCopula


def make_adata():
    rng = np.random.default_rng(0)
    return AnnData(
        X=rng.poisson(2.0, size=(200, 5)).astype(float),
        obs=pd.DataFrame({"g": rng.normal(size=200)}),
    )


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def test_stray_device_kwarg_is_ignored():
    sim = NegBinCopula(mean_formula="~ g", device="cpu")

    sim.fit(
        make_adata(),
        device="meta",
        max_epochs=2,
        batch_size=64,
        verbose=False,
        val_frac=0,
    )
    sample = sim.sample(device="meta")

    assert sim.marginal.device.type == "cpu"
    assert sample.shape == (200, 5)


def test_to_moves_predictor_and_loader_data():
    sim = NegBinCopula(device="cpu")
    assert sim.to("cpu") is sim

    sim.fit(make_adata(), max_epochs=2, batch_size=64, verbose=False, val_frac=0)
    sim.to("meta")

    predictor = sim.marginal.predict
    assert all(tensor.device.type == "meta" for tensor in predictor.parameters())
    assert all(tensor.device.type == "meta" for tensor in predictor.buffers())
    assert all(
        tensor.device.type == "meta"
        for state in predictor.optimizer.state.values()
        for tensor in _tensors(state)
    )
    assert sim.marginal.device.type == "meta"
    assert sim.marginal.loader.dataset.y.device.type == "meta"
    assert sim.copula.loader.dataset.y.device.type == "meta"


def test_composite_to_moves_all_marginals():
    sim = CompositeCopula(
        [
            (["g0"], Poisson("~ 1", device="cpu")),
            (["g1"], Bernoulli("~ 1", device="cpu")),
        ]
    )

    assert sim.to("meta") is sim
    assert all(marginal.device.type == "meta" for _, marginal in sim.marginals)


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="requires CUDA or MPS",
)
def test_constructor_device_cpu_on_gpu_machine():
    sim = ZeroInflatedPoissonCopula(
        mean_formula="~ g",
        zero_inflation_formula="~ 1",
        device="cpu",
    )

    sim.fit(make_adata(), max_epochs=2, batch_size=64, verbose=False, val_frac=0)
    sim.sample()

    assert next(iter(sim.marginal.loader))[0].device.type == "cpu"
