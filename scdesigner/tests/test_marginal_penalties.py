"""Penalization changes fitted effects without changing reported likelihoods."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.distributions import NegBin, PenalizedNegBin
from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty


def counts():
    s = np.linspace(-1., 1., 40)
    rates = np.exp(1.5 * s[:, None]) * np.array([5., 8.])
    return AnnData(X=np.random.default_rng(7).poisson(rates).astype(float),
                   obs=pd.DataFrame({"s": s}, index=[f"cell{i}" for i in range(40)]))


def penalty(lam):
    return CoefficientPenalty("mean", ["s"], [1.], QuadraticPenalty(lam=lam))


def fit(model, data, **kwargs):
    torch.manual_seed(0)
    model.setup_data(data, batch_size=40, device="cpu")
    model.fit(max_epochs=60, val_frac=0, lr=0.03, verbose=False, **kwargs)
    return model


def test_disabled_penalty_matches_ordinary_fitted_model():
    data = counts()
    formula = {"mean": "~ s", "dispersion": "~ 1"}
    ordinary = fit(NegBin(formula.copy(), device="cpu"), data)
    penalized = fit(PenalizedNegBin(formula.copy(), penalties=[penalty(0)], device="cpu"), data)
    for name in ordinary.parameters:
        np.testing.assert_allclose(penalized.parameters[name], ordinary.parameters[name],
                                   rtol=1e-6, atol=1e-7)


def test_regularization_reduces_fitted_covariate_effect():
    data = counts()
    formula = {"mean": "~ s", "dispersion": "~ 1"}
    ordinary = fit(NegBin(formula.copy(), device="cpu"), data)
    penalized = fit(PenalizedNegBin(formula.copy(), penalties=[penalty(10000)], device="cpu"), data)
    ordinary_effect = np.linalg.norm(ordinary.parameters["mean"].loc["s"])
    penalized_effect = np.linalg.norm(penalized.parameters["mean"].loc["s"])
    assert penalized_effect < ordinary_effect / 2
    assert np.isfinite(penalized.parameters["dispersion"].to_numpy()).all()


def test_reported_validation_loss_is_unpenalized_likelihood():
    model = PenalizedNegBin("~ s", penalties=[penalty(100)], device="cpu")
    torch.manual_seed(0)
    model.setup_data(counts(), batch_size=10, device="cpu")
    model.fit(max_epochs=3, val_frac=0.25, verbose=False)
    loss, entries = 0., 0
    with torch.no_grad():
        for y, x in model.validation_loader:
            loss -= model.likelihood((y, x)).sum().item()
            entries += y.numel()
    assert model.best_validation_loss == pytest.approx(loss / entries)
