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


def initialized_effect(model, data, **kwargs):
    """Coefficient of ``s`` straight after initialization (no optimization steps)."""
    torch.manual_seed(0)
    model.setup_data(data, batch_size=40, device="cpu")
    model.fit(max_epochs=0, val_frac=0, verbose=False, **kwargs)
    return model.parameters["mean"].loc["s"].to_numpy()


def test_penalized_model_starts_from_the_intercept_only_fit():
    data = counts()
    formula = {"mean": "~ s", "dispersion": "~ 1"}
    ordinary = initialized_effect(NegBin(formula.copy(), device="cpu"), data)
    penalized = initialized_effect(
        PenalizedNegBin(formula.copy(), penalties=[penalty(1)], device="cpu"), data)
    assert np.abs(ordinary).max() > 0.5
    np.testing.assert_array_equal(penalized, 0)


def test_explicit_max_iter_overrides_the_penalized_default():
    penalized = initialized_effect(
        PenalizedNegBin({"mean": "~ s", "dispersion": "~ 1"}, penalties=[penalty(1)], device="cpu"),
        counts(), max_iter=10)
    assert np.abs(penalized).max() > 0.5
