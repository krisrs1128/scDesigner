"""Fitted-output contracts for penalized and spatial ZITG recipes."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.stats import truncnorm

from scdesigner import datasets
from scdesigner.copulas import StandardCopula
from scdesigner.data import GPBasis, SpatialTerm
from scdesigner.distributions import (
    PenalizedZeroInflatedTruncatedGaussian,
    ZeroInflatedTruncatedGaussian,
)
from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty
from scdesigner.simulators import (
    PenalizedZeroInflatedTruncatedGaussianCopula,
    SpatialZeroInflatedTruncatedGaussianCopula,
    ZeroInflatedTruncatedGaussianCopula,
    spatial_copula,
)

FIT = dict(max_epochs=2, val_frac=0, batch_size=10, verbose=False)
FORMULA = {"mean": "~ 1", "sdev": "~ 1", "zero_inflation": "~ 1"}


@pytest.fixture
def spatial_expression():
    """40 cells x 3 genes; ZITG with latent mean 2 + spatial1 * (1, 1.5, 2), sdev 0.5, pi 0.3."""
    rng = np.random.default_rng(41)
    coords = rng.uniform(-1, 1, size=(40, 2))
    mean = 2 + coords[:, :1] * np.array([1.0, 1.5, 2.0])
    values = truncnorm.rvs(-mean / 0.5, np.inf, loc=mean, scale=0.5, random_state=rng)
    values[rng.random(values.shape) < 0.3] = 0
    obs = pd.DataFrame(
        {"spatial1": coords[:, 0], "spatial2": coords[:, 1], "batch": np.resize(["a", "b"], 40)},
        index=[f"cell{i}" for i in range(40)],
    )
    return AnnData(X=values, obs=obs, var=pd.DataFrame(index=[f"gene{j}" for j in range(3)]))


def simulator(**kwargs):
    return SpatialZeroInflatedTruncatedGaussianCopula(**{
        "mean_df": 3, "zi_df": 2, "n_landmarks": 8, "estimator": "oas", "device": "cpu", **kwargs})


def fit(sim, data):
    np.random.seed(0)
    torch.manual_seed(0)
    sim.fit(data, **FIT)
    return sim


def test_default_model_has_spatial_mean_and_zero_inflation_and_constant_sdev(spatial_expression):
    sim = SpatialZeroInflatedTruncatedGaussianCopula(estimator="oas", device="cpu")
    terms = sim.observation_transform.terms
    assert {name: (type(term.basis), term.basis.df, term.penalty)
            for name, term in terms.items()} == {
        "mean": (GPBasis, 100, QuadraticPenalty()),
        "zero_inflation": (GPBasis, 15, QuadraticPenalty()),
    }
    prediction = fit(sim, spatial_expression).predict()
    assert set(prediction) == set(FORMULA)
    assert all(value.shape == spatial_expression.shape for value in prediction.values())
    # initialization alone gives ranges near 0.003 (mean) and 0.0004 (zero inflation)
    assert np.ptp(prediction["mean"][:, 0]) > 0.1
    assert np.ptp(prediction["zero_inflation"][:, 0]) > 0.01
    np.testing.assert_allclose(prediction["sdev"], np.repeat(prediction["sdev"][:1], 40, axis=0))


def test_convenience_and_explicit_recipe_predict_the_same_values(spatial_expression):
    def term(df):
        return SpatialTerm(GPBasis(df=df, n_landmarks=8, standardize=True), QuadraticPenalty())

    explicit = spatial_copula(
        ZeroInflatedTruncatedGaussian(FORMULA.copy(), prior_alpha=2.0, device="cpu"),
        {"mean": term(3), "zero_inflation": term(2)},
        copula=StandardCopula("~ 1", estimator="oas"),
    )
    wrapper = fit(simulator(prior_alpha=2.0), spatial_expression)
    fit(explicit, spatial_expression)
    first, second = wrapper.predict(), explicit.predict()
    assert set(first) == set(second) == set(FORMULA)
    for name in FORMULA:
        np.testing.assert_allclose(first[name], second[name], rtol=1e-5, atol=1e-6)


def test_spatial_sdev_and_zero_inflation_use_their_own_terms(spatial_expression):
    sim = fit(simulator(sdev_df=2, lam_sdev=0.0, lam_zi=0.25), spatial_expression)
    assert {spec.parameter: spec.penalty.lam for spec in sim.marginal.penalties} == {
        "mean": 1.0, "sdev": 0.0, "zero_inflation": 0.25}
    assert len(sim.parameters["marginal"]["sdev"]) == 3
    shifted = spatial_expression.obs.copy()
    shifted[["spatial1", "spatial2"]] += 0.25
    original, changed = sim.predict(), sim.predict(shifted)
    assert (changed["sdev"] > 0).all()
    assert ((changed["zero_inflation"] > 0) & (changed["zero_inflation"] < 1)).all()
    for name in ("sdev", "zero_inflation"):
        assert not np.allclose(original[name], changed[name])


def test_tps_recipe_returns_nonnegative_values_with_original_metadata(spatial_expression):
    data = spatial_expression.copy()
    data.obs = data.obs.rename(columns={"spatial1": "x", "spatial2": "y"})
    original_obs, original_values = data.obs.copy(deep=True), data.X.copy()
    sim = fit(simulator(basis="tps", zi_df=None, spatial_cols=("x", "y"),
                        copula_formula="~ batch", top_k=2), data)
    assert list(sim.parameters["marginal"]["zero_inflation"].index) == ["Intercept"]
    sample = sim.sample()
    assert sample.shape == data.shape
    assert np.isfinite(sample.X).all() and (sample.X >= 0).all()
    assert (sample.X == 0).any() and (sample.X > 0).any()
    pd.testing.assert_frame_equal(sample.obs, original_obs)
    pd.testing.assert_frame_equal(sample.var, data.var)
    pd.testing.assert_frame_equal(data.obs, original_obs)
    np.testing.assert_array_equal(data.X, original_values)


def test_spatial_recipe_improves_held_out_fit_on_scgemrna():
    data = datasets.scgemrna()
    spatial = SpatialZeroInflatedTruncatedGaussianCopula(
        spatial_cols=("UMAP1_integrated", "UMAP2_integrated"), estimator="oas", device="cpu")
    nonspatial = ZeroInflatedTruncatedGaussianCopula(estimator="oas", device="cpu")
    for sim in (spatial, nonspatial):
        np.random.seed(0)
        torch.manual_seed(0)
        sim.fit(data, verbose=False)
    # held-out NLL per entry is about 1.30 spatial and 1.55 nonspatial
    assert spatial.marginal.best_validation_loss < nonspatial.marginal.best_validation_loss - 0.1
    np.random.seed(0)
    sample = spatial.sample()
    assert sample.shape == data.shape
    assert np.isfinite(sample.X).all() and (sample.X >= 0).all()
    assert abs((sample.X == 0).mean() - (data.X == 0).mean()) < 0.05
    pd.testing.assert_frame_equal(sample.obs, data.obs)


def fit_marginal(model, data):
    torch.manual_seed(0)
    model.setup_data(data, batch_size=40, device="cpu")
    model.fit(max_epochs=60, val_frac=0, lr=0.03, verbose=False)
    return model


def penalty(lam):
    return CoefficientPenalty("mean", ["spatial1"], [1.], QuadraticPenalty(lam=lam))


def test_disabled_penalty_matches_ordinary_fitted_model(spatial_expression):
    formula = {**FORMULA, "mean": "~ spatial1"}
    ordinary = fit_marginal(
        ZeroInflatedTruncatedGaussian(formula.copy(), prior_alpha=2.0, device="cpu"),
        spatial_expression)
    penalized = fit_marginal(PenalizedZeroInflatedTruncatedGaussian(
        formula.copy(), prior_alpha=2.0, penalties=[penalty(0)], device="cpu"), spatial_expression)
    assert set(penalized.parameters) == set(FORMULA)
    for name in FORMULA:
        np.testing.assert_allclose(penalized.parameters[name], ordinary.parameters[name],
                                   rtol=1e-6, atol=1e-7)


def test_regularization_reduces_fitted_covariate_effect(spatial_expression):
    formula = {**FORMULA, "mean": "~ spatial1"}
    ordinary = fit_marginal(
        ZeroInflatedTruncatedGaussian(formula.copy(), device="cpu"), spatial_expression)
    penalized = fit_marginal(PenalizedZeroInflatedTruncatedGaussian(
        formula.copy(), penalties=[penalty(10000)], device="cpu"), spatial_expression)
    ordinary_effect = np.linalg.norm(ordinary.parameters["mean"].loc["spatial1"])
    penalized_effect = np.linalg.norm(penalized.parameters["mean"].loc["spatial1"])
    assert penalized_effect < ordinary_effect / 2
    for name in ("sdev", "zero_inflation"):
        assert np.isfinite(penalized.parameters[name].to_numpy()).all()


def test_observed_maximum_caps_only_generated_values(spatial_expression):
    spec = penalty(1.0)
    samples = []
    for cap in (False, True):
        sim = fit(PenalizedZeroInflatedTruncatedGaussianCopula(
            "~ spatial1", penalties=[spec], prior_alpha=2.0, cap_at_observed_max=cap,
            estimator="oas", device="cpu"), spatial_expression)
        assert sim.marginal.penalties == (spec,) and sim.marginal.prior_alpha == 2.0
        np.random.seed(0)
        samples.append(sim.sample().X)
    uncapped, capped = samples
    maximum = spatial_expression.X.max(axis=0)
    assert (uncapped > maximum).any() and (uncapped >= 0).all()
    np.testing.assert_array_equal(capped, np.minimum(uncapped, maximum))
