"""Public fitted-output contracts for composed NB simulators."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from scdesigner.copulas import StandardCopula
from scdesigner.data import GPBasis, SpatialTerm
from scdesigner.distributions import NegBin
from scdesigner.distributions.penalties import QuadraticPenalty
from scdesigner.simulators import PenalizedNegBinCopula, SpatialNegBinCopula, spatial_copula

FIT = dict(max_epochs=2, val_frac=0, batch_size=10, max_iter=1, verbose=False)


def simulator(**kwargs):
    return SpatialNegBinCopula(**{
        "mean_df": 3, "n_landmarks": 8, "estimator": "oas", "device": "cpu", **kwargs})


def fit(sim, data):
    np.random.seed(0)
    torch.manual_seed(0)
    sim.fit(data, **FIT)
    return sim


def assert_predictions_equal(first, second):
    assert set(first) == set(second) == {"mean", "dispersion"}
    for name in first:
        np.testing.assert_allclose(first[name], second[name], rtol=1e-5, atol=1e-6)


def test_default_model_predicts_spatial_mean_and_constant_dispersion(spatial_counts):
    sim = fit(SpatialNegBinCopula(device="cpu", estimator="oas"), spatial_counts)
    prediction = sim.predict()
    assert prediction["mean"].shape == prediction["dispersion"].shape == spatial_counts.shape
    assert np.isfinite(prediction["mean"]).all() and (prediction["mean"] > 0).all()
    assert np.ptp(prediction["mean"][:, 0]) > 1e-4
    np.testing.assert_allclose(prediction["dispersion"],
                               np.repeat(prediction["dispersion"][:1], 40, axis=0))


def test_convenience_and_explicit_recipe_predict_the_same_values(spatial_counts):
    term = SpatialTerm(GPBasis(df=3, n_landmarks=8, standardize=True), QuadraticPenalty())
    explicit = spatial_copula(
        NegBin({"mean": "~ 1", "dispersion": "~ 1"}, device="cpu"),
        {"mean": term}, copula=StandardCopula("~ 1", estimator="oas"),
        cap_at_observed_max=True,
    )
    wrapper = fit(simulator(), spatial_counts)
    fit(explicit, spatial_counts)
    assert_predictions_equal(wrapper.predict(), explicit.predict())


def test_nonspatial_dispersion_ignores_coordinate_changes(spatial_counts):
    sim = fit(simulator(dispersion_formula="~ batch"), spatial_counts)
    shifted = spatial_counts.obs.copy()
    shifted[["spatial1", "spatial2"]] += 0.25
    np.testing.assert_allclose(sim.predict()["dispersion"],
                               sim.predict(shifted)["dispersion"])
    assert len(sim.parameters["marginal"]["dispersion"]) == 2


def test_spatial_dispersion_can_vary_with_coordinates(spatial_counts):
    sim = fit(simulator(disp_df=2, lam_disp=0), spatial_counts)
    shifted = spatial_counts.obs.copy()
    shifted[["spatial1", "spatial2"]] += 0.25
    original = sim.predict()["dispersion"]
    changed = sim.predict(shifted)["dispersion"]
    assert np.isfinite(changed).all() and (changed > 0).all()
    assert not np.allclose(original, changed)


def test_tps_recipe_returns_counts_with_original_metadata(spatial_counts):
    data = spatial_counts.copy()
    data.obs = data.obs.rename(columns={"spatial1": "x", "spatial2": "y"})
    original_obs, original_values = data.obs.copy(deep=True), data.X.copy()
    sim = fit(simulator(basis="tps", spatial_cols=("x", "y"),
                        copula_formula="~ batch", top_k=2), data)
    sample = sim.sample()
    assert sample.shape == data.shape
    assert np.isfinite(sample.X).all() and (sample.X >= 0).all()
    np.testing.assert_array_equal(sample.X, np.floor(sample.X))
    pd.testing.assert_frame_equal(sample.obs, original_obs)
    pd.testing.assert_frame_equal(sample.var, data.var)
    pd.testing.assert_frame_equal(data.obs, original_obs)
    np.testing.assert_array_equal(data.X, original_values)


def test_observed_maximum_caps_only_generated_counts(spatial_counts):
    data = spatial_counts.copy()
    data.X[:] = 1
    samples = []
    for cap in (False, True):
        sim = fit(PenalizedNegBinCopula(cap_at_observed_max=cap,
                                        estimator="oas", device="cpu"), data)
        np.random.seed(11)
        samples.append(sim.sample().X)
    uncapped, capped = samples
    assert uncapped.max() > 1
    np.testing.assert_array_equal(capped, np.minimum(uncapped, data.X.max(axis=0)))


def test_prediction_does_not_depend_on_other_query_coordinates(spatial_counts):
    sim = fit(simulator(), spatial_counts)
    query = spatial_counts.obs.iloc[:10].copy()
    query[["spatial1", "spatial2"]] += 0.25
    others = spatial_counts.obs.iloc[10:20].copy()
    others[["spatial1", "spatial2"]] += 2
    alone = sim.predict(query, batch_size=7)
    together = sim.predict(pd.concat([query, others]), batch_size=20)
    assert_predictions_equal(alone, {key: value[:10] for key, value in together.items()})


def test_complexity_reports_finite_metrics_and_consistent_marginal_likelihood(spatial_counts):
    sim = fit(simulator(), spatial_counts)
    template = sim.complexity()
    external = sim.complexity(spatial_counts.copy())
    assert set(template) == set(external) == {
        "marginal_aic", "marginal_bic", "copula_aic", "copula_bic"}
    assert all(np.isfinite(value) for value in [*template.values(), *external.values()])
    for key in ("marginal_aic", "marginal_bic"):
        np.testing.assert_allclose(template[key], external[key])


def test_refit_matches_fresh_fit_on_reordered_features(spatial_counts):
    reused = fit(simulator(), spatial_counts)
    changed = spatial_counts[:, [2, 0]].copy()
    changed.X *= 2
    changed.obs[["spatial1", "spatial2"]] += 0.25
    fit(reused, changed)
    fresh = fit(simulator(), changed)
    assert_predictions_equal(reused.predict(), fresh.predict())
    sample = reused.sample()
    assert list(sample.var_names) == list(changed.var_names)
    assert (sample.X <= changed.X.max(axis=0)).all()


def test_dense_and_sparse_inputs_give_the_same_predictions(spatial_counts):
    sparse = spatial_counts.copy()
    sparse.X = sp.csr_matrix(sparse.X)
    dense_sim, sparse_sim = fit(simulator(), spatial_counts), fit(simulator(), sparse)
    assert_predictions_equal(dense_sim.predict(), sparse_sim.predict())


def test_default_spatial_term_is_a_scale_free_gp_with_100_columns():
    basis = SpatialNegBinCopula(device="cpu").observation_transform.terms["mean"].basis
    assert type(basis) is GPBasis
    assert basis.df == 100
    assert basis.length_scale is None


def test_spatial_fit_starts_from_the_intercept_only_fit(spatial_counts):
    sim = simulator()
    np.random.seed(0)
    torch.manual_seed(0)
    sim.fit(spatial_counts, max_epochs=0, val_frac=0, batch_size=10, verbose=False)
    spatial = sim.parameters["marginal"]["mean"].loc[[f"sp_mean_{k}" for k in range(3)]]
    np.testing.assert_array_equal(spatial.to_numpy(), 0)
