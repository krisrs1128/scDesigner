"""Public fit, prediction, and simulation contracts for NegBinEQTLCopula."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.stats import nbinom

from scdesigner.simulators import NegBinEQTLCopula

N_DONORS, N_GENES, CELLS_PER_DONOR = 12, 4, 15
DONORS = [f"SAMP{k}" for k in range(N_DONORS)]
GENES = [f"gene{j}" for j in range(N_GENES)]


def _data(snps_per_gene=1):
    rng = np.random.default_rng(0)
    snp_map = {
        gene: [f"snp{j}_{m}" for m in range(snps_per_gene)]
        for j, gene in enumerate(GENES)
    }
    snp_cols = [snp for snps in snp_map.values() for snp in snps]
    dosage = pd.DataFrame(
        rng.binomial(2, 0.3, size=(N_DONORS, len(snp_cols))),
        index=DONORS,
        columns=snp_cols,
    )
    donor_idx = np.repeat(np.arange(N_DONORS), CELLS_PER_DONOR)
    obs = pd.DataFrame(
        {
            "indiv": np.array(DONORS)[donor_idx],
            "cell_type": np.resize(["T", "B"], len(donor_idx)),
            "t": np.tile(np.linspace(0, 1, CELLS_PER_DONOR), N_DONORS),
        },
        index=[f"cell{i}" for i in range(len(donor_idx))],
    )
    u = rng.normal(0.0, 0.5, size=(N_DONORS, N_GENES))
    log_mu = np.log(5.0) + u[donor_idx]
    effect = 0.5 + 0.4 * (obs["cell_type"].to_numpy() == "T") + 0.3 * obs["t"].to_numpy()
    for j, gene in enumerate(GENES):
        log_mu[:, j] += dosage[snp_map[gene]].sum(axis=1).to_numpy()[donor_idx] * effect
    y = nbinom(n=5.0, p=5.0 / (5.0 + np.exp(log_mu))).rvs(random_state=rng)
    return AnnData(X=y.astype(float), obs=obs, var=pd.DataFrame(index=GENES)), dosage, snp_map


def _fit(adata, dosage, snp_map, mean_formula="~ genotype + (1 | indiv)"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        sim = NegBinEQTLCopula(
            mean_formula, dosage=dosage, snp_map=snp_map, device="cpu"
        )
        sim.fit(adata, batch_size=64, max_epochs=5, verbose=False, val_frac=0.0)
    return sim


@pytest.fixture
def fitted():
    adata, dosage, snp_map = _data()
    return _fit(adata, dosage, snp_map), adata


def test_fit_predicts_valid_parameters_without_changing_input():
    adata, dosage, snp_map = _data()
    original = adata.copy()
    sim = _fit(adata, dosage, snp_map)

    assert set(sim.parameters) >= {"marginal", "copula"}
    params = sim.predict(batch_size=32)
    for name in ("mean", "dispersion"):
        assert params[name].shape == adata.shape
        assert np.all(np.isfinite(params[name]))
        assert np.all(params[name] > 0)
    pd.testing.assert_frame_equal(adata.obs, original.obs)
    pd.testing.assert_frame_equal(adata.var, original.var)
    np.testing.assert_array_equal(adata.X, original.X)


def test_sample_returns_counts_with_cell_and_gene_metadata(fitted):
    sim, adata = fitted
    synthetic = sim.sample(batch_size=32)

    assert isinstance(synthetic, AnnData)
    assert synthetic.shape == adata.shape
    pd.testing.assert_frame_equal(synthetic.obs[adata.obs.columns], adata.obs)
    pd.testing.assert_frame_equal(synthetic.var, adata.var)
    assert np.all(np.isfinite(synthetic.X))
    assert np.all(synthetic.X >= 0)
    np.testing.assert_array_equal(synthetic.X, np.floor(synthetic.X))


@pytest.mark.parametrize("method", ["predict", "sample"])
def test_fresh_observations_are_supported_without_mutation(fitted, method):
    sim, adata = fitted
    fresh = adata.obs.iloc[[45, 2, 46, 3, 47]].copy()
    fresh.index = [f"new{i}" for i in range(len(fresh))]
    original = fresh.copy(deep=True)

    result = getattr(sim, method)(obs=fresh, batch_size=2)
    values = result["mean"] if method == "predict" else result.X
    assert values.shape == (len(fresh), N_GENES)
    assert np.all(np.isfinite(values))
    if method == "sample":
        pd.testing.assert_frame_equal(result.obs[fresh.columns], original)
    pd.testing.assert_frame_equal(fresh, original)


def test_predictions_follow_rows_regardless_of_donor_order_or_frequency(fitted):
    sim, adata = fitted
    reference = sim.predict(obs=adata.obs, batch_size=32)
    # A reordered subset with repeated donors and a different batch size.
    indices = [179, 0, 47, 0, 179, 46, 178, 1, 179]
    obs = adata.obs.iloc[indices].copy()
    obs.index = [f"new{i}" for i in range(len(obs))]
    reordered = sim.predict(obs=obs, batch_size=2)

    for name in ("mean", "dispersion"):
        np.testing.assert_allclose(reordered[name], reference[name][indices], rtol=1e-5)


@pytest.mark.parametrize("context", ["C(cell_type)", "bs(t, df=4)"], ids=["categorical", "spline"])
def test_interactions_allow_context_dependent_genotype_effects(context):
    adata, dosage, snp_map = _data()
    sim = _fit(adata, dosage, snp_map, f"~ genotype * {context} + (1 | indiv)")
    obs = pd.DataFrame(
        {
            "indiv": [DONORS[0], DONORS[0], DONORS[1], DONORS[1]],
            "cell_type": ["B", "T", "B", "T"],
            "t": [0.25, 0.75, 0.25, 0.75],
        },
        index=[f"new{i}" for i in range(4)],
    )
    log_mean = np.log(sim.predict(obs=obs, batch_size=2)["mean"])
    context_effect = log_mean[[1, 3]] - log_mean[[0, 2]]
    # Without a genotype interaction, the context effect is identical for both donors.
    assert np.any(np.abs(context_effect[0] - context_effect[1]) > 1e-5)
    assert sim.sample(obs=obs, batch_size=2).shape == (len(obs), N_GENES)


def test_multiple_snps_per_gene_support_prediction_and_sampling():
    adata, dosage, snp_map = _data(snps_per_gene=3)
    sim = _fit(adata, dosage, snp_map)
    mean = sim.predict(batch_size=32)["mean"]

    assert mean.shape == adata.shape
    assert np.all(np.isfinite(mean))
    assert np.all(mean > 0)
    assert sim.sample(batch_size=32).shape == adata.shape


def test_complexity_reports_finite_scores(fitted):
    sim, _ = fitted
    metrics = sim.complexity(batch_size=64)

    assert set(metrics) == {"marginal_aic", "marginal_bic", "copula_aic", "copula_bic"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_simulated_donor_effects_are_reproducible_and_restore_fitted_predictions(fitted):
    sim, _ = fitted
    fitted_mean = sim.predict(batch_size=32)["mean"]
    with sim.marginal.simulated_donor_effects(seed=12):
        first = sim.predict(batch_size=32)["mean"]
    with sim.marginal.simulated_donor_effects(seed=12):
        repeated = sim.predict(batch_size=32)["mean"]
    with sim.marginal.simulated_donor_effects(seed=13):
        different = sim.predict(batch_size=32)["mean"]

    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, fitted_mean)
    assert not np.allclose(first, different)
    np.testing.assert_array_equal(sim.predict(batch_size=32)["mean"], fitted_mean)


def test_sampling_rejects_unknown_donor(fitted):
    sim, adata = fitted
    stranger = adata.obs.iloc[:1].copy()
    stranger["indiv"] = "NOT_A_DONOR"
    with pytest.raises(ValueError):
        sim.sample(obs=stranger, batch_size=8)
