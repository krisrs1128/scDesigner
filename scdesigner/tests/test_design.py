"""Designs fitted once on the complete table agree across chunks and queries."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.data.design import code_levels, fit_design
from scdesigner.simulators import NegBinCopula

CHUNK = 7
SUPPORTED = [
    "~ x", "x + ct", "~ ct", "~ C(ct)", "~ C(batch)", "~ log(z)", "~ bs(x, df=4)",
    "~ bs(x, df=5, degree=2)", "~ 0 + ct", "~ -1 + cat + x", "~ x * ct",
    "~ bs(x, df=4):ct", "~ C(batch) + log(z):cat",
    "~  bs(x, df=6,lower_bound=2, extrapolation='zero')  + bs(z, df=4, lower_bound=2, extrapolation='zero')",
    "~ cr(x, df=4)", "~ cr(x, df=4):ct",
]


@pytest.fixture
def obs():
    """60 rows ordered by ``x``; ``ct`` levels b > c > a by frequency; ``batch`` in runs."""
    rng = np.random.default_rng(3)
    n = 60
    return pd.DataFrame(
        {
            "x": np.sort(rng.uniform(0, 10, n)),
            "z": rng.uniform(1, 5, n),
            "ct": rng.choice(np.array(list("abc"), dtype=object), n, p=[0.2, 0.5, 0.3]),
            "cat": pd.Categorical(rng.choice(["u", "v"], n), categories=["v", "u"]),
            "batch": np.repeat([3, 1, 2, 4], n // 4),
        },
        index=[f"cell{i}" for i in range(n)],
    )


def chunked_matrix(design, obs, key="k"):
    parts = [design.matrices(obs.iloc[s:s + CHUNK])[key] for s in range(0, len(obs), CHUNK)]
    return pd.concat(parts)


@pytest.mark.parametrize("formula", SUPPORTED)
@pytest.mark.parametrize("partition", [False, True])
def test_chunked_designs_match_full_table(obs, formula, partition):
    keys = {"k"} if partition else set()
    full = fit_design(obs, {"k": formula}, partition_keys=keys).matrices(obs)["k"]
    chunked = fit_design(obs, {"k": formula}, partition_keys=keys, chunked=True)
    result = chunked_matrix(chunked, obs)
    assert list(result.columns) == list(full.columns) == chunked.predictor_names["k"]
    np.testing.assert_allclose(result.to_numpy(), full.to_numpy(), rtol=0, atol=1e-12)


def test_regression_designs_drop_the_most_common_level(obs):
    design = fit_design(obs, {"mean": "~ ct", "free": "~ 0 + ct", "group": "~ 0 + ct"},
                        partition_keys={"group"}, chunked=True)
    assert design.predictor_names == {
        "mean": ["Intercept", "ct[T.c]", "ct[T.a]"],
        "free": ["ct[c]", "ct[a]"],
        "group": ["ct[b]", "ct[c]", "ct[a]"],
    }


@pytest.mark.parametrize("chunked", [False, True])
def test_queries_use_fit_time_state(obs, chunked):
    design = fit_design(obs, {"k": "~ ct + C(batch) + bs(x, df=4)"}, chunked=chunked)
    full = design.matrices(obs)["k"]

    # Level a is absent and c is most common; x covers only part of its range.
    query = obs[(obs.ct != "a") & (obs.x < 5)]
    query = pd.concat([query[query.ct == "c"], query[query.ct == "b"].iloc[:1]])
    np.testing.assert_array_equal(design.matrices(query)["k"].to_numpy(),
                                  full.loc[query.index].to_numpy())

    for column, value in [("ct", "d"), ("batch", 9)]:
        unseen = obs.iloc[:3].copy()
        unseen[column] = unseen[column].astype(object)
        unseen.iloc[1, unseen.columns.get_loc(column)] = value
        with pytest.raises(ValueError, match="null values"):
            design.matrices(unseen)


@pytest.mark.parametrize("chunked", [False, True])
def test_missing_covariates_raise(obs, chunked):
    design = fit_design(obs, {"k": "~ x + ct"}, chunked=chunked)
    for column in ("x", "ct"):
        missing = obs.iloc[:3].copy()
        missing.iloc[1, missing.columns.get_loc(column)] = None
        with pytest.raises(ValueError, match="null values"):
            design.matrices(missing)


@pytest.mark.parametrize("formula", [
    "~ {x - x.mean()}", "~ log(z + 1)", "~ bs(x, knots=[5.0])", "~ exp(z)",
])
def test_chunked_fit_rejects_unsupported_formulas(obs, formula):
    fit_design(obs, {"k": formula})
    with pytest.raises(ValueError, match="Chunked fitting does not support"):
        fit_design(obs, {"k": formula}, chunked=True)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("partition, columns, expected", [
    (False, ["ct[c]", "ct[a]"], [[1, 0], [0, 0]]),
    (True, ["ct[b]", "ct[c]", "ct[a]"], [[0, 1, 0], [1, 0, 0]]),
])
def test_fit_formula_retains_reference_and_query_levels(obs, chunked, partition, columns, expected):
    from scdesigner.data.design import fit_formula

    design = fit_formula(
        obs, "~ 0 + ct", rows=obs.iloc[:1] if chunked else obs,
        column_levels={}, partition=partition, chunked=chunked,
    )
    query = pd.DataFrame({"ct": ["c", "b"]}, index=["query1", "query2"])
    result = design.matrix(query)
    assert design.predictor_names == columns
    assert list(result.columns) == columns
    assert result.index.equals(query.index)
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_code_levels_casts_unseen_values_without_changing_input():
    query = pd.DataFrame({"ct": ["a", "new"], "x": [1, 2]})
    original = query.copy(deep=True)
    dtype = pd.CategoricalDtype(["b", "a"])

    result = code_levels(query, {"ct": dtype})

    assert result.ct.dtype == dtype
    assert result.ct.iloc[0] == "a"
    assert pd.isna(result.ct.iloc[1])
    pd.testing.assert_series_equal(result.x, query.x)
    pd.testing.assert_frame_equal(query, original)


@pytest.mark.parametrize("chunked", [False, True])
def test_formula_keys_keep_independent_fitted_state(obs, chunked):
    design = fit_design(obs, {
        "mean": "~ ct + bs(x, df=4)",
        "batch": "~ 0 + C(batch)",
        "group": "~ 0 + ct",
    }, partition_keys={"group"}, chunked=chunked)
    assert design.predictor_names == {
        "mean": ["Intercept", "ct[T.c]", "ct[T.a]",
                 *[f"bs(x, df=4)[{i}]" for i in range(1, 5)]],
        "batch": ["C(batch)[1]", "C(batch)[2]", "C(batch)[3]", "C(batch)[4]"],
        "group": ["ct[b]", "ct[c]", "ct[a]"],
    }
    full = design.matrices(obs)
    query = obs[(obs.ct != "a") & (obs.x < 5)]
    result = design.matrices(query)
    for key in full:
        pd.testing.assert_frame_equal(result[key], full[key].loc[query.index])

    # Fit-time numeric levels are sorted, although batch 3 occurs first.
    np.testing.assert_array_equal(full["batch"].iloc[0].to_numpy(), [0, 0, 1, 0])


def test_eager_fit_accepts_expressions_beyond_simple_calls(obs):
    design = fit_design(obs, {"k": "~ log(z + 1)"})
    query = obs.iloc[[3, 1, 5]]
    result = design.matrices(query)["k"]
    assert list(result.columns) == ["Intercept", "log(z + 1)"]
    np.testing.assert_allclose(result["log(z + 1)"], np.log(query.z + 1))


@pytest.mark.parametrize("formula, expected", [
    ("~ cell_type", "~ -1 + cell_type"),
    ("~ 1", "~ 1"),
    ("~ 0", "~ 1"),
    ("~ 0 + cell_type", "~ -1 + cell_type"),
])
def test_partition_formula_normalization(formula, expected):
    from scdesigner.data.formula import partition_formula

    assert partition_formula(formula) == expected


def test_partition_formula_produces_full_indicators(obs):
    from scdesigner.data.formula import partition_formula

    design = fit_design(
        obs, {"group": partition_formula("~ ct")}, partition_keys={"group"}, chunked=True,
    )
    result = design.matrices(pd.DataFrame({"ct": ["a", "b", "c"]}))["group"]
    assert list(result.columns) == ["ct[b]", "ct[c]", "ct[a]"]
    np.testing.assert_array_equal(result.to_numpy(), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_predictions_do_not_depend_on_other_query_rows(obs):
    rng = np.random.default_rng(0)
    rate = np.exp(0.5 + np.sin(obs.x.to_numpy()) + (obs.ct == "c").to_numpy())
    data = AnnData(X=rng.poisson(np.tile(rate[:, None], (1, 3))).astype(float), obs=obs)
    sim = NegBinCopula(mean_formula="~ ct + bs(x, df=4)", copula_formula="~ ct")
    sim.marginal.device = torch.device("cpu")
    torch.manual_seed(0)
    sim.fit(data, max_epochs=2, val_frac=0, verbose=False, device="cpu")

    query = obs[(obs.ct != "a") & (obs.x < 5)]
    rows = obs.index.get_indexer(query.index)
    whole, part = sim.predict(device="cpu"), sim.predict(query, device="cpu")
    for name in whole:
        np.testing.assert_allclose(part[name], whole[name][rows], rtol=1e-6)
    np.random.seed(0)
    assert sim.sample(query, device="cpu").shape == (len(query), 3)
