"""Offset terms: formula parsing, design matrices, data loading, and model use."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData, read_h5ad
from formulaic import Formula

from scdesigner.base.marginal import EQTLPredictor, GLMPredictor
from scdesigner.data import formula
from scdesigner.data.design import fit_design
from scdesigner.data.loader import adata_loader
from scdesigner.distributions import NegBin, NegBinEQTL, Poisson
from scdesigner.distributions.negbin_irls_funs import initialize_parameters
from scdesigner.simulators import NegBinCopula, NegBinEQTLCopula, PoissonCopula


@pytest.fixture
def obs():
    return pd.DataFrame({"x": [1., 2., 4., 8., 16., 32.],
                         "lib": [1., 2., 4., 2., 1., 4.], "ct": ["a"] * 6},
                        index=[f"c{i}" for i in range(6)])


# Formula parsing

def test_split_offset_terms():
    cases = [  # text, remaining terms, offset-only formula
        ("~ x + offset(log(lib))", ["1", "x"], "~ 0 + {log(lib)}"),
        ("~ offset(x)", ["1"], "~ 0 + `x`"),
        ("~ 0 + offset(x)", [], "~ 0 + `x`"),
        ("~ 0 + offset(2 * x)", [], "~ 0 + {2 * x}"),
        ("~ x + offset(z) - offset(z)", ["1", "x"], None),
        ("~ x", ["1", "x"], None),
        ("~  0 + C(ct) : x", None, None),
    ]
    for text, terms, offsets in cases:
        fixed, actual = formula.split_offset_terms(text)
        assert actual == offsets
        if terms is not None:
            assert [str(t) for t in Formula(fixed)] == terms
        if offsets is None:
            assert fixed == text
    with pytest.raises(ValueError, match="At most one"):
        formula.split_offset_terms("~ offset(x) + offset(log(lib))")


# Design matrices

def test_design_matrices_values_and_metadata(obs):
    obs["with space"] = obs.x + obs.lib
    for chunked in (False, True):
        design = fit_design(obs, {"mean": "~ x + offset(log(lib))", "dispersion": "~ 1"}, chunked=chunked)
        matrices = design.matrices(obs)
        np.testing.assert_allclose(matrices["offset:mean"].iloc[:, 0], np.log(obs.lib))
        assert matrices["offset:mean"].shape == (6, 1)
        assert design.predictor_names == {"mean": ["Intercept", "x"], "dispersion": ["Intercept"]}
        assert design.offset_keys == {"offset:mean"}
        assert set(design.matrix_keys) == set(matrices)

        assert fit_design(obs, {"mean": "~ 0 + offset(x)"}, chunked=chunked).matrices(obs)["mean"].shape == (6, 0)

    for factor, column in [("{x + lib}", "x + lib"), ("`with space`", "with space")]:
        other = fit_design(obs, {"mean": f"~ {factor} + offset(x)"}).matrices(obs)["mean"]
        assert list(other.columns) == ["Intercept", column]
        np.testing.assert_allclose(other.iloc[:, 1], obs.x + obs.lib)


def test_design_query_rows_and_validation(obs):
    for chunked in (False, True):
        design = fit_design(obs, {"mean": "~ offset(center(x))"}, chunked=chunked)
        query = obs.iloc[[4, 0, 4]]
        result = design.matrices(query)["offset:mean"]
        np.testing.assert_allclose(result.iloc[:, 0], query.x - obs.x.mean())
        assert result.index.equals(query.index)
        with pytest.raises(Exception, match="x"):
            design.matrices(query.drop(columns="x"))

    design = fit_design(obs, {"mean": "~ offset(log(lib))"})
    for bad in (np.nan, -1.):
        query = obs.iloc[:2].copy()
        query.iloc[1, query.columns.get_loc("lib")] = bad
        with pytest.raises(ValueError):
            design.matrices(query)

    with pytest.raises(ValueError, match="Chunked"):
        fit_design(obs, {"mean": "~ offset(log(lib + 1))"}, chunked=True)
    with pytest.raises(ValueError, match="offset"):
        fit_design(obs, {"group": "~ offset(x)"}, partition_keys={"group"})


# Data loading

@pytest.mark.parametrize("mode", ["preloaded", "chunked", "backed"])
def test_loader_alignment(obs, tmp_path, mode):
    counts = np.arange(12.).reshape(6, 2)
    adata = AnnData(counts, obs=obs)
    if mode == "backed":
        path = tmp_path / "offsets.h5ad"
        adata.write_h5ad(path)
        adata = read_h5ad(path, backed="r")
    try:
        loader = adata_loader(adata, {"mean": "~ x + offset(log(lib))"},
                              batch_size=4, chunk_size=None if mode == "preloaded" else 3,
                              device="cpu")
        indices = [4, 0, 4, 2]
        y, x = loader.dataset.__getitems__(indices)
        torch.testing.assert_close(y, torch.tensor(counts[indices]).float())
        torch.testing.assert_close(x["offset:mean"], torch.tensor(np.log(obs.lib.iloc[indices].values)).float()[:, None])
        assert all(xb["offset:mean"].shape[1] == 1 for _, xb in loader)
    finally:
        if adata.isbacked:
            adata.file.close()


def test_model_dimensions_exclude_offsets(obs):
    model = NegBin("~ x + offset(log(lib))", device="cpu")
    model.setup_data(AnnData(np.ones((6, 2)), obs=obs), batch_size=4, chunk_size=3)
    assert model.feature_dims == {"mean": 2, "dispersion": 1}
    model.setup_optimizer()
    assert set(model.predict.coefs) == {"mean", "dispersion"}
    assert list(model.format_parameters()["mean"].index) == ["Intercept", "x"]


# Predictors and initialization

def test_glm_predictor_adds_fixed_offsets_before_link():
    for link in (lambda x: x, torch.exp):
        model = GLMPredictor(2, {"mean": 1, "other": 1},
                             link_fns={"mean": link, "other": link}, device="cpu")
        with torch.no_grad():
            model.coefs["mean"].copy_(torch.tensor([[1., 2.]]))
            model.coefs["other"].copy_(torch.tensor([[2., 3.]]))
        offset = torch.tensor([[0.5], [-1.]])
        x = {"mean": torch.ones(2, 1), "other": torch.ones(2, 1),
             "offset:mean": offset, "offset:other": -offset}
        out = model(x)
        torch.testing.assert_close(out["mean"], link(torch.tensor([[1.5, 2.5], [0., 1.]])))
        torch.testing.assert_close(out["other"], link(torch.tensor([[1.5, 2.5], [3., 4.]])))
        assert sum(p.numel() for p in model.parameters()) == 4
        sum(v.sum() for v in out.values()).backward()
        model.optimizer.step()
        torch.testing.assert_close(offset, torch.tensor([[0.5], [-1.]]))


def test_eqtl_predictor_adds_offsets_to_each_component():
    for simulated in (False, True):
        model = EQTLPredictor(2, {"mean": 1, "dispersion": 1},
                              dosage=torch.tensor([[[1.], [2.]], [[2.], [1.]]]),
                              mask=torch.ones(2, 1), interaction_dim=1, device="cpu")
        with torch.no_grad():
            model.coefs["mean"].fill_(0.1)
            model.coefs["dispersion"].fill_(0.2)
            model.U.fill_(0.3)
            model.U_sim.fill_(-0.2)
            model.beta.fill_(0.4)
            model.zeta.fill_(0.5)
        model.use_simulated_donor_effects = simulated
        x = {"mean": torch.ones(2, 1), "dispersion": torch.ones(2, 1),
             "genotype": torch.tensor([[0.], [1.]]),
             "interaction": torch.tensor([[1., 2.], [1., 3.]]),
             "offset:mean": torch.tensor([[0.2], [-0.1]]),
             "offset:dispersion": torch.tensor([[-0.3], [0.4]])}
        eta = torch.tensor([[1.7, 3.1], [3.8, 1.9]]) + (-0.2 if simulated else 0.3)
        torch.testing.assert_close(model(x)["mean"], eta.exp())
        torch.testing.assert_close(model(x)["dispersion"], torch.tensor([[-0.1, -0.1], [0.6, 0.6]]).exp())


def test_irls_initialization_accounts_for_offsets():
    exposure = torch.tensor([[1.], [2.], [4.]])
    rates = torch.tensor([[2., 3.]])
    x = {"mean": torch.ones(3, 1), "offset:mean": exposure.log()}
    beta, _ = initialize_parameters([(exposure * rates, x)], 2, 1, 0)
    torch.testing.assert_close(beta, rates.log(), rtol=1e-5, atol=1e-5)
    x["offset:mean"] = exposure.log() + 1
    shifted, _ = initialize_parameters([(exposure * rates, x)], 2, 1, 0)
    torch.testing.assert_close(shifted, rates.log() - 1, rtol=1e-5, atol=1e-5)

    # Dispersion offset enters the moment estimate; no fitted mean columns.
    y, mu = torch.tensor([[0.], [2.], [7.]]), torch.tensor([[1.], [2.], [4.]])
    d = torch.tensor([[0.], [0.5], [1.]])
    x = {"mean": torch.empty(3, 0), "dispersion": torch.ones(3, 1),
         "offset:mean": mu.log(), "offset:dispersion": d}
    beta, gamma = initialize_parameters([(y, x)], 1, 0, 1)
    assert beta.shape == (0, 1)
    torch.testing.assert_close(gamma, ((mu.square() * (-d).exp()).sum() / 3).log().reshape(1, 1))


# Models

@pytest.mark.parametrize("model_class", [PoissonCopula, NegBinCopula, NegBinEQTLCopula])
def test_offsets_end_to_end(model_class):
    torch.manual_seed(4)
    obs = pd.DataFrame({"lib": [1., 2., 4., 2.] * 3,
                        "donor": np.repeat(["a", "b", "c"], 4)},
                       index=[f"c{i}" for i in range(12)])
    counts = np.array([[0, 3], [2, 1], [5, 4], [1, 5], [2, 4], [3, 2],
                       [7, 5], [2, 6], [3, 5], [4, 3], [9, 6], [3, 7]], dtype=float)
    data = AnnData(counts, obs=obs, var=pd.DataFrame(index=["g1", "g2"]))
    kwargs = {}
    mean_formula = "~ offset(log(lib))"
    if model_class == NegBinEQTLCopula:
        mean_formula += " + genotype + (1 | donor)"
        kwargs = {"dosage": pd.DataFrame({"s": [0., 1., 2.]}, index=["a", "b", "c"]),
                  "snp_map": {"g1": ["s"], "g2": ["s"]}}
    sim = model_class(mean_formula=mean_formula, device="cpu", **kwargs)
    sim.fit(data, batch_size=4, chunk_size=3, max_epochs=1, val_frac=0, verbose=False)

    whole = sim.predict(obs, batch_size=4)
    indices = [9, 1, 5, 0]
    query = obs.iloc[indices].copy()
    part = sim.predict(query, batch_size=2)
    doubled = query.copy()
    doubled["lib"] *= 2
    scaled = sim.predict(doubled, batch_size=2)
    for name in whole:
        np.testing.assert_allclose(part[name], whole[name][indices], rtol=1e-5)
        np.testing.assert_allclose(scaled[name], part[name] * (2 if name == "mean" else 1), rtol=1e-5)

    sampled = sim.sample(query, batch_size=2)
    assert sampled.shape == (4, 2)
    assert np.isfinite(sampled.X).all()
    with pytest.raises(Exception, match="lib"):
        sim.predict(query.drop(columns="lib"))


@pytest.mark.parametrize("kind", ["poisson", "negbin", "eqtl"])
def test_models_without_offsets_match_reference(kind):
    """Initialization outputs frozen before offsets were added; do not regenerate."""
    obs = pd.DataFrame({"donor": np.repeat(["a", "b", "c"], 4),
                        "x": [0., 1., 2., 1.] * 3},
                       index=[f"c{i}" for i in range(12)])
    counts = np.array([[1, 3], [2, 1], [5, 4], [1, 5], [2, 4], [3, 2],
                       [7, 5], [2, 6], [3, 5], [4, 3], [9, 6], [3, 7]], dtype=float)
    adata = AnnData(counts, obs=obs, var=pd.DataFrame(index=["g1", "g2"]))
    if kind == "eqtl":
        model = NegBinEQTL("~ x + genotype + (1 | donor)", device="cpu",
                           dosage=pd.DataFrame({"s": [0., 1., 2.]}, index=["a", "b", "c"]),
                           snp_map={"g1": ["s"], "g2": ["s"]})
    else:
        model = {"poisson": Poisson, "negbin": NegBin}[kind]("~ x", device="cpu")
    torch.manual_seed(17)
    model.setup_data(adata, batch_size=4)
    model.setup_optimizer()
    model._initialize_parameters()

    batches = list(model.loader)
    assert all(not any(k.startswith("offset:") for k in x) for _, x in batches)
    actual = {"coefs": {k: v.detach() for k, v in model.predict.coefs.items()},
              "predictions": {k: torch.cat([model.predict(x)[k].detach() for _, x in batches])
                              for k in model.predict(batches[0][1])}}
    if kind == "eqtl":
        actual["beta"], actual["lam"] = model.predict.beta.detach(), model.predict.lam.detach()

    expected = json.loads((Path(__file__).parent / "fixtures/offset_compatibility.json").read_text())[kind]
    assert model.predictor_names == expected["names"]
    for section in ("coefs", "predictions"):
        assert actual[section].keys() == expected[section].keys()
        for key, value in actual[section].items():
            torch.testing.assert_close(value, torch.tensor(expected[section][key]), rtol=1e-6, atol=1e-6)
    for key in ("beta", "lam") if kind == "eqtl" else ():
        torch.testing.assert_close(actual[key], torch.tensor(expected[key]), rtol=1e-6, atol=1e-6)
