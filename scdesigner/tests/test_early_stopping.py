import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.base.marginal import GLMPredictor, Marginal
from scdesigner.data.loader import AnnDataDataset
from scdesigner.distributions import Bernoulli, Poisson
from scdesigner.simulators import (
    CompositeCopula,
    NegBinCopula,
    NegBinIRLSCopula,
    PoissonCopula,
)


def _adata(n_obs=20, n_vars=4):
    rng = np.random.default_rng(10)
    obs = pd.DataFrame(
        {"group": np.resize(np.array(["A", "B", "C"], dtype=object), n_obs)}
    )
    adata = AnnData(
        X=rng.poisson(2.0, size=(n_obs, n_vars)).astype(float),
        obs=obs,
    )
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    return adata


class ConstantValidationMarginal(Marginal):
    def __init__(self):
        super().__init__({"mean": "~ 1"}, device="cpu")

    def setup_optimizer(self, optimizer_class=torch.optim.Adam, **optimizer_kwargs):
        def loss_fn(batch):
            return sum((coef ** 2).sum() for coef in self.predict.coefs.values())

        self.predict = GLMPredictor(
            n_outcomes=self.n_outcomes,
            feature_dims=self.feature_dims,
            loss_fn=loss_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.device,
        )

    def likelihood(self, batch):
        y, _ = batch
        return torch.zeros_like(y)

    def invert(self, u, x):
        return u

    def uniformize(self, y, x):
        return torch.full_like(y, 0.5)


class RecordingInitializerMarginal(ConstantValidationMarginal):
    def __init__(self):
        super().__init__()
        self.initializer_train_size = None
        self.initializer_validation_size = None

    def _initialize_parameters(self, **kwargs):
        self.initializer_train_size = len(self._active_train_loader().dataset)
        self.initializer_validation_size = len(self.validation_loader.dataset)


class ScriptedValidationMarginal(ConstantValidationMarginal):
    def __init__(self, validation_losses):
        super().__init__()
        self.validation_losses = list(validation_losses)

    def _validation_loss(self):
        if self.validation_loader is None:
            return None
        index = min(len(self.fit_history), len(self.validation_losses) - 1)
        return self.validation_losses[index]


def test_validation_split_keeps_full_loader_and_is_deterministic():
    adata = _adata(n_obs=12)
    model = ConstantValidationMarginal()
    model.setup_data(adata, batch_size=4, device="cpu")
    model.setup_validation_split(val_frac=0.25, validation_seed=42)

    assert len(model.loader.dataset) == 12
    assert len(model.train_loader.dataset) == 9
    assert len(model.validation_loader.dataset) == 3
    assert set(model.train_loader.dataset.indices).isdisjoint(
        set(model.validation_loader.dataset.indices)
    )

    model_same_seed = ConstantValidationMarginal()
    model_same_seed.setup_data(adata, batch_size=4, device="cpu")
    model_same_seed.setup_validation_split(val_frac=0.25, validation_seed=42)

    np.testing.assert_array_equal(
        model.train_loader.dataset.indices,
        model_same_seed.train_loader.dataset.indices,
    )
    np.testing.assert_array_equal(
        model.validation_loader.dataset.indices,
        model_same_seed.validation_loader.dataset.indices,
    )


def test_early_stopping_waits_until_min_epochs():
    model = ConstantValidationMarginal()
    model.setup_data(_adata(n_obs=20), batch_size=5, device="cpu")

    model.fit(
        max_epochs=20,
        val_frac=0.25,
        min_epochs=5,
        patience=2,
        loss_tol=0.01,
        validation_seed=0,
        verbose=False,
    )

    assert model.stopped_epoch == 5
    assert model.fit_history[-1]["stopped"]
    assert len(model.fit_history) == 5
    assert not any(row["stopped"] for row in model.fit_history[:-1])


def test_loss_tol_is_relative_to_best_validation_loss():
    model = ScriptedValidationMarginal([1.0, 0.99995, 0.9998])
    model.setup_data(_adata(n_obs=20), batch_size=5, device="cpu")

    model.fit(
        max_epochs=3,
        val_frac=0.25,
        min_epochs=10,
        patience=10,
        validation_seed=0,
        verbose=False,
    )

    assert [row["best"] for row in model.fit_history] == [True, False, True]
    assert model.best_epoch == 3
    assert model.best_validation_loss == pytest.approx(0.9998)


def test_val_frac_zero_preserves_full_data_training():
    model = ConstantValidationMarginal()
    model.setup_data(_adata(n_obs=8), batch_size=4, device="cpu")

    model.fit(max_epochs=3, val_frac=0, verbose=False)

    assert model.train_loader is model.loader
    assert model.validation_loader is None
    assert len(model.fit_history) == 3
    assert all(row["val_loss"] is None for row in model.fit_history)


def test_glm_predictor_default_lr_and_learning_rate_alias():
    model = ConstantValidationMarginal()
    model.setup_data(_adata(n_obs=8), batch_size=4, device="cpu")
    model.setup_validation_split(val_frac=0)
    model.setup_optimizer()
    assert model.predict.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

    alias_model = ConstantValidationMarginal()
    alias_model.setup_data(_adata(n_obs=8), batch_size=4, device="cpu")
    alias_model.setup_validation_split(val_frac=0)
    alias_model.setup_optimizer(learning_rate=0.02)
    assert alias_model.predict.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)

    precedence_model = ConstantValidationMarginal()
    precedence_model.setup_data(_adata(n_obs=8), batch_size=4, device="cpu")
    precedence_model.setup_validation_split(val_frac=0)
    precedence_model.setup_optimizer(lr=0.03, learning_rate=0.02)
    assert precedence_model.predict.optimizer.param_groups[0]["lr"] == pytest.approx(0.03)


def test_initializer_hook_runs_after_validation_split():
    model = RecordingInitializerMarginal()
    model.setup_data(_adata(n_obs=20), batch_size=5, device="cpu")

    model.fit(max_epochs=1, val_frac=0.25, verbose=False)

    assert model.initializer_train_size == 15
    assert model.initializer_validation_size == 5


def test_object_covariates_keep_stable_chunk_design_columns():
    obs = pd.DataFrame({"group": ["A", "B", "C", "C", "A", "B"]})
    adata = AnnData(X=np.ones((6, 2)), obs=obs)
    dataset = AnnDataDataset(
        adata,
        {"mean": "~ group"},
        chunk_size=2,
        device=torch.device("cpu"),
    )

    _, x_first = dataset[0]
    n_columns = x_first["mean"].numel()
    _, x_middle = dataset[2]
    _, x_last = dataset[5]

    assert x_middle["mean"].numel() == n_columns
    assert x_last["mean"].numel() == n_columns


def test_poisson_copula_smoke_fit_with_early_stopping():
    adata = _adata(n_obs=30, n_vars=4)
    sim = PoissonCopula(mean_formula="~ group", copula_formula="~ 1")

    sim.fit(
        adata,
        batch_size=10,
        device="cpu",
        max_epochs=6,
        val_frac=0.2,
        min_epochs=2,
        patience=1,
        loss_tol=1e9,
        validation_seed=1,
        verbose=False,
        top_k=2,
    )

    assert sim.marginal.stopped_epoch == 2
    assert len(sim.marginal.loader.dataset) == adata.n_obs
    assert sim.marginal.validation_loader is not None
    assert len(sim.marginal.train_loader.dataset) < len(sim.marginal.loader.dataset)
    assert sim.parameters["copula"] is not None


def test_negbin_copula_smoke_fit_with_early_stopping():
    adata = _adata(n_obs=30, n_vars=4)
    sim = NegBinCopula(
        mean_formula="~ group",
        dispersion_formula="~ 1",
        copula_formula="~ 1",
    )

    sim.fit(
        adata,
        batch_size=10,
        device="cpu",
        max_epochs=4,
        val_frac=0.2,
        min_epochs=2,
        patience=1,
        loss_tol=1e9,
        validation_seed=2,
        verbose=False,
        top_k=2,
    )

    assert sim.marginal.stopped_epoch == 2
    assert len(sim.marginal.loader.dataset) == adata.n_obs
    assert sim.parameters["copula"] is not None


def test_negbin_irls_copula_uses_full_data_without_validation_stopping():
    adata = _adata(n_obs=20, n_vars=3)
    sim = NegBinIRLSCopula(
        mean_formula="~ group",
        dispersion_formula="~ 1",
        copula_formula="~ 1",
    )

    sim.fit(
        adata,
        batch_size=10,
        max_epochs=1,
        val_frac=0.5,
        min_epochs=1,
        patience=1,
        loss_tol=1e9,
        validation_seed=2,
        verbose=False,
        top_k=2,
    )

    assert sim.marginal.train_loader is sim.marginal.loader
    assert sim.marginal.validation_loader is None
    assert sim.marginal.fit_history[-1]["val_loss"] is None
    assert sim.marginal.stopped_epoch is None
    assert sim.parameters["copula"] is not None


def test_composite_copula_smoke_fit_with_early_stopping():
    rng = np.random.default_rng(12)
    obs = pd.DataFrame(
        {"group": np.resize(np.array(["A", "B", "C"], dtype=object), 30)}
    )
    x_counts = rng.poisson(2.0, size=(30, 2))
    x_binary = rng.binomial(1, 0.4, size=(30, 2))
    adata = AnnData(X=np.concatenate([x_counts, x_binary], axis=1).astype(float), obs=obs)
    adata.var_names = [f"g{i}" for i in range(4)]

    sim = CompositeCopula(
        [
            (["g0", "g1"], Poisson("~ group", device="cpu")),
            (["g2", "g3"], Bernoulli("~ group", device="cpu")),
        ],
        copula_formula="~ 1",
    )

    sim.fit(
        adata,
        batch_size=10,
        device="cpu",
        max_epochs=4,
        val_frac=0.2,
        min_epochs=2,
        patience=1,
        loss_tol=1e9,
        validation_seed=2,
        verbose=False,
        top_k=2,
    )

    assert [marginal.stopped_epoch for _, marginal in sim.marginals] == [2, 2]
    assert sim.parameters["copula"] is not None
