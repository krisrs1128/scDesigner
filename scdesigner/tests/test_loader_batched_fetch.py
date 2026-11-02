"""Tests for the batched fetch path in PreloadedDataset.

The guarantee being protected is that batching *how* rows are fetched does not
change *which* rows land in a batch: the sampler still decides that, so every
batch must be bitwise identical to the per-row path it replaced.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData, read_h5ad
from torch.utils.data import DataLoader, Subset

from scdesigner.data.design import fit_design
from scdesigner.data.loader import (
    AnnDataDataset,
    PreloadedDataset,
    _preloaded_adata,
    adata_loader,
    dict_collate_fn,
)

FORMULA = {"mean": "~ grp", "genotype": "~ 0 + code"}


def _adata(n_obs=120, n_vars=7, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "grp": np.resize(np.array(["a", "b", "c"], dtype=object), n_obs),
            "code": (np.arange(n_obs) // 10).astype(float),
        }
    )
    adata = AnnData(X=rng.poisson(3.0, size=(n_obs, n_vars)).astype(float), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    return adata


def _dataset(adata, device="cpu"):
    """Build the dataset directly.

    `adata_loader` filters its kwargs through DEFAULT_ALLOWED_KWARGS['data'],
    which does not include `drop_last` or `generator`, so those cannot be
    exercised through it. Constructing the loader here keeps these tests on
    the unit that actually changed: PreloadedDataset plus dict_collate_fn.
    """
    return _preloaded_adata(adata, fit_design(adata.obs, FORMULA), torch.device(device))


def _loader(dataset, batch_size, shuffle=False, drop_last=False, seed=1234):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=dict_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )


class _PerRowDataset(torch.utils.data.Dataset):
    """The pre-change dataset: only ``__getitem__``, no batched fetch.

    Holds the same tensors as the real dataset, so a loader built over it takes
    the per-row path and ``dict_collate_fn`` stacks the rows -- exactly the
    behaviour that shipped before ``__getitems__`` existed.

    Comparing two identically-constructed DataLoaders (rather than replaying
    the sampler by hand) keeps the test honest about index order: DataLoader
    draws once from its generator for ``_base_seed`` before the sampler runs,
    so a hand-rolled ``randperm`` does not reproduce the shuffled order.
    """

    def __init__(self, y, x):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], {k: v[idx] for k, v in self.x.items()}


def _assert_batches_equal(produced, expected):
    assert len(produced) == len(expected), "different number of batches"
    for (y_a, x_a), (y_b, x_b) in zip(produced, expected):
        assert torch.equal(y_a, y_b), "X batch differs"
        assert set(x_a) == set(x_b), "obs keys differ"
        for key in x_a:
            assert torch.equal(x_a[key], x_b[key]), f"obs[{key!r}] differs"


# --- Tier A: bitwise equivalence with the per-row path ----------------------


@pytest.mark.parametrize("batch_size", [1, 7, 32, 1000])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_batched_fetch_matches_per_row(batch_size, shuffle, drop_last):
    seed = 1234
    dataset = _dataset(_adata())
    reference = _PerRowDataset(dataset.y, dataset.x)

    batched = _loader(dataset, batch_size, shuffle, drop_last, seed)
    per_row = _loader(reference, batch_size, shuffle, drop_last, seed)
    _assert_batches_equal(list(batched), list(per_row))


def test_batched_fetch_covers_every_row_exactly_once():
    """A ragged final batch (95 rows, batch 16) must not drop or repeat rows."""
    adata = _adata(n_obs=95)
    loader = adata_loader(adata, FORMULA, batch_size=16, device="cpu")
    seen = torch.cat([y for y, _ in loader])
    assert seen.shape[0] == adata.n_obs
    torch.testing.assert_close(seen, loader.dataset.y)


# --- Tier B: the batched path is actually the one being used ---------------


def test_batched_path_is_used_not_silently_falling_back(monkeypatch):
    """A fallback to per-row would still be correct, and so would pass every
    other test in this file while delivering none of the speedup. Make the
    per-row accessor fail loudly, then confirm iteration still works.
    """
    adata = _adata()
    loader = adata_loader(adata, FORMULA, batch_size=16, device="cpu")

    def explode(self, idx):
        raise AssertionError("__getitem__ called; batched fetch was not used")

    monkeypatch.setattr(PreloadedDataset, "__getitem__", explode)
    batches = list(loader)
    assert sum(y.shape[0] for y, _ in batches) == adata.n_obs


def test_getitems_returns_assembled_batch():
    adata = _adata()
    dataset = adata_loader(adata, FORMULA, batch_size=8, device="cpu").dataset
    y, x = dataset.__getitems__([3, 1, 2])

    assert y.shape == (3, adata.n_vars)
    torch.testing.assert_close(y, dataset.y[[3, 1, 2]])
    for key in FORMULA:
        torch.testing.assert_close(x[key], dataset.x[key][[3, 1, 2]])


# --- Tier C: Subset / validation-split path --------------------------------


def test_batched_fetch_through_subset_matches_per_row():
    """torch's Subset.__getitems__ delegates to the parent dataset, so the
    train/validation split in Marginal.setup_validation_split inherits the
    batched path without any change to _subset_loader.
    """
    adata = _adata()
    dataset = _dataset(adata)
    reference = _PerRowDataset(dataset.y, dataset.x)
    indices = list(range(0, len(dataset), 3))

    batched = DataLoader(
        Subset(dataset, indices), batch_size=16, collate_fn=dict_collate_fn
    )
    per_row = DataLoader(
        Subset(reference, indices), batch_size=16, collate_fn=dict_collate_fn
    )
    _assert_batches_equal(list(batched), list(per_row))


def test_subset_uses_batched_path(monkeypatch):
    adata = _adata()
    dataset = adata_loader(adata, FORMULA, batch_size=16, device="cpu").dataset
    subset_loader = DataLoader(
        Subset(dataset, list(range(40))), batch_size=8, collate_fn=dict_collate_fn
    )

    def explode(self, idx):
        raise AssertionError("__getitem__ called through Subset")

    monkeypatch.setattr(PreloadedDataset, "__getitem__", explode)
    assert sum(y.shape[0] for y, _ in subset_loader) == 40


# --- Tier D: device handling ------------------------------------------------


def test_batches_land_on_requested_device():
    adata = _adata()
    loader = adata_loader(adata, FORMULA, batch_size=16, device="cpu")
    y, x = next(iter(loader))
    assert y.device.type == "cpu"
    for key in FORMULA:
        assert x[key].device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)
def test_batched_fetch_agrees_across_devices():
    adata = _adata()
    cpu_loader = adata_loader(adata, FORMULA, batch_size=16, device="cpu")
    mps_loader = adata_loader(adata, FORMULA, batch_size=16, device="mps")

    for (y_c, x_c), (y_m, x_m) in zip(cpu_loader, mps_loader):
        assert y_m.device.type == "mps"
        torch.testing.assert_close(y_c, y_m.cpu())
        for key in FORMULA:
            torch.testing.assert_close(x_c[key], x_m[key].cpu())


# --- Tier E: backed chunked batched fetch ----------------------------------


def test_backed_subset_batched_fetch_matches_per_row(tmp_path, monkeypatch):
    """A batch spanning chunks must preserve subset order and repeated rows."""
    adata = _adata(n_obs=50)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "counts.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        design = fit_design(backed.obs, FORMULA, chunked=True)
        dataset = AnnDataDataset(backed, design, chunk_size=7, device=torch.device("cpu"))
        indices = [0, 3, 6, 7, 7, 11, 13, 14, 19, 21, 22, 28, 35, 49]
        expected = [dict_collate_fn([dataset[i] for i in indices[:12]]),
                    dict_collate_fn([dataset[i] for i in indices[12:]])]

        def explode(self, idx):
            raise AssertionError("chunked dataset fetched rows individually")

        monkeypatch.setattr(AnnDataDataset, "__getitem__", explode)
        batches = list(DataLoader(Subset(dataset, indices), batch_size=12,
                                  collate_fn=dict_collate_fn))
        _assert_batches_equal(batches, expected)
    finally:
        backed.file.close()


# --- Tier F: fitted results are unchanged -----------------------------------


def _fit_negbin(dataset_cls_patch, monkeypatch, seed=0):
    """Fit a NegBin marginal, optionally forcing the per-row loader path."""
    from scdesigner.distributions import NegBin

    if dataset_cls_patch:
        monkeypatch.delattr(PreloadedDataset, "__getitems__")

    adata = _adata(n_obs=200, n_vars=5, seed=3)
    torch.manual_seed(seed)
    model = NegBin({"mean": "~ grp", "dispersion": "~ 1"}, device="cpu")
    model.setup_data(adata, batch_size=64)
    model.fit(max_epochs=15, verbose=False, val_frac=0.0, min_epochs=15)
    return {k: v.copy() for k, v in model.parameters.items()}


def test_fitted_parameters_identical_to_per_row_path(monkeypatch):
    """Batching the fetch must not move any fitted coefficient.

    Batch contents are bitwise identical and the model is deterministic given
    a seed, so this is an exact-equality check, not an approximate one.
    """
    batched = _fit_negbin(False, monkeypatch)
    per_row = _fit_negbin(True, monkeypatch)

    assert set(batched) == set(per_row)
    for key in batched:
        np.testing.assert_array_equal(
            batched[key].to_numpy(), per_row[key].to_numpy()
        )
