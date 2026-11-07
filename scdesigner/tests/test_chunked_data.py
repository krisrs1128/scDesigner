"""Chunked fits agree with in-memory results, read in row order, and stay bounded."""

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData

from formulaic import ModelSpec

import scdesigner.data.loader as loader_module
from scdesigner.data import GPBasis
from scdesigner.data.feature_summary import summarize_features
from scdesigner.data.loader import adata_loader
from scdesigner.data.response import iter_response_chunks, read_response_rows
from scdesigner.distributions import NegBin
from scdesigner.simulators import NegBinCopula, SpatialNegBinCopula

STORAGE = {"dense": np.asarray, "csr": sp.csr_matrix, "csc": sp.csc_matrix}
BACKED = ("dense", "csr")
FIT = dict(chunk_size=7, batch_size=10, max_epochs=2, max_iter=1, val_frac=0.25,
           verbose=False)


@contextmanager
def backed_copy(data: AnnData, storage: str, tmp_path: Path) -> Iterator[AnnData]:
    """``data`` written to ``tmp_path`` in ``storage`` layout, opened with ``backed="r"``."""
    data = data.copy()
    data.X = STORAGE[storage](data.X)
    path = tmp_path / f"{storage}.h5ad"
    data.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        yield backed
    finally:
        backed.file.close()


def record_payload_reads(monkeypatch, data: AnnData) -> list:
    """Entries per read of the backed file's ``/X/data`` and ``/X/indices``.

    Reads that are not slices, including whole-array conversion, count as ``inf``.
    """
    reads = []
    filename = Path(data.filename).resolve()
    getitem, to_array = h5py.Dataset.__getitem__, h5py.Dataset.__array__

    def is_payload(dataset):
        return (dataset.name in ("/X/data", "/X/indices")
                and Path(dataset.file.filename).resolve() == filename)

    def counted_getitem(dataset, key, *args, **kwargs):
        if is_payload(dataset):
            if isinstance(key, slice):
                start, stop, _ = key.indices(len(dataset))
                reads.append(stop - start)
            else:
                reads.append(math.inf)
        return getitem(dataset, key, *args, **kwargs)

    def counted_array(dataset, *args, **kwargs):
        if is_payload(dataset):
            reads.append(math.inf)
        return to_array(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counted_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", counted_array)
    return reads


def concatenate_batches(loader):
    """Responses and per-key designs of all batches, stacked in loader order."""
    batches = list(loader)
    y = np.concatenate([y.numpy() for y, _ in batches])
    x = {key: np.concatenate([x[key].numpy() for _, x in batches]) for key in batches[0][1]}
    return y, x


def fit_spatial(data):
    sim = SpatialNegBinCopula(mean_df=3, n_landmarks=8, copula_formula="~ batch",
                              estimator="oas", top_k=2, device="cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    sim.fit(data, **FIT)
    return sim


def record_chunk_loads(monkeypatch) -> list:
    """``(start, stop)`` of each chunk an :class:`AnnDataDataset` loads."""
    loads, read = [], loader_module.read_response_rows

    def counted(adata, start, stop):
        loads.append((start, stop))
        return read(adata, start, stop)

    monkeypatch.setattr(loader_module, "read_response_rows", counted)
    return loads


@pytest.mark.parametrize("storage", BACKED)
def test_backed_summary_matches_reference(storage, tmp_path):
    values = np.array([[0., 1., 0.], [0., 3., 0.], [0., 2., 4.], [0., 2., 0.], [0., 12., 0.]])
    with backed_copy(AnnData(X=values), storage, tmp_path) as data:
        result = summarize_features(data, include_max=True, chunk_size=2)
    np.testing.assert_allclose(result.mean, [0., 4., 0.8], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.maximum, [0., 12., 4.])


def test_default_response_chunk_size():
    data = AnnData(X=np.zeros((20_001, 1), dtype=np.float32))
    chunks = [(start, stop) for start, stop, _ in iter_response_chunks(data)]
    assert chunks == [(0, 20_000), (20_000, 20_001)]


def test_backed_csc_is_rejected_and_in_memory_csc_is_read(spatial_counts, tmp_path):
    with backed_copy(spatial_counts, "csc", tmp_path) as data:
        with pytest.raises(ValueError, match="Backed CSC"):
            fit_spatial(data)
        with pytest.raises(ValueError, match="Backed CSC"):
            read_response_rows(data[::2], 0, 3)

    memory = spatial_counts.copy()
    memory.X = sp.csc_matrix(memory.X)
    np.testing.assert_array_equal(read_response_rows(memory[::2], 0, 3),
                                  spatial_counts.X[[0, 2, 4]])


def test_csr_response_reads_are_bounded(spatial_counts, tmp_path, monkeypatch):
    with backed_copy(spatial_counts, "csr", tmp_path) as data:
        reads = record_payload_reads(monkeypatch, data)
        fit_spatial(data).complexity(data)
    assert reads and max(reads) <= 7 * data.n_vars


def test_chunked_loader_preserves_values_and_covariates(monkeypatch):
    values = sp.csr_matrix(np.arange(21.).reshape(7, 3))
    obs = pd.DataFrame({"z": np.arange(7), "label": list("aabbcca")},
                       index=[f"cell{i}" for i in range(7)])
    data = AnnData(X=values, obs=obs)
    original_obs = data.obs.copy()
    formula = {"mean": "~ z + label", "group": "~ 0 + label"}
    options = dict(batch_size=3, partition_keys={"group"}, device="cpu")
    reference_y, reference_x = concatenate_batches(adata_loader(data, formula, **options))

    densify = sp.csr_matrix.toarray

    def bounded_densify(matrix, *args, **kwargs):
        assert matrix.shape[0] <= 2
        return densify(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", bounded_densify)
    chunked_y, chunked_x = concatenate_batches(
        adata_loader(data, formula, chunk_size=2, **options))

    np.testing.assert_array_equal(chunked_y, reference_y)
    for key in ("mean", "group"):
        np.testing.assert_array_equal(chunked_x[key], reference_x[key])
    np.testing.assert_array_equal(chunked_x["group"].sum(axis=1), np.ones(7))
    pd.testing.assert_frame_equal(data.obs, original_obs)


def test_chunked_spatial_fit_builds_no_full_design_or_kernel(spatial_counts, monkeypatch):
    design_rows, kernel_rows = [], []
    evaluate, kernel = ModelSpec.get_model_matrix, GPBasis._kernel

    def recorded_evaluate(spec, data, *args, **kwargs):
        design_rows.append(len(data))
        return evaluate(spec, data, *args, **kwargs)

    def recorded_kernel(basis, A, B):
        if A is not B:  # the landmark-by-landmark kernel is exempt
            kernel_rows.append(len(A))
        return kernel(basis, A, B)

    monkeypatch.setattr(ModelSpec, "get_model_matrix", recorded_evaluate)
    monkeypatch.setattr(GPBasis, "_kernel", recorded_kernel)
    fit_spatial(spatial_counts).complexity()
    assert max(design_rows) <= 7 and max(kernel_rows) <= 7


def test_chunked_basis_standardization_matches_full(spatial_counts):
    coords = spatial_counts.obs[["spatial1", "spatial2"]].to_numpy()
    fitted = []
    for chunk_size in (None, 7):
        np.random.seed(0)
        fitted.append(GPBasis(df=3, n_landmarks=8, standardize=True).fit(coords, chunk_size))
    full, chunked = fitted
    unstandardized = full._kernel(coords, full.landmarks_) @ full.projection_
    np.testing.assert_array_equal(full.col_sd_, np.maximum(unstandardized.std(axis=0), full.eps))
    np.testing.assert_allclose(chunked.col_sd_, full.col_sd_, rtol=1e-12)


def test_chunked_split_keeps_membership_and_reads_rows_in_order(spatial_counts, monkeypatch):
    n_chunks = math.ceil(spatial_counts.n_obs / 7)
    splits = {}
    for chunk_size in (None, 7):
        model = NegBin({"mean": "~ batch", "dispersion": "~ 1"}, device="cpu")
        model.setup_data(spatial_counts, batch_size=5, chunk_size=chunk_size)
        model.setup_validation_split(val_frac=0.25, validation_seed=3)
        splits[chunk_size] = model
    preloaded, chunked = splits[None], splits[7]

    loads = record_chunk_loads(monkeypatch)
    for name in ("train_loader", "validation_loader"):
        indices = getattr(chunked, name).dataset.indices
        assert sorted(indices) == sorted(getattr(preloaded, name).dataset.indices)
        assert indices == sorted(indices)
        loads.clear()
        list(getattr(chunked, name))
        assert len(loads) <= n_chunks

    with pytest.raises(ValueError, match="shuffle=True"):
        adata_loader(spatial_counts, {"mean": "~ 1"}, chunk_size=7, shuffle=True)


@pytest.mark.parametrize("chunk_size", [None, 7])
def test_setup_data_retains_linearly_dependent_predictors(spatial_counts, chunk_size):
    data = spatial_counts.copy()
    data.obs["double"] = 2 * data.obs["spatial1"]
    model = NegBin({"mean": "~ spatial1 + double"}, device="cpu")
    model.setup_data(data, chunk_size=chunk_size)
    matrix = model.design.matrices(data.obs)["mean"]
    assert list(matrix.columns) == ["Intercept", "spatial1", "double"]
    np.testing.assert_array_equal(matrix["double"], 2 * matrix["spatial1"])


def test_chunked_and_preloaded_fits_agree_on_designs_and_parameters(spatial_counts):
    data = spatial_counts[np.argsort(spatial_counts.obs["spatial1"].to_numpy())].copy()
    fits = {}
    for chunk_size in (None, 7):
        sim = NegBinCopula(
            mean_formula="~ batch + bs(spatial1, df=4)",
            copula_formula="~ batch",
            device="cpu",
        )
        torch.manual_seed(0)
        sim.fit(data, **{**FIT, "chunk_size": chunk_size, "val_frac": 0})
        fits[chunk_size] = sim
    preloaded, chunked = fits[None], fits[7]

    for component in ("marginal", "copula"):
        first, second = getattr(preloaded, component).design, getattr(chunked, component).design
        assert first.predictor_names == second.predictor_names
        for key, matrix in first.matrices(data.obs).items():
            np.testing.assert_allclose(second.matrices(data.obs)[key], matrix, atol=1e-12)
    for name, value in preloaded.predict().items():
        np.testing.assert_allclose(chunked.predict()[name], value, rtol=1e-4)


@pytest.mark.parametrize("storage", BACKED)
def test_backed_spatial_fit_matches_memory(spatial_counts, storage, tmp_path):
    memory = spatial_counts[:, [2, 0, 1]].copy()
    reference_values = memory.X.copy()
    query = memory.obs.copy()
    query[["spatial1", "spatial2"]] += 0.25
    memory_sim = fit_spatial(memory)
    memory_prediction = memory_sim.predict(query)
    with backed_copy(memory, storage, tmp_path) as backed:
        backed_sim = fit_spatial(backed)
        backed_prediction = backed_sim.predict(query)
        sample = backed_sim.sample()
        metrics = backed_sim.complexity(backed)

    for name in ("mean", "dispersion"):
        np.testing.assert_allclose(backed_prediction[name], memory_prediction[name],
                                   rtol=1e-5, atol=1e-6)
    for group in memory_sim.parameters["copula"]:
        np.testing.assert_allclose(backed_sim.parameters["copula"][group].cov,
                                   memory_sim.parameters["copula"][group].cov,
                                   rtol=1e-5, atol=1e-6)
    assert np.isfinite(sample.X).all() and (sample.X >= 0).all()
    assert (sample.X <= reference_values.max(axis=0)).all()
    pd.testing.assert_frame_equal(sample.obs, backed.obs)
    assert list(sample.var_names) == list(backed.var_names)
    assert all(np.isfinite(value) for value in metrics.values())
