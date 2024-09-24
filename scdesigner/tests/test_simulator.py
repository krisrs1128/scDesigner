from scdesigner.margins.negative_binomial import NegativeBinomial
from scdesigner.simulator import simulator
from scipy.sparse import csr_matrix
import anndata as ad
import numpy as np
import pandas as pd


def generate_adata(N=100, G=2000):
    """
    Helper function used in tests
    """
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
    adata = ad.AnnData(counts)
    ct = np.random.choice(["B", "T", "Monocyte"], size=(adata.n_obs,))
    adata.obs["cell_type"] = pd.Categorical(ct)
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata


def test_nb_simulator():
    adata = generate_adata()
    sim = simulator(adata, NegativeBinomial("~ cell_type"))
    genes, params = sim.parameters()
    assert all(genes == adata.var_names)
    assert params["B"].shape == (3, adata.n_vars)
    assert params["A"].shape == (1, adata.n_vars)


def test_nb_predict():
    adata = generate_adata()
    sim = simulator(adata, NegativeBinomial("~ cell_type"))
    y_hat = sim.predict(adata.obs)
    assert np.all(np.sort(list(y_hat.keys())) == ["alpha", "mu"])
    assert y_hat["alpha"].shape == (adata.n_obs, adata.n_vars)
    assert y_hat["mu"].shape == (adata.n_obs, adata.n_vars)
    assert np.all(y_hat["mu"] > 0)
    assert np.all(y_hat["alpha"] > 0)
